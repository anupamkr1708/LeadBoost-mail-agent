"""
Regression tests for Stage D -- State Reconciliation (spec sections 12-18).

Root problem these guard against: before this, every semantic
extraction (memory/store.py's build_known_facts_context) was folded
into a plain dict keyed by fact TEXT with last-write-wins semantics.
That gave correct behavior for current_solution (a real field, "later
wins") but no real reconciliation for generic facts -- two genuinely
different fact strings just accumulated forever with no notion of one
superseding another, and no provenance (which message, when) rode
along with any of it.

These tests exercise mailer_agent.semantic_models.attach_fact_provenance
and mailer_agent.memory.store.reconcile_known_facts /
build_known_facts_context directly -- no LLM involved, since
reconciliation is entirely deterministic Python operating on already-
classified (already-persisted-as-JSON) semantic_analysis dicts.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailer_agent.memory.store import build_known_facts_context, reconcile_known_facts
from mailer_agent.models import Base, Campaign, Contact, Message, MessageDirection, MessageStatus, MessageType
from mailer_agent.semantic_models import Certainty, SemanticFact, SemanticIntent, attach_fact_provenance


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _make_contact(db_session):
    campaign = Campaign(
        name="Reconciliation Test", organization_id="default",
        sender_name="Jordan", sender_org="TestCorp", sender_email="jordan@testcorp.example.com",
        value_prop="We help teams move faster.", proof_points="Trusted by many teams.",
    )
    db_session.add(campaign)
    db_session.flush()
    contact = Contact(campaign_id=campaign.id, name="Sam", email="sam@prospect.example.com", status="active")
    db_session.add(contact)
    db_session.flush()
    return contact


def _inbound(contact, *, semantic_analysis, body="reply"):
    msg = Message(
        contact_id=contact.id, direction=MessageDirection.INBOUND.value,
        message_type=MessageType.INITIAL.value, subject="Re:", body=body,
        status=MessageStatus.RECEIVED.value, semantic_analysis=semantic_analysis,
    )
    return msg


def _analysis(*, current_solution=None, new_facts=None, unresolved_items=None, contradicted_facts=None):
    """Build a minimal raw semantic_analysis dict, the shape
    serialize_semantic_intent() produces and what's actually stored on
    Message.semantic_analysis."""
    return {
        "semantic_schema_version": "3",
        "current_solution": current_solution,
        "new_facts": new_facts or [],
        "unresolved_items": unresolved_items or [],
        "contradicted_facts": contradicted_facts or [],
    }


# ---------------------------------------------------------------------------
# 1. Provenance stamping (semantic_models.attach_fact_provenance)
# ---------------------------------------------------------------------------

def test_attach_fact_provenance_stamps_all_facts():
    intent = SemanticIntent(
        current_solution=SemanticFact(value="Salesforce", certainty=Certainty.EXPLICIT),
        new_facts=[SemanticFact(value="team is 50 people", certainty=Certainty.EXPLICIT)],
        competitors_mentioned=[SemanticFact(value="HubSpot", certainty=Certainty.STRONGLY_INFERRED)],
    )
    attach_fact_provenance(intent, source_message_id=42, observed_at="2026-01-01T00:00:00+00:00")

    assert intent.current_solution.source_message_id == 42
    assert intent.current_solution.observed_at == "2026-01-01T00:00:00+00:00"
    assert intent.new_facts[0].source_message_id == 42
    assert intent.competitors_mentioned[0].source_message_id == 42


def test_attach_fact_provenance_is_idempotent_never_overwrites():
    """A fact that somehow already has provenance keeps its original
    stamp -- attach_fact_provenance must never clobber an existing value
    with a different one."""
    intent = SemanticIntent(
        current_solution=SemanticFact(value="Salesforce", source_message_id=7, observed_at="2025-01-01T00:00:00+00:00"),
    )
    attach_fact_provenance(intent, source_message_id=999, observed_at="2026-06-06T00:00:00+00:00")

    assert intent.current_solution.source_message_id == 7
    assert intent.current_solution.observed_at == "2025-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# 2. current_solution supersession -- the literal spec section 13 example
# ---------------------------------------------------------------------------

def test_current_solution_supersession_keeps_history(db_session):
    contact = _make_contact(db_session)
    db_session.add(_inbound(
        contact, body="We use Salesforce for this.",
        semantic_analysis=_analysis(
            current_solution={"value": "Salesforce", "certainty": "explicit", "source_message_id": 1, "observed_at": "2026-01-01T00:00:00+00:00"},
        ),
    ))
    db_session.add(_inbound(
        contact, body="Actually, we moved off Salesforce last month.",
        semantic_analysis=_analysis(
            current_solution={
                "value": "none -- evaluating replacement", "certainty": "explicit",
                "supersedes": "Salesforce",
                "source_message_id": 2, "observed_at": "2026-02-01T00:00:00+00:00",
            },
        ),
    ))
    db_session.commit()

    knowledge = reconcile_known_facts(contact)

    assert knowledge.current_solution.value == "none -- evaluating replacement"
    assert knowledge.current_solution.status == "current"
    assert len(knowledge.current_solution_history) == 1
    assert knowledge.current_solution_history[0].value == "Salesforce"
    assert knowledge.current_solution_history[0].status == "superseded"

    text = build_known_facts_context(contact)
    assert "current_solution: none -- evaluating replacement" in text
    assert "Salesforce" in text  # history preserved, not erased
    assert "superseded" in text.lower()


# ---------------------------------------------------------------------------
# 3. Generic fact supersession via explicit `supersedes`
# ---------------------------------------------------------------------------

def test_generic_fact_explicit_supersession(db_session):
    contact = _make_contact(db_session)
    db_session.add(_inbound(
        contact, body="Team is about 12 people right now.",
        semantic_analysis=_analysis(new_facts=[
            {"value": "team size is 12 people", "certainty": "explicit", "source_message_id": 1, "observed_at": "2026-01-01T00:00:00+00:00"},
        ]),
    ))
    db_session.add(_inbound(
        contact, body="We've actually grown to 20 people since we last spoke.",
        semantic_analysis=_analysis(new_facts=[
            {
                "value": "team size is 20 people", "certainty": "explicit",
                "supersedes": "team size is 12 people",
                "source_message_id": 2, "observed_at": "2026-03-01T00:00:00+00:00",
            },
        ]),
    ))
    db_session.commit()

    knowledge = reconcile_known_facts(contact)
    by_value = {f.value: f for f in knowledge.facts}

    assert by_value["team size is 12 people"].status == "superseded"
    assert by_value["team size is 20 people"].status == "current"
    # History is retained, not deleted.
    assert len(knowledge.facts) == 2


# ---------------------------------------------------------------------------
# 4. Two unrelated facts do NOT supersede each other just by arriving later
# ---------------------------------------------------------------------------

def test_unrelated_facts_both_remain_current(db_session):
    contact = _make_contact(db_session)
    db_session.add(_inbound(
        contact, body="Team is 50 people.",
        semantic_analysis=_analysis(new_facts=[
            {"value": "team size is 50 people", "certainty": "explicit", "source_message_id": 1, "observed_at": "2026-01-01T00:00:00+00:00"},
        ]),
    ))
    db_session.add(_inbound(
        contact, body="We're also evaluating two competitors.",
        semantic_analysis=_analysis(new_facts=[
            {"value": "evaluating two competitors", "certainty": "explicit", "source_message_id": 2, "observed_at": "2026-01-02T00:00:00+00:00"},
        ]),
    ))
    db_session.commit()

    knowledge = reconcile_known_facts(contact)
    statuses = {f.value: f.status for f in knowledge.facts}

    assert statuses["team size is 50 people"] == "current"
    assert statuses["evaluating two competitors"] == "current"


# ---------------------------------------------------------------------------
# 5. Exact-duplicate restatement doesn't create a second current entry
# ---------------------------------------------------------------------------

def test_duplicate_restatement_not_double_counted(db_session):
    contact = _make_contact(db_session)
    for i in range(2):
        db_session.add(_inbound(
            contact, body="Team is 50 people.",
            semantic_analysis=_analysis(new_facts=[
                {"value": "team size is 50 people", "certainty": "explicit", "source_message_id": i, "observed_at": "2026-01-01T00:00:00+00:00"},
            ]),
        ))
    db_session.commit()

    knowledge = reconcile_known_facts(contact)
    current = [f for f in knowledge.facts if f.status == "current" and f.value == "team size is 50 people"]

    assert len(current) == 1


# ---------------------------------------------------------------------------
# 6. Provenance survives the reconcile step
# ---------------------------------------------------------------------------

def test_provenance_fields_survive_reconciliation(db_session):
    contact = _make_contact(db_session)
    db_session.add(_inbound(
        contact, body="We use Salesforce.",
        semantic_analysis=_analysis(
            current_solution={
                "value": "Salesforce", "certainty": "explicit",
                "source_message_id": 123, "observed_at": "2026-04-04T12:00:00+00:00",
            },
        ),
    ))
    db_session.commit()

    knowledge = reconcile_known_facts(contact)

    assert knowledge.current_solution.source_message_id == 123
    assert knowledge.current_solution.observed_at == "2026-04-04T12:00:00+00:00"


# ---------------------------------------------------------------------------
# 7. Empty conversation stays a no-op (matches pre-existing behavior)
# ---------------------------------------------------------------------------

def test_no_inbound_messages_yields_empty(db_session):
    contact = _make_contact(db_session)
    db_session.commit()

    assert build_known_facts_context(contact) == ""
    knowledge = reconcile_known_facts(contact)
    assert knowledge.current_solution is None
    assert knowledge.facts == []
