"""
Regression tests: responder fabrication is hard-blocked, and a draft held
for grounding review parks its contact instead of leaving it
scheduler-eligible.

Background
----------
A real GPT-OSS-120B responder call once produced "We ran a similar local
integration test last month and cut manual outreach time by 70% without
sacrificing personalization." None of that was in the approved campaign
facts. Deterministic grounding correctly hard-blocked it, but the
engine's hold branches left the contact in NEW/ACTIVE with a due
``next_action_at``, so the next scheduler cycle would claim it again and
generate yet another draft.

No live LLM calls: the fake provider from conftest returns only what a
test explicitly queues.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailer_agent.followup.engine_v2 import IntegratedFollowUpEngine
from mailer_agent.followup.work_claiming import claim_due_contacts
from mailer_agent.llm.agent import draft_message
from mailer_agent.models import (
    Base,
    Campaign,
    Contact,
    ContactStatus,
    Message,
    MessageDirection,
    MessageStatus,
    MessageType,
)
from mailer_agent.utils.datetime_utils import utcnow

FABRICATED_BODY = (
    "Hi Sam,\n\n"
    "We ran a similar local integration test last month and cut manual "
    "outreach time by 70% without sacrificing personalization.\n\n"
    "Worth a short call?\n\nJordan"
)
CLEAN_BODY = (
    "Hi Sam,\n\n"
    "We help sales teams automate outreach with intelligent AI agents.\n\n"
    "Worth a short call?\n\nJordan"
)
VALUE_PROP = "We help sales teams automate outreach with intelligent AI agents."
PROOF_POINTS = "Controlled local integration test."
CONTEXT_NOTES = "Controlled SMTP and IMAP integration test. Recipient is controlled by the system owner."


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture(autouse=True)
def _never_send(monkeypatch):
    """A grounding hold must never reach SMTP."""
    import mailer_agent.followup.engine_v2 as engine_mod

    def _boom(*a, **kw):  # pragma: no cover - only runs on a regression
        raise AssertionError("send_email must not be called for a grounding-held draft")

    monkeypatch.setattr(engine_mod, "send_email", _boom)


def _campaign(db) -> Campaign:
    c = Campaign(
        name="Live test", organization_id="default", sender_name="Jordan",
        sender_org="TestCorp", sender_email="jordan@testcorp.example.com",
        value_prop=VALUE_PROP, proof_points=PROOF_POINTS,
    )
    db.add(c)
    db.flush()
    return c


def _due_contact(db, campaign, *, status, email="sam@prospect.example.com") -> Contact:
    c = Contact(
        campaign_id=campaign.id, name="Sam", email=email, company="ProspectCo",
        context_notes=CONTEXT_NOTES, status=status, follow_up_index=0,
        next_action_at=utcnow() - timedelta(minutes=5),
    )
    if status == ContactStatus.ACTIVE.value:
        c.last_outbound_at = utcnow() - timedelta(days=3)
    db.add(c)
    db.flush()
    return c


def _queue_responder(fake_llm, body: str) -> None:
    fake_llm.queue_response({"subject": "Quick question", "body": body, "reasoning": "test"})


def _assert_parked(contact: Contact) -> None:
    assert contact.status == ContactStatus.NEEDS_REVIEW.value
    assert contact.claimed_by is None
    assert contact.claimed_at is None
    assert contact.next_action_at is None


# ---------------------------------------------------------------------------
# 1. Fabricated metric stays hard-blocked
# ---------------------------------------------------------------------------

def test_fabricated_metric_from_responder_is_hard_blocked(db_session, fake_llm):
    campaign = _campaign(db_session)
    contact = _due_contact(db_session, campaign, status=ContactStatus.NEW.value)
    _queue_responder(fake_llm, FABRICATED_BODY)

    draft = draft_message(
        campaign=campaign, contact=contact, action_type="initial_outreach",
        context_transcript="(no messages sent yet)",
    )

    assert draft.source == "llm"
    assert draft.grounding.is_grounded is False
    assert draft.grounding.is_safe_to_send is False
    assert draft.grounding.hard_block is True
    assert any("70" in claim for claim in draft.grounding.unsupported_claims)


def test_responder_prompt_keeps_unknown_facts_unknown(db_session, fake_llm):
    """The prompt the model actually receives forbids inventing facts."""
    campaign = _campaign(db_session)
    campaign.proof_points = None
    contact = _due_contact(db_session, campaign, status=ContactStatus.NEW.value)
    contact.context_notes = None
    _queue_responder(fake_llm, CLEAN_BODY)

    draft_message(
        campaign=campaign, contact=contact, action_type="initial_outreach",
        context_transcript="(no messages sent yet)",
    )

    system_prompt, human_prompt = fake_llm.last_prompts[-1]
    assert "UNKNOWN" in system_prompt
    for forbidden in ("percentages", "case studies", "earlier tests", "pricing", "integrations or capabilities"):
        assert forbidden in system_prompt
    # Missing proof points/contact facts are stated as missing, not left blank.
    assert "No proof points are available" in human_prompt
    assert "do not guess any" in human_prompt


# ---------------------------------------------------------------------------
# 2. Initial outreach grounding hold
# ---------------------------------------------------------------------------

def test_initial_outreach_grounding_hold_parks_contact(db_session, fake_llm):
    campaign = _campaign(db_session)
    contact = _due_contact(db_session, campaign, status=ContactStatus.NEW.value)
    contact.claimed_by, contact.claimed_at = "worker-1", utcnow()
    _queue_responder(fake_llm, FABRICATED_BODY)

    result = IntegratedFollowUpEngine().send_initial_outreach(db_session, contact)

    assert result["action"] == "held_grounding_review"
    msgs = db_session.query(Message).filter_by(contact_id=contact.id).all()
    assert len(msgs) == 1
    assert msgs[0].status == MessageStatus.DRAFT.value
    assert msgs[0].message_type == MessageType.INITIAL.value
    assert msgs[0].direction == MessageDirection.OUTBOUND.value
    assert msgs[0].message_id_header is None
    _assert_parked(contact)
    assert contact.status != ContactStatus.ACTIVE.value  # never treated as SENT


# ---------------------------------------------------------------------------
# 3. Follow-up grounding hold
# ---------------------------------------------------------------------------

def test_followup_grounding_hold_parks_contact(db_session, fake_llm):
    campaign = _campaign(db_session)
    contact = _due_contact(db_session, campaign, status=ContactStatus.ACTIVE.value)
    contact.claimed_by, contact.claimed_at = "worker-1", utcnow()
    _queue_responder(fake_llm, FABRICATED_BODY)

    result = IntegratedFollowUpEngine().send_followup_if_due(db_session, contact)

    assert result["action"] == "held_grounding_review"
    msgs = db_session.query(Message).filter_by(contact_id=contact.id).all()
    assert len(msgs) == 1
    assert msgs[0].status == MessageStatus.DRAFT.value
    assert msgs[0].message_type == MessageType.FOLLOW_UP.value
    _assert_parked(contact)
    assert contact.follow_up_index == 0  # no follow-up counted as sent


# ---------------------------------------------------------------------------
# 4. A held contact cannot be reclaimed by the normal scheduler
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status,run",
    [
        (ContactStatus.NEW.value, lambda e, db, c: e.send_initial_outreach(db, c)),
        (ContactStatus.ACTIVE.value, lambda e, db, c: e.send_followup_if_due(db, c)),
    ],
    ids=["initial_outreach", "follow_up"],
)
def test_grounding_held_contact_is_not_reclaimed(db_session, fake_llm, status, run):
    campaign = _campaign(db_session)
    held = _due_contact(db_session, campaign, status=status, email="held@prospect.example.com")
    control = _due_contact(db_session, campaign, status=status, email="control@prospect.example.com")
    _queue_responder(fake_llm, FABRICATED_BODY)

    run(IntegratedFollowUpEngine(), db_session, held)
    _assert_parked(held)

    # Both scheduler queries (dispatch_new_contacts_job uses NEW,
    # run_followup_cycle uses ACTIVE): only the untouched control is due.
    for scheduler_status in (ContactStatus.NEW.value, ContactStatus.ACTIVE.value):
        claimed = claim_due_contacts(db_session, "worker-2", status=scheduler_status, limit=10)
        assert held.id not in [c.id for c in claimed]
    assert control.claimed_by == "worker-2"  # positive control: the query does find due contacts

    # No second draft was generated for the held contact.
    assert db_session.query(Message).filter_by(contact_id=held.id).count() == 1
    assert fake_llm.call_count == 1


# ---------------------------------------------------------------------------
# 5. Approval-time grounding revalidation is intact for a held draft
# ---------------------------------------------------------------------------

def test_held_draft_still_hard_blocked_at_approval(db_session, fake_llm):
    from fastapi import HTTPException

    from mailer_agent.api.messages import approve_and_send_draft

    campaign = _campaign(db_session)
    contact = _due_contact(db_session, campaign, status=ContactStatus.NEW.value)
    _queue_responder(fake_llm, FABRICATED_BODY)
    IntegratedFollowUpEngine().send_initial_outreach(db_session, contact)
    draft = db_session.query(Message).filter_by(contact_id=contact.id).one()

    with pytest.raises(HTTPException) as exc:
        approve_and_send_draft(draft.id, org_id="default", db=db_session)

    assert exc.value.status_code == 409
    db_session.refresh(draft)
    assert draft.status == MessageStatus.DRAFT.value
