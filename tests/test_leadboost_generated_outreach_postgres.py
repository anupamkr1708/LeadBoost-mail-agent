"""
Real-PostgreSQL race tests for POST /integrations/leadboost/outreach-requests (C9.2).

Skipped (never faked with SQLite) unless POSTGRES_TEST_URL points at a
reachable PostgreSQL -- same harness/contract as test_external_dispatch_postgres.py.
Dedicated test database only: it is TRUNCATEd.

Each racer is its own thread with its own engine-backed session, calling the
endpoint handler directly. M2-B: intake makes no LLM call, so the barrier now
sits right after the idempotency lookup -- every racer has seen "no such
dispatch" before any of them commits -- which keeps the final
ExternalDispatch commit race real, not an accident of timing. Generation is a
separate worker (mail/outreach_generation_worker.py), raced separately below
(FOR UPDATE SKIP LOCKED). The LLM stub is local to this module (the shared fake
never infers from prompts, and context-isolation needs a response that depends
on which request's context the prompt carries); it makes no network call.
"""

from __future__ import annotations

import os
import threading

import pytest

pytest.importorskip("psycopg2", reason="psycopg2 not installed -- cannot test real PostgreSQL")

from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

POSTGRES_TEST_URL = os.environ.get("POSTGRES_TEST_URL")


def _pg_available() -> bool:
    if not POSTGRES_TEST_URL:
        return False
    try:
        eng = create_engine(POSTGRES_TEST_URL)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


pytestmark.append(
    pytest.mark.skipif(not _pg_available(), reason="POSTGRES_TEST_URL not set or PostgreSQL unreachable")
)

from mailer_agent.api import integrations as integ  # noqa: E402
from mailer_agent.api import integrations_generated as gen  # noqa: E402
from mailer_agent.config import get_settings  # noqa: E402
from mailer_agent.llm import provider_v2  # noqa: E402
from mailer_agent.llm.provider_v2 import LLMJsonResult  # noqa: E402
from mailer_agent.mail import outreach_generation_worker as gw  # noqa: E402
from mailer_agent.mail.exact_message import evaluate_exact_message_grounding  # noqa: E402
from mailer_agent.models import (  # noqa: E402
    Base,
    Campaign,
    Contact,
    ExternalDispatch,
    Message,
)
from mailer_agent.schemas import LeadBoostOutreachRequestIn  # noqa: E402
from tests.dispatch_support import seed_mailbox  # noqa: E402

ORG = "org-race"
VP_A = "We cut manual invoice reconciliation time by 40% for finance teams."
BODY_A = "Hi Jane,\n\nWe cut manual invoice reconciliation time by 40% for finance teams.\n\nBest"
VP_B = "We help support teams cut ticket backlog by 65% with automatic triage."
BODY_B = "Hi Sam,\n\nWe help support teams cut ticket backlog by 65% with automatic triage.\n\nBest"

_TABLES = "external_dispatches, mailboxes, messages, contacts, campaigns, suppression_list"


@pytest.fixture()
def factory():
    eng = create_engine(POSTGRES_TEST_URL, pool_size=30)
    Base.metadata.create_all(bind=eng)
    with eng.begin() as c:
        c.execute(text(f"TRUNCATE {_TABLES} RESTART IDENTITY CASCADE"))
    sm = sessionmaker(bind=eng)
    with sm() as s:
        seed_mailbox(s, org=ORG, email="outreach@mailer.example.com")   # M2-A: intake needs exactly one ACTIVE mailbox
        s.commit()
    yield sm
    with eng.begin() as c:
        c.execute(text(f"TRUNCATE {_TABLES} RESTART IDENTITY CASCADE"))
    eng.dispose()


@pytest.fixture(autouse=True)
def _sender(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "leadboost_integration_sender_email", "outreach@mailer.example.com")


class _Gate:
    """LLM stub: optionally parks every caller on a barrier (so all racers are
    mid-generation together), then answers from whichever offer the prompt carries."""

    def __init__(self, parties: int | None):
        self.barrier = threading.Barrier(parties) if parties and parties > 1 else None
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, system_prompt, human_prompt, **kw):
        with self._lock:
            self.calls += 1
        if self.barrier is not None:
            self.barrier.wait(timeout=30)
        body = BODY_B if VP_B in human_prompt else BODY_A
        assert (VP_A in human_prompt) != (VP_B in human_prompt), "prompt must carry exactly one request's offer"
        return LLMJsonResult(
            data={"subject": "Hello", "body": body, "reasoning": "stub"},
            model_used="stub", requested_model="stub", attempts=1, used_fallback=False,
            response_mode="json_object",
        )


@pytest.fixture()
def llm(monkeypatch):
    def install(parties=None):
        gate = _Gate(parties)
        monkeypatch.setattr(provider_v2, "call_llm_json", gate)
        monkeypatch.setattr(provider_v2, "is_llm_available", lambda: True)
        return gate

    return install


def _payload(key="idem-1", action="481", email="jane@acme.example.com", name="Jane", vp=VP_A):
    return LeadBoostOutreachRequestIn(
        external_action_id=action, idempotency_key=key,
        recipient={"email": email, "name": name},
        context={"value_proposition": vp, "recipient_facts": []},
    )


def _call(factory, payload, org=ORG):
    s = factory()
    try:
        return gen.create_leadboost_generated_outreach(payload=payload, org_id=org, db=s)
    finally:
        s.close()


def _race(fn, n):
    barrier = threading.Barrier(n)
    results, errors = [None] * n, []

    def target(i):
        try:
            barrier.wait(timeout=30)
            results[i] = fn(i)
        except HTTPException as exc:
            results[i] = exc
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=target, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)
    assert not errors, errors
    return results


@pytest.fixture()
def race_after_lookup(monkeypatch):
    """Hold each racer (once, per thread) right after its idempotency lookup."""
    def install(parties):
        barrier = threading.Barrier(parties)
        local = threading.local()
        real = gen._find_dispatch

        def find(db, org_id, key):
            found = real(db, org_id, key)
            if not getattr(local, "waited", False):
                local.waited = True
                barrier.wait(timeout=30)
            return found

        monkeypatch.setattr(gen, "_find_dispatch", find)

    return install


def _generate(factory, n=50, worker="gen-pg"):
    return gw.run_generation_cycle(
        session_factory=factory, worker_id=worker, runtime=gw.DispatchRuntime(), max_items=n
    )


def _counts(factory):
    with factory() as s:
        return {
            "campaigns": s.query(Campaign).count(),
            "contacts": s.query(Contact).count(),
            "messages": s.query(Message).count(),
            "dispatches": s.query(ExternalDispatch).count(),
        }


# --------------------------------------------------- A. same idempotency key

def test_concurrent_same_idempotency_key_yields_one_dispatch_and_no_llm_call(factory, llm, race_after_lookup):
    n = 6
    gate = llm()
    race_after_lookup(n)                       # all n have passed the lookup when they race to commit
    results = _race(lambda i: _call(factory, _payload()), n)

    assert all(not isinstance(r, HTTPException) for r in results), results
    refs = {r.mailing_agent_reference for r in results}
    assert len(refs) == 1                      # every caller converged on the same operation
    assert gate.calls == 0                     # acceptance never touches the LLM
    c = _counts(factory)
    assert c["messages"] == 0 and c["dispatches"] == 1 and c["campaigns"] == 1 and c["contacts"] == 1
    with factory() as s:
        d = s.query(ExternalDispatch).one()
        assert d.public_reference in refs and d.message_id is None and d.mailbox_id is not None
        assert d.state == "queued"
    # then generation produces exactly one Message
    assert [r.outcome for r in _generate(factory)] == ["generated"] and gate.calls == 1
    assert _counts(factory)["messages"] == 1


def test_concurrent_same_key_conflicting_operation_is_409_and_creates_nothing_extra(factory, llm, race_after_lookup):
    llm()
    race_after_lookup(2)
    results = _race(
        lambda i: _call(factory, _payload(action="481" if i == 0 else "999")), 2
    )
    ok = [r for r in results if not isinstance(r, HTTPException)]
    bad = [r for r in results if isinstance(r, HTTPException)]
    assert len(ok) == 1 and len(bad) == 1 and bad[0].status_code == 409
    c = _counts(factory)
    assert c["messages"] == 0 and c["dispatches"] == 1


def test_replay_after_acceptance_is_a_noop_without_llm(factory, llm):
    gate = llm()
    first = _call(factory, _payload())
    again = _call(factory, _payload())
    assert again.mailing_agent_reference == first.mailing_agent_reference
    assert gate.calls == 0
    c = _counts(factory)
    assert c["messages"] == 0 and c["dispatches"] == 1


# --------------------------------------------------- B. campaign get-or-create

def test_concurrent_first_use_creates_exactly_one_integration_campaign(factory):
    n = 8

    def go(i):
        s = factory()
        try:
            return integ._get_or_create_integration_campaign(s, ORG).id
        finally:
            s.close()

    ids = _race(go, n)
    assert len(set(ids)) == 1
    with factory() as s:
        rows = s.query(Campaign).filter(Campaign.organization_id == ORG,
                                        Campaign.integration_source == "leadboost").all()
        assert len(rows) == 1


def test_concurrent_distinct_requests_still_share_one_campaign_and_one_contact_per_recipient(factory, llm, race_after_lookup):
    n = 6
    llm()
    race_after_lookup(n)
    _race(lambda i: _call(factory, _payload(key=f"k-{i}", action=f"a-{i}")), n)   # same recipient, n operations
    c = _counts(factory)
    assert c["campaigns"] == 1 and c["contacts"] == 1
    assert c["messages"] == 0 and c["dispatches"] == n
    assert len(_generate(factory)) == n and _counts(factory)["messages"] == n


# --------------------------------------------------- C. contact get-or-create

def test_concurrent_requests_for_same_recipient_create_exactly_one_contact(factory):
    with factory() as s:
        campaign_id = integ._get_or_create_integration_campaign(s, ORG).id
    n = 8

    def go(i):
        s = factory()
        try:
            return integ._get_or_create_integration_contact(
                s, campaign_id, "jane@acme.example.com", "Jane").id
        finally:
            s.close()

    ids = _race(go, n)
    assert len(set(ids)) == 1
    with factory() as s:
        assert s.query(Contact).filter(Contact.campaign_id == campaign_id).count() == 1


# --------------------------------------------------- D. context isolation

def test_concurrent_requests_a_and_b_are_grounded_by_their_own_context(factory, llm, race_after_lookup):
    llm()
    race_after_lookup(2)
    results = _race(lambda i: _call(
        factory,
        _payload(key="A", action="1", email="jane@acme.example.com", name="Jane", vp=VP_A) if i == 0
        else _payload(key="B", action="2", email="sam@globex.example.com", name="Sam", vp=VP_B),
    ), 2)
    assert all(not isinstance(r, HTTPException) for r in results), results
    assert [r.outcome for r in _generate(factory)] == ["generated", "generated"]   # prompts carry ONE offer each (stub asserts)

    with factory() as s:
        da = s.query(ExternalDispatch).filter_by(idempotency_key="A").one()
        db_ = s.query(ExternalDispatch).filter_by(idempotency_key="B").one()
        ma, mb = s.get(Message, da.message_id), s.get(Message, db_.message_id)
        ca, cb, camp = s.get(Contact, da.contact_id), s.get(Contact, db_.contact_id), s.get(Campaign, da.campaign_id)
        assert ma.body == BODY_A and mb.body == BODY_B          # each message generated from its own context
        assert da.grounding_context["value_prop"] == VP_A and db_.grounding_context["value_prop"] == VP_B
        assert not evaluate_exact_message_grounding(ma, ca, camp, da.grounding_context).blocked
        assert not evaluate_exact_message_grounding(mb, cb, camp, db_.grounding_context).blocked
        # cross-grounding must fail: A's 40% claim is not in B's context and vice versa
        assert evaluate_exact_message_grounding(ma, ca, camp, db_.grounding_context).blocked
        assert evaluate_exact_message_grounding(mb, cb, camp, da.grounding_context).blocked
        # nothing request-specific leaked into the shared rows
        assert camp.proof_points is None and ca.context_notes is None and cb.context_notes is None
        assert VP_A not in (camp.value_prop or "") and VP_B not in (camp.value_prop or "")


def test_a_then_b_on_the_same_contact_keeps_each_dispatch_snapshot(factory, llm):
    llm()
    _call(factory, _payload(key="A", action="1", vp=VP_A))
    _call(factory, _payload(key="B", action="2", vp=VP_B))        # same recipient, different offer
    assert len(_generate(factory)) == 2
    with factory() as s:
        da = s.query(ExternalDispatch).filter_by(idempotency_key="A").one()
        db_ = s.query(ExternalDispatch).filter_by(idempotency_key="B").one()
        assert da.contact_id == db_.contact_id
        assert da.grounding_context["value_prop"] == VP_A       # B did not rewrite A's snapshot
        assert db_.grounding_context["value_prop"] == VP_B
        assert s.query(Message).count() == 2


# --------------------------------------------------- atomicity

def test_dispatch_with_its_snapshot_is_created_in_one_commit(factory, llm, monkeypatch):
    """If the final commit fails, no dispatch (and so no snapshot) survives."""
    llm()
    from sqlalchemy.orm import Session
    real_commit = Session.commit

    def flaky_commit(self):
        # campaign + contact get-or-create commit first; fail the FINAL commit only
        if any(isinstance(o, ExternalDispatch) for o in self.new):
            raise RuntimeError("boom at final commit")
        return real_commit(self)

    monkeypatch.setattr(Session, "commit", flaky_commit)
    with pytest.raises(RuntimeError, match="boom"):
        _call(factory, _payload())
    monkeypatch.setattr(Session, "commit", real_commit)
    c = _counts(factory)
    assert c["messages"] == 0 and c["dispatches"] == 0
    # and a clean retry with the same key then succeeds with exactly one dispatch
    _call(factory, _payload())
    c = _counts(factory)
    assert c["messages"] == 0 and c["dispatches"] == 1


# --------------------------------------------------- E. generation workers (M2-B)

def test_concurrent_generation_workers_generate_each_dispatch_exactly_once(factory, llm):
    n, workers = 12, 4
    gate = llm()
    for i in range(n):
        _call(factory, _payload(key=f"k-{i}", action=f"a-{i}"))
    assert _counts(factory)["messages"] == 0

    results = _race(lambda i: _generate(factory, worker=f"gen-{i}"), workers)
    flat = [r for batch in results for r in batch]

    assert len(flat) == n and {r.outcome for r in flat} == {"generated"}
    assert len({r.dispatch_id for r in flat}) == n          # no dispatch processed twice
    assert gate.calls == n                                   # SKIP LOCKED: one LLM call per dispatch, ever
    with factory() as s:
        rows = s.query(ExternalDispatch).all()
        assert {d.state for d in rows} == {"queued"} and all(d.claimed_by is None for d in rows)
        assert len({d.message_id for d in rows}) == n and None not in {d.message_id for d in rows}
        assert s.query(Message).count() == n


def test_generating_rows_are_invisible_to_a_second_claimer_on_postgres(factory, llm):
    from mailer_agent.followup import work_claiming as wc

    llm()
    _call(factory, _payload())
    s1, s2 = factory(), factory()
    try:
        assert wc.claim_next_generation_dispatch(s1, "gen-a") is not None   # lock held, uncommitted
        assert wc.claim_next_generation_dispatch(s2, "gen-b") is None       # SKIP LOCKED: not blocked, not duplicated
        s1.commit()
        assert wc.claim_next_generation_dispatch(s2, "gen-b") is None       # now GENERATING, no longer QUEUED
    finally:
        s1.close()
        s2.close()
