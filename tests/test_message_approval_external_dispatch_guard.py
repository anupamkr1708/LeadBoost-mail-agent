"""
Ownership boundary: a Message linked to an ACTIVE LeadBoost ExternalDispatch
(QUEUED or SENDING) belongs to the async worker and must not be deliverable
through the legacy human-approval endpoint.

Why it matters: the integration path creates the Message as DRAFT and the
worker only flips it to SENDING when it claims the row, so while the dispatch
is QUEUED the message looks exactly like an ordinary draft to
POST /messages/{id}/approve. Without this guard a person (or a script) could
send it out-of-band, then the worker would find it no longer sendable.

The guard is deliberately narrow: only the two ACTIVE states block; terminal
dispatches (SENT/FAILED/UNKNOWN) leave the endpoint's existing behaviour
exactly as it was, and ordinary drafts are unaffected.

API tests use per-request sessions on a file-backed SQLite so the async
worker and the HTTP layer see each other's COMMITTED state, as in production.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailer_agent.api import messages as messages_module
from mailer_agent.api.deps import get_current_org_id, require_api_key
from mailer_agent.api.main import app
from mailer_agent.db import get_db
from mailer_agent.mail import external_dispatch_worker as w
from mailer_agent.mail.sender import SendOutcome, SendResult
from mailer_agent.models import (
    Base,
    Campaign,
    Contact,
    ExternalDispatch,
    ExternalDispatchState as S,
    Message,
    MessageStatus,
)
from tests.dispatch_support import FakeSender, TrackingFactory, age, seed_dispatch

ORG_A, ORG_B = "org-a", "org-b"
OWNED_DETAIL = "external dispatch workflow"
CLEAN_BODY = "Hi Jane,\n\nWorth a quick chat?\n\nBest"


class _Org:
    org_id = ORG_A


@pytest.fixture()
def sm(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path/'g.db'}", connect_args={"check_same_thread": False, "timeout": 15})
    Base.metadata.create_all(bind=eng)
    yield sessionmaker(bind=eng)
    eng.dispose()


@pytest.fixture()
def org():
    return _Org()


@pytest.fixture()
def client(sm, org):
    def _get_db():
        s = sm()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_current_org_id] = lambda: org.org_id
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


class _Sends:
    """Legacy endpoint's send_email: records calls, always reports SENT."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return SendResult(True, "<legacy@x>", None, SendOutcome.SENT)


@pytest.fixture()
def legacy_send(monkeypatch):
    f = _Sends()
    monkeypatch.setattr(messages_module, "send_email", f)
    monkeypatch.setattr(messages_module.settings, "live_sending_enabled", True)
    return f


def _seed(sm, **kw):
    kw.setdefault("body", CLEAN_BODY)
    with sm() as s:
        d = seed_dispatch(s, **kw)
        return d.id, d.message_id


def _msg_state(sm, mid):
    with sm() as s:
        m = s.get(Message, mid)
        return (m.status, m.subject, m.body, m.message_id_header, m.error_message)


def _dispatch_state(sm, did):
    with sm() as s:
        return s.get(ExternalDispatch, did).state


def _approve(client, mid):
    return client.post(f"/messages/{mid}/approve")


# ------------------------------------------------------- active states block

def test_queued_leadboost_message_cannot_be_approved_or_sent(sm, client, legacy_send):
    did, mid = _seed(sm)
    before = _msg_state(sm, mid)

    r = _approve(client, mid)

    assert r.status_code == 409 and OWNED_DETAIL in r.json()["detail"]
    assert legacy_send.calls == []                       # nothing was sent
    assert _msg_state(sm, mid) == before                 # message untouched (still DRAFT, exact text)
    assert before[0] == MessageStatus.DRAFT.value
    assert _dispatch_state(sm, did) == S.QUEUED.value    # dispatch untouched, still worker-owned


def test_sending_leadboost_message_is_reported_as_owned_not_as_a_generic_non_draft(sm, client, legacy_send):
    did, mid = _seed(sm, state=S.SENDING.value, claimed_by="w-1", claimed_at=age(5))
    with sm() as s:
        s.get(Message, mid).status = MessageStatus.SENDING.value
        s.commit()

    r = _approve(client, mid)

    assert r.status_code == 409 and OWNED_DETAIL in r.json()["detail"]
    assert legacy_send.calls == []
    assert _dispatch_state(sm, did) == S.SENDING.value


def test_guard_holds_even_when_live_sending_is_disabled(sm, client, monkeypatch):
    """The refusal is about ownership, not the live-send gate."""
    monkeypatch.setattr(messages_module.settings, "live_sending_enabled", False)
    _, mid = _seed(sm)
    r = _approve(client, mid)
    assert r.status_code == 409 and OWNED_DETAIL in r.json()["detail"]


def test_repeated_approval_attempts_never_send_and_never_change_anything(sm, client, legacy_send):
    did, mid = _seed(sm)
    before = _msg_state(sm, mid)
    for _ in range(3):
        assert _approve(client, mid).status_code == 409
    assert legacy_send.calls == [] and _msg_state(sm, mid) == before
    assert _dispatch_state(sm, did) == S.QUEUED.value


def test_other_tenants_still_get_404_not_the_ownership_message(sm, client, org, legacy_send):
    _, mid = _seed(sm)                      # belongs to org-a
    org.org_id = ORG_B
    r = _approve(client, mid)
    assert r.status_code == 404             # unchanged tenant isolation; no info leaked
    assert OWNED_DETAIL not in r.text


# --------------------------------------------- terminal states: unchanged

@pytest.mark.parametrize(
    "dispatch_state, msg_status",
    [
        (S.SENT, MessageStatus.SENT),
        (S.FAILED, MessageStatus.FAILED),
        (S.UNKNOWN, MessageStatus.UNKNOWN),
    ],
)
def test_terminal_dispatch_leaves_the_legacy_response_exactly_as_before(sm, client, legacy_send, dispatch_state, msg_status):
    did, mid = _seed(sm, state=dispatch_state.value)
    with sm() as s:
        s.get(Message, mid).status = msg_status.value      # what the worker leaves behind
        s.commit()

    r = _approve(client, mid)

    # Exactly the pre-existing behaviour: not a draft -> 400. Not the new 409.
    assert r.status_code == 400
    assert r.json()["detail"] == f"Message is not a draft (status={msg_status.value})"
    assert legacy_send.calls == []


def test_guard_is_keyed_to_active_dispatch_state_only(sm, client, legacy_send):
    """A DRAFT message whose dispatch is terminal is not blocked by the guard:
    the endpoint's own pre-existing rules apply (here: it proceeds and sends)."""
    _, mid = _seed(sm, state=S.FAILED.value)               # terminal dispatch, message still DRAFT
    r = _approve(client, mid)
    assert r.status_code == 200 and r.json()["status"] == "sent"
    assert len(legacy_send.calls) == 1


def test_ordinary_draft_without_any_dispatch_still_approves_normally(sm, client, legacy_send):
    with sm() as s:
        camp = Campaign(name="c", organization_id=ORG_A, sender_name="S", sender_org="O",
                        sender_email="s@example.org", value_prop="vp")
        s.add(camp); s.flush()
        contact = Contact(campaign_id=camp.id, name="J", email="j@example.com")
        s.add(contact); s.flush()
        msg = Message(contact_id=contact.id, direction="outbound", subject="Hi",
                      body=CLEAN_BODY, status=MessageStatus.DRAFT.value)
        s.add(msg); s.commit()
        mid = msg.id

    r = _approve(client, mid)

    assert r.status_code == 200 and r.json()["status"] == "sent"
    assert len(legacy_send.calls) == 1 and legacy_send.calls[0]["to_email"] == "j@example.com"


def test_unrelated_messages_in_the_same_campaign_are_not_blocked(sm, client, legacy_send):
    """Only the message the dispatch points at is owned -- not its siblings."""
    did, owned_mid = _seed(sm)
    with sm() as s:
        contact_id = s.get(ExternalDispatch, did).contact_id
        sibling = Message(contact_id=contact_id, direction="outbound", subject="Other",
                          body=CLEAN_BODY, status=MessageStatus.DRAFT.value)
        s.add(sibling); s.commit()
        sibling_id = sibling.id

    assert _approve(client, owned_mid).status_code == 409
    assert _approve(client, sibling_id).status_code == 200


# ------------------------------------------------- async worker is unchanged

@pytest.fixture()
def worker_live(monkeypatch):
    monkeypatch.setattr(w.settings, "live_sending_enabled", True)


def test_worker_still_sends_the_exact_message_after_blocked_approval_attempts(sm, client, legacy_send, worker_live, monkeypatch):
    did, mid = _seed(sm, subject="Exact subject", body=CLEAN_BODY)
    for _ in range(2):
        assert _approve(client, mid).status_code == 409
    fake = FakeSender()
    monkeypatch.setattr(w, "send_email", fake)

    res = w.process_next_external_dispatch(session_factory=TrackingFactory(sm), worker_id="w-1", runtime=w.DispatchRuntime())

    assert res.outcome == "sent" and res.persisted is True
    assert len(fake.calls) == 1 and legacy_send.calls == []          # exactly one send, via the worker
    assert (fake.calls[0]["subject"], fake.calls[0]["body_text"]) == ("Exact subject", CLEAN_BODY)
    status, subject, body, msg_id, _ = _msg_state(sm, mid)
    assert (status, subject, body) == (MessageStatus.SENT.value, "Exact subject", CLEAN_BODY)
    assert msg_id == res.message_id_header == fake.calls[0]["message_id_header"]
    assert _dispatch_state(sm, did) == S.SENT.value


def test_approval_attempted_while_the_worker_is_mid_smtp_is_refused_and_does_not_disturb_it(sm, client, legacy_send, worker_live, monkeypatch):
    did, mid = _seed(sm)
    seen = {}

    def approve_during_smtp(kw):
        r = _approve(client, mid)                     # message is SENDING, dispatch is SENDING
        seen["status"], seen["detail"] = r.status_code, r.json()["detail"]

    fake = FakeSender(on_call=approve_during_smtp)
    monkeypatch.setattr(w, "send_email", fake)

    res = w.process_next_external_dispatch(session_factory=TrackingFactory(sm), worker_id="w-1", runtime=w.DispatchRuntime())

    assert seen["status"] == 409 and OWNED_DETAIL in seen["detail"]
    assert res.outcome == "sent" and res.persisted is True
    assert len(fake.calls) == 1 and legacy_send.calls == []
    assert _dispatch_state(sm, did) == S.SENT.value


@pytest.mark.parametrize(
    "outcome, expected",
    [(SendOutcome.FAILED, S.FAILED), (SendOutcome.UNKNOWN, S.UNKNOWN)],
)
def test_worker_failure_and_unknown_outcomes_are_unchanged_and_approval_does_not_reopen_them(sm, client, legacy_send, worker_live, monkeypatch, outcome, expected):
    did, mid = _seed(sm)
    fake = FakeSender(outcome=outcome, error="boom")
    monkeypatch.setattr(w, "send_email", fake)
    w.process_next_external_dispatch(session_factory=TrackingFactory(sm), worker_id="w-1", runtime=w.DispatchRuntime())
    assert _dispatch_state(sm, did) == expected.value

    r = _approve(client, mid)                          # terminal now: legacy behaviour, cannot be re-sent
    assert r.status_code == 400 and "not a draft" in r.json()["detail"]
    assert legacy_send.calls == [] and len(fake.calls) == 1
