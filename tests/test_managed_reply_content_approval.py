"""
Managed-reply exact-content approval -- the safety suite (SQLite, offline, deterministic: no live LLM, no real SMTP).

Drives the REAL chain wherever it can:

    HTTP accept (in_reply_to) -> real generation worker (deterministic fake LLM) -> AWAITING_APPROVAL
    -> GET draft -> POST release(digest) -> real SMTP worker (recording fake sender) -> SENT

and attacks every seam: tenant, digest, tampering after approval, cancellation, races, failure modes.

Every "nothing was sent" assertion is made against the recording sender, so it proves SMTP was never reached rather
than that a row has a particular state.
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailer_agent.api.deps import get_current_org_id, get_integration_org_id, require_api_key
from mailer_agent.api.main import app
from mailer_agent.config import get_settings
from mailer_agent.db import get_db
from mailer_agent.followup import work_claiming as wc
from mailer_agent.mail import content_digest as cd
from mailer_agent.mail import external_dispatch_worker as w
from mailer_agent.mail import outreach_generation_worker as gw
from mailer_agent.models import (
    Base,
    ExternalDispatch,
    ExternalDispatchState as S,
    Message,
    MessageStatus,
)
from mailer_agent.mail.sender import SendOutcome
from tests.dispatch_support import FakeSender, TrackingFactory, seed_dispatch, seed_org_mailboxes
from tests.test_message_approval_external_dispatch_guard import OWNED_DETAIL

ORG_A, ORG_B = "org-a", "org-b"
LEAD = "lead@example.com"
REAL_OUT = "<out-1@sender.example>"
REAL_IN = "<in-1@prospect.example>"
REQ_URL = "/integrations/leadboost/outreach-requests"
DRAFTS = "/integrations/leadboost/reply-drafts"
VP = "We cut manual invoice reconciliation time by 40% for finance teams."
BODY = (
    "Hi Jane,\n\nWe cut manual invoice reconciliation time by 40% for finance teams.\n\n"
    "Worth a quick chat?\n\nBest,\nTest Sender"
)
KEY = "reply-1"


class _Who:
    org = ORG_A


# ------------------------------------------------------------------ fixtures


@pytest.fixture(autouse=True)
def _sender_identity(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "leadboost_integration_sender_email", "outreach@mailer.example.com")
    monkeypatch.setattr(s, "leadboost_integration_sender_name", "Test Sender")
    monkeypatch.setattr(s, "leadboost_integration_sender_org", "Test Org")


@pytest.fixture()
def sm(tmp_path):
    eng = create_engine(
        f"sqlite:///{tmp_path / 'mr.db'}", connect_args={"check_same_thread": False, "timeout": 15}
    )
    Base.metadata.create_all(bind=eng)
    sm_ = sessionmaker(bind=eng)
    with sm_() as s:
        seed_org_mailboxes(s)
    yield sm_
    eng.dispose()


@pytest.fixture()
def factory(sm):
    return TrackingFactory(sm)


@pytest.fixture()
def who():
    w_ = _Who()
    w_.org = ORG_A
    return w_


@pytest.fixture()
def client(sm, who):
    def _get_db():
        s = sm()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_integration_org_id] = lambda: who.org
    app.dependency_overrides[get_current_org_id] = lambda: who.org
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def smtp(monkeypatch):
    """Recording stand-in for send_email at the worker's call site, with live sending ON so that only the
    approval gates -- not the live-send flag -- can be what stops a send."""
    monkeypatch.setattr(w.settings, "live_sending_enabled", True)
    f = FakeSender()
    monkeypatch.setattr(w, "send_email", f)
    return f


# ------------------------------------------------------------------ helpers


def _world(sm):
    """A prior (already-sent) outbound and one real inbound reply from the lead. Returns the inbound's opaque ref."""
    from mailer_agent.models import Mailbox, Message as M

    with sm() as s:
        d = seed_dispatch(
            s, org=ORG_A, idem="seed-1", email=LEAD, state=S.SENT.value,
            sender_email="outreach@mailer.example.com",
        )
        mailbox_id = s.query(Mailbox).filter(Mailbox.organization_id == ORG_A).one().id
        inbound = M(
            contact_id=d.contact_id, direction="inbound", message_type=None, subject="Hello",
            body="Tell me more.", status="received", mailbox_id=mailbox_id,
            message_id_header=REAL_IN, in_reply_to_header=REAL_OUT, references_header=None,
        )
        s.add(inbound)
        s.commit()
        return inbound.reply_reference


def _reply_req(ref, key=KEY, email=LEAD, action="481"):
    return {
        "external_action_id": action,
        "idempotency_key": key,
        "recipient": {"email": email, "name": "Jane", "title": None, "company": None},
        "context": {"value_proposition": VP, "recipient_facts": []},
        "in_reply_to": {"message_reference": ref},
    }


def _accept(client, fake_llm, ref, key=KEY, body=BODY, queue=True):
    if queue:
        fake_llm.queue_response({"subject": "ignored: subject is forced from the parent", "body": body, "reasoning": "t"})
    r = client.post(REQ_URL, json=_reply_req(ref, key))
    assert r.status_code == 202, r.text
    return r.json()["mailing_agent_reference"]


def _gen(factory, worker="gen-1"):
    return gw.process_next_generation(session_factory=factory, worker_id=worker, runtime=gw.DispatchRuntime())


def _send(factory, worker="snd-1"):
    return w.process_next_external_dispatch(session_factory=factory, worker_id=worker, runtime=w.DispatchRuntime())


def _held(sm, client, fake_llm, factory, key=KEY, body=BODY):
    """Accept and generate one managed reply; returns (ref, draft json). Leaves it AWAITING_APPROVAL."""
    ref = _world(sm)
    _accept(client, fake_llm, ref, key, body)
    assert _gen(factory).outcome == "generated"
    return ref, client.get(f"{DRAFTS}/{key}").json()


def _row(sm, key=KEY):
    with sm() as s:
        d = s.query(ExternalDispatch).filter_by(idempotency_key=key).one()
        return {
            "id": d.id, "state": d.state, "message_id": d.message_id, "flag": d.content_approval_required,
            "approved": d.approved_content_sha256, "approved_at": d.approved_at, "error": d.error_message,
            "claimed_by": d.claimed_by,
        }


def _msg(sm, mid):
    with sm() as s:
        m = s.get(Message, mid)
        return {"status": m.status, "subject": m.subject, "body": m.body, "in_reply_to": m.in_reply_to_header,
                "refs": m.references_header, "mid": m.message_id_header}


def _n_messages(sm):
    with sm() as s:
        return s.query(Message).count()


def _release(client, key=KEY, digest=None):
    return client.post(f"{DRAFTS}/{key}/release", json={"content_sha256": digest})


def _tamper(sm, mid, **fields):
    with sm() as s:
        m = s.get(Message, mid)
        for k, v in fields.items():
            setattr(m, k, v)
        s.commit()


# =========================================================== 1-2: one persisted draft, held, not sendable


def test_managed_reply_generation_persists_one_draft_and_holds_it(sm, client, fake_llm, factory):
    ref = _world(sm)
    before = _n_messages(sm)
    _accept(client, fake_llm, ref)
    assert _row(sm)["flag"] is True, "a reply must be marked as requiring approval at acceptance"
    assert _row(sm)["state"] == S.QUEUED.value and _row(sm)["message_id"] is None

    assert _gen(factory).outcome == "generated"

    row = _row(sm)
    assert row["state"] == S.AWAITING_APPROVAL.value, "must NOT land in a sendable state"
    assert row["message_id"] is not None and row["flag"] is True
    assert row["approved"] is None and row["approved_at"] is None
    assert _n_messages(sm) == before + 1, "exactly one Message created"
    m = _msg(sm, row["message_id"])
    assert m["status"] == MessageStatus.DRAFT.value and m["body"] == BODY
    assert m["in_reply_to"] == REAL_IN, "threading comes from the stored parent"


def test_held_draft_is_never_claimed_or_sent(sm, client, fake_llm, factory, smtp):
    _held(sm, client, fake_llm, factory)
    for _ in range(3):
        assert _send(factory) is None
    with sm() as s:
        assert wc.claim_next_external_dispatch(s, worker_id="x") is None
    assert smtp.calls == []
    assert _row(sm)["state"] == S.AWAITING_APPROVAL.value


def test_generation_makes_exactly_one_llm_call_and_no_smtp(sm, client, fake_llm, factory, smtp):
    _held(sm, client, fake_llm, factory)
    assert fake_llm.call_count == 1 and smtp.calls == []


# =========================================================== 3-5: draft read


def test_draft_read_returns_exact_subject_body_and_stable_digest_and_mutates_nothing(sm, client, fake_llm, factory):
    _, d = _held(sm, client, fake_llm, factory)
    row = _row(sm)
    m = _msg(sm, row["message_id"])
    assert set(d) == {"state", "subject", "body", "content_sha256", "draft_version", "updated_at"}
    assert d["state"] == "awaiting_approval"
    assert d["subject"] == m["subject"] and d["body"] == BODY
    assert d["content_sha256"] == cd.content_sha256(m["subject"], BODY)
    assert d["draft_version"] == 1

    for _ in range(3):
        again = client.get(f"{DRAFTS}/{KEY}").json()
        assert again == d, "repeated reads are identical"
    assert _row(sm) == row and _msg(sm, row["message_id"]) == m, "reads mutate nothing"


def test_draft_read_never_leaks_internal_identifiers_or_headers(sm, client, fake_llm, factory):
    _, d = _held(sm, client, fake_llm, factory)
    row = _row(sm)
    blob = client.get(f"{DRAFTS}/{KEY}").text
    for secret in (REAL_IN, REAL_OUT, "in_reply_to", "references", "grounding", "message_id", "mailbox", "dispatch"):
        assert secret not in blob, f"{secret!r} leaked into the draft response"
    assert f'"id":{row["id"]}' not in blob.replace(" ", "")


def test_draft_read_while_generating_has_no_content(sm, client, fake_llm):
    ref = _world(sm)
    _accept(client, fake_llm, ref)
    d = client.get(f"{DRAFTS}/{KEY}").json()
    assert d["state"] == "generating"
    assert d["subject"] is None and d["body"] is None and d["content_sha256"] is None and d["draft_version"] is None


def test_foreign_tenant_cannot_read_release_or_cancel(sm, client, fake_llm, factory, who, smtp):
    _, d = _held(sm, client, fake_llm, factory)
    before = _row(sm)
    who.org = ORG_B
    r_get = client.get(f"{DRAFTS}/{KEY}")
    r_rel = _release(client, digest=d["content_sha256"])
    r_can = client.post(f"{DRAFTS}/{KEY}/cancel")
    r_missing = client.get(f"{DRAFTS}/no-such-key")
    for r in (r_get, r_rel, r_can):
        assert r.status_code == 404
        assert r.json() == r_missing.json(), "foreign-tenant and nonexistent are indistinguishable"
    assert d["body"] not in r_get.text
    assert _row(sm) == before and smtp.calls == []


def test_ordinary_dispatch_is_not_a_draft(sm, client):
    with sm() as s:
        seed_dispatch(s, org=ORG_A, idem="ordinary", state=S.QUEUED.value)
    assert client.get(f"{DRAFTS}/ordinary").status_code == 404
    assert _release(client, "ordinary", "a" * 64).status_code == 404
    assert client.post(f"{DRAFTS}/ordinary/cancel").status_code == 404
    assert _row(sm, "ordinary")["state"] == S.QUEUED.value


def test_draft_endpoints_fail_closed_without_integration_credentials(sm):
    """No dependency override for the integration org: an anonymous caller must not reach any draft."""
    def _get_db():
        s = sm()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    try:
        c = TestClient(app)
        for r in (
            c.get(f"{DRAFTS}/{KEY}"),
            c.post(f"{DRAFTS}/{KEY}/release", json={"content_sha256": "a" * 64}),
            c.post(f"{DRAFTS}/{KEY}/cancel"),
        ):
            assert r.status_code in (401, 403, 503), r.status_code   # 503: no keys configured -> fails closed
            assert BODY not in r.text
    finally:
        app.dependency_overrides.clear()


# =========================================================== 6-8: digest validation, release, idempotency


@pytest.mark.parametrize("payload", [{}, {"content_sha256": ""}, {"content_sha256": "abc"}, {"content_sha256": "A" * 64},
                                      {"content_sha256": "g" * 64}, {"content_sha256": "a" * 65},
                                      {"content_sha256": 5}, {"content_sha256": "a" * 64, "extra": 1}])
def test_missing_or_malformed_digest_is_rejected_and_changes_nothing(sm, client, fake_llm, factory, smtp, payload):
    _held(sm, client, fake_llm, factory)
    before = _row(sm)
    r = client.post(f"{DRAFTS}/{KEY}/release", json=payload)
    assert r.status_code == 422
    assert _row(sm) == before and smtp.calls == []


def test_wrong_digest_is_a_409_and_changes_nothing(sm, client, fake_llm, factory, smtp):
    _held(sm, client, fake_llm, factory)
    before = _row(sm)
    r = _release(client, digest=cd.content_sha256("Re: x", "some other text"))
    assert r.status_code == 409 and r.json()["detail"] == "draft_changed"
    assert _row(sm) == before and smtp.calls == []


def test_stale_digest_after_the_stored_draft_changed_is_refused(sm, client, fake_llm, factory, smtp):
    _, d = _held(sm, client, fake_llm, factory)
    _tamper(sm, _row(sm)["message_id"], body=BODY + "\nP.S. changed after the user looked")
    r = _release(client, digest=d["content_sha256"])
    assert r.status_code == 409 and r.json()["detail"] == "draft_changed"
    assert _row(sm)["state"] == S.AWAITING_APPROVAL.value and smtp.calls == []


def test_correct_release_records_evidence_and_releases_without_sending_or_regenerating(
    sm, client, fake_llm, factory, smtp
):
    _, d = _held(sm, client, fake_llm, factory)
    msgs, dispatches = _n_messages(sm), None
    with sm() as s:
        dispatches = s.query(ExternalDispatch).count()
    calls_before = fake_llm.call_count

    r = _release(client, digest=d["content_sha256"])

    assert r.status_code == 200 and r.json()["state"] == "queued"
    row = _row(sm)
    assert row["state"] == S.QUEUED.value and row["approved"] == d["content_sha256"] and row["approved_at"] is not None
    assert smtp.calls == [], "release must not send"
    assert fake_llm.call_count == calls_before, "no LLM call after approval"
    assert _n_messages(sm) == msgs, "no second Message"
    with sm() as s:
        assert s.query(ExternalDispatch).count() == dispatches, "no second ExternalDispatch"


def test_repeated_identical_release_is_idempotent(sm, client, fake_llm, factory, smtp):
    _, d = _held(sm, client, fake_llm, factory)
    first = _release(client, digest=d["content_sha256"])
    snapshot = _row(sm)
    for _ in range(3):
        again = _release(client, digest=d["content_sha256"])
        assert again.status_code == 200 and again.json()["state"] == first.json()["state"]
    assert _row(sm) == snapshot
    assert smtp.calls == []


def test_a_different_digest_cannot_reuse_an_existing_approval(sm, client, fake_llm, factory, smtp):
    _, d = _held(sm, client, fake_llm, factory)
    assert _release(client, digest=d["content_sha256"]).status_code == 200
    other = cd.content_sha256(d["subject"], d["body"] + "x")
    r = _release(client, digest=other)
    assert r.status_code == 409 and r.json()["detail"] == "draft_already_released_with_different_content"
    assert _row(sm)["approved"] == d["content_sha256"]


# =========================================================== 9-11: exact send, threading, no regeneration


def test_worker_sends_exactly_the_approved_subject_and_body_with_parent_threading(
    sm, client, fake_llm, factory, smtp
):
    _, d = _held(sm, client, fake_llm, factory)
    assert _release(client, digest=d["content_sha256"]).status_code == 200
    calls_before = fake_llm.call_count

    res = _send(factory)

    assert res is not None and res.outcome == "sent"
    assert len(smtp.calls) == 1
    sent = smtp.calls[0]
    assert sent["subject"] == d["subject"] and sent["body_text"] == d["body"] == BODY
    assert cd.content_sha256(sent["subject"], sent["body_text"]) == d["content_sha256"]
    assert sent["in_reply_to_header"] == REAL_IN, "In-Reply-To is the original inbound Message-ID"
    assert REAL_IN in (sent.get("references_header") or sent["in_reply_to_header"])
    assert sent["message_id_header"] and sent["message_id_header"] != REAL_IN, "the new Message-ID is distinct"
    assert fake_llm.call_count == calls_before, "sending never regenerates"
    assert _row(sm)["state"] == S.SENT.value
    assert client.get(f"{DRAFTS}/{KEY}").json()["state"] == "sent"


def test_message_mutated_after_approval_never_reaches_smtp(sm, client, fake_llm, factory, smtp):
    _, d = _held(sm, client, fake_llm, factory)
    assert _release(client, digest=d["content_sha256"]).status_code == 200
    _tamper(sm, _row(sm)["message_id"], body=BODY + "\nBuy crypto now")

    res = _send(factory)

    assert res is not None and res.outcome == "failed" and res.smtp_attempted is False
    assert smtp.calls == [], "tampered content must never reach SMTP"
    row = _row(sm)
    assert row["state"] == S.FAILED.value
    assert row["error"].startswith("approval_content_mismatch")
    assert "crypto" not in row["error"] and BODY not in row["error"], "the failure reason carries no content"
    # and a second worker pass does not retry it
    assert _send(factory) is None and smtp.calls == []


def test_subject_mutation_after_approval_is_also_caught(sm, client, fake_llm, factory, smtp):
    _, d = _held(sm, client, fake_llm, factory)
    assert _release(client, digest=d["content_sha256"]).status_code == 200
    _tamper(sm, _row(sm)["message_id"], subject="Re: something else")
    res = _send(factory)
    assert res.outcome == "failed" and smtp.calls == []


def test_required_dispatch_without_recorded_approval_is_refused_by_the_worker(sm, smtp, factory):
    """Defence in depth: even if a held row were force-queued, 'approved' is something it must PROVE."""
    with sm() as s:
        seed_dispatch(s, org=ORG_A, idem="forced", state=S.QUEUED.value, content_approval_required=True)
    res = _send(factory)
    assert res.outcome == "failed" and smtp.calls == []
    assert _row(sm, "forced")["error"].startswith("approval_missing")


def test_clearing_the_flag_does_not_bypass_a_recorded_digest(sm, smtp, factory):
    """A row that carries an approved digest is verified whether or not the flag is set."""
    with sm() as s:
        seed_dispatch(
            s, org=ORG_A, idem="flagless", state=S.QUEUED.value, content_approval_required=False,
            approved_content_sha256=cd.content_sha256("Quick question", "different text entirely"),
        )
    res = _send(factory)
    assert res.outcome == "failed" and smtp.calls == []
    assert _row(sm, "flagless")["error"].startswith("approval_content_mismatch")


def test_ordinary_outreach_is_completely_unchanged(sm, client, fake_llm, factory, smtp):
    """No in_reply_to: flag stays FALSE, generation lands in QUEUED (not the hold), the worker sends with no digest."""
    fake_llm.queue_response({"subject": "Quick question", "body": BODY, "reasoning": "t"})
    r = client.post(REQ_URL, json={
        "external_action_id": "9", "idempotency_key": "plain-1",
        "recipient": {"email": "plain@example.com", "name": "Pat", "title": None, "company": None},
        "context": {"value_proposition": VP, "recipient_facts": []},
    })
    assert r.status_code == 202
    assert _row(sm, "plain-1")["flag"] is False
    assert _gen(factory).outcome == "generated"
    row = _row(sm, "plain-1")
    assert row["state"] == S.QUEUED.value and row["message_id"] is not None and row["approved"] is None
    res = _send(factory)
    assert res.outcome == "sent" and len(smtp.calls) == 1 and smtp.calls[0]["body_text"] == BODY


# =========================================================== 16: cancellation


def test_cancelled_draft_can_never_be_released_or_sent(sm, client, fake_llm, factory, smtp):
    _, d = _held(sm, client, fake_llm, factory)
    c = client.post(f"{DRAFTS}/{KEY}/cancel")
    assert c.status_code == 200 and c.json()["state"] == "cancelled"
    assert _row(sm)["state"] == S.CANCELLED.value

    r = _release(client, digest=d["content_sha256"])
    assert r.status_code == 409 and r.json()["detail"] == "draft_cancelled"
    assert _send(factory) is None and smtp.calls == []
    assert _row(sm)["approved"] is None


def test_cancel_is_idempotent(sm, client, fake_llm, factory):
    _held(sm, client, fake_llm, factory)
    first = client.post(f"{DRAFTS}/{KEY}/cancel")
    row = _row(sm)
    for _ in range(3):
        again = client.post(f"{DRAFTS}/{KEY}/cancel")
        assert again.status_code == 200 and again.json()["state"] == first.json()["state"] == "cancelled"
    assert _row(sm) == row


def test_cancel_after_release_is_refused_and_does_not_block_the_send(sm, client, fake_llm, factory, smtp):
    _, d = _held(sm, client, fake_llm, factory)
    assert _release(client, digest=d["content_sha256"]).status_code == 200
    r = client.post(f"{DRAFTS}/{KEY}/cancel")
    assert r.status_code == 409 and r.json()["detail"] == "draft_already_released"
    assert _row(sm)["state"] == S.QUEUED.value
    assert _send(factory).outcome == "sent" and len(smtp.calls) == 1


def test_cancel_before_generation_starts_means_no_llm_and_no_message(sm, client, fake_llm, factory, smtp):
    ref = _world(sm)
    _accept(client, fake_llm, ref)
    before = _n_messages(sm)
    c = client.post(f"{DRAFTS}/{KEY}/cancel")
    assert c.status_code == 200 and c.json()["state"] == "cancelled"
    assert _gen(factory) is None, "a cancelled request is never claimed for generation"
    assert fake_llm.call_count == 0 and _n_messages(sm) == before and smtp.calls == []
    assert c.json()["subject"] is None and c.json()["body"] is None


def test_cancel_during_generation_leaves_no_message_and_nothing_sendable(sm, client, fake_llm, factory, smtp, monkeypatch):
    ref = _world(sm)
    _accept(client, fake_llm, ref)
    did = _row(sm)["id"]
    before = _n_messages(sm)
    real = gw.draft_message

    def draft_then_cancelled(*a, **k):
        out = real(*a, **k)               # the LLM has answered ...
        with sm() as s:                   # ... and the user cancels before the worker can persist it
            assert wc.cancel_unreleased_dispatch(s, organization_id=ORG_A, dispatch_id=did) is True
            s.commit()
        return out

    monkeypatch.setattr(gw, "draft_message", draft_then_cancelled)
    res = _gen(factory)

    assert res.outcome == "lease_lost" and res.persisted is False, "the worker loses its fence"
    assert _n_messages(sm) == before, "no Message may be left behind"
    row = _row(sm)
    assert row["state"] == S.CANCELLED.value and row["message_id"] is None
    assert _send(factory) is None and smtp.calls == []


# =========================================================== 17: races have one atomic winner


def _atomic(sm, fn, **kw):
    with sm() as s:
        ok = fn(s, **kw)
        s.commit()
        return ok


def test_release_then_cancel_release_wins(sm, client, fake_llm, factory):
    _, d = _held(sm, client, fake_llm, factory)
    row = _row(sm)
    kw = {"organization_id": ORG_A, "dispatch_id": row["id"]}
    assert _atomic(sm, wc.release_held_dispatch, message_id=row["message_id"],
                   content_sha256=d["content_sha256"], **kw) is True
    assert _atomic(sm, wc.cancel_unreleased_dispatch, **kw) is False
    assert _row(sm)["state"] == S.QUEUED.value


def test_cancel_then_release_cancel_wins(sm, client, fake_llm, factory):
    _, d = _held(sm, client, fake_llm, factory)
    row = _row(sm)
    kw = {"organization_id": ORG_A, "dispatch_id": row["id"]}
    assert _atomic(sm, wc.cancel_unreleased_dispatch, **kw) is True
    assert _atomic(sm, wc.release_held_dispatch, message_id=row["message_id"],
                   content_sha256=d["content_sha256"], **kw) is False
    after = _row(sm)
    assert after["state"] == S.CANCELLED.value and after["approved"] is None


def test_release_is_pinned_to_tenant_and_message(sm, client, fake_llm, factory):
    _, d = _held(sm, client, fake_llm, factory)
    row = _row(sm)
    assert _atomic(sm, wc.release_held_dispatch, organization_id=ORG_B, dispatch_id=row["id"],
                   message_id=row["message_id"], content_sha256=d["content_sha256"]) is False
    assert _atomic(sm, wc.release_held_dispatch, organization_id=ORG_A, dispatch_id=row["id"],
                   message_id=row["message_id"] + 999, content_sha256=d["content_sha256"]) is False
    assert _row(sm)["state"] == S.AWAITING_APPROVAL.value


def test_concurrent_release_and_cancel_have_exactly_one_winner(sm, client, fake_llm, factory):
    """Two real threads, two real sessions, released together by a barrier. SQLite serialises the writers, so this
    proves the single-UPDATE guard yields exactly one winner and a consistent final state; the row-lock behaviour on
    PostgreSQL is the same statement and is covered by the postgres-marked suites when a server is available."""
    ref = _world(sm)
    _accept(client, fake_llm, ref, KEY)
    assert _gen(factory).outcome == "generated"
    row = _row(sm)
    d = client.get(f"{DRAFTS}/{KEY}").json()
    barrier, results = threading.Barrier(2), {}

    def release():
        barrier.wait()
        results["release"] = _atomic(sm, wc.release_held_dispatch, organization_id=ORG_A, dispatch_id=row["id"],
                                     message_id=row["message_id"], content_sha256=d["content_sha256"])

    def cancel():
        barrier.wait()
        results["cancel"] = _atomic(sm, wc.cancel_unreleased_dispatch, organization_id=ORG_A, dispatch_id=row["id"])

    ts = [threading.Thread(target=release), threading.Thread(target=cancel)]
    [t.start() for t in ts]
    [t.join(timeout=30) for t in ts]
    assert sorted(results.values()) == [False, True], f"exactly one winner, got {results}"
    after = _row(sm)
    if results["release"]:
        assert after["state"] == S.QUEUED.value and after["approved"] == d["content_sha256"]
    else:
        assert after["state"] == S.CANCELLED.value and after["approved"] is None


# =========================================================== 18: failures create no sendable draft


def test_llm_failure_creates_no_draft_and_nothing_sendable(sm, client, fake_llm, factory, smtp, monkeypatch):
    ref = _world(sm)
    _accept(client, fake_llm, ref, queue=False)
    before = _n_messages(sm)

    def boom(*a, **k):
        raise RuntimeError("provider is down: secret-token-123")

    monkeypatch.setattr(gw, "draft_message", boom)
    _gen(factory)

    row = _row(sm)
    assert row["state"] == S.FAILED.value and row["message_id"] is None
    assert _n_messages(sm) == before
    d = client.get(f"{DRAFTS}/{KEY}").json()
    assert d["state"] == "failed" and d["subject"] is None and d["body"] is None and d["content_sha256"] is None
    assert "secret-token-123" not in str(d), "no raw error text reaches the caller"
    r = _release(client, digest="a" * 64)
    assert r.status_code == 409 and r.json()["detail"] == "draft_generation_failed"
    assert _send(factory) is None and smtp.calls == []


def test_grounding_hard_block_creates_no_draft_and_nothing_sendable(sm, client, fake_llm, factory, smtp):
    ref = _world(sm)
    ungrounded = "Hi Jane,\n\nWe increased revenue by 300% for every client last quarter.\n\nBest,\nTest Sender"
    _accept(client, fake_llm, ref, body=ungrounded)
    before = _n_messages(sm)
    _gen(factory)
    row = _row(sm)
    assert row["state"] == S.FAILED.value and row["message_id"] is None, row
    assert _n_messages(sm) == before
    assert client.get(f"{DRAFTS}/{KEY}").json()["state"] == "failed"
    assert _send(factory) is None and smtp.calls == []


# =========================================================== 19: ambiguous SMTP is never resent


def test_ambiguous_smtp_stays_unknown_and_is_never_resent(sm, client, fake_llm, factory, monkeypatch):
    monkeypatch.setattr(w.settings, "live_sending_enabled", True)
    amb = FakeSender(outcome=SendOutcome.UNKNOWN, error="timeout after DATA")
    monkeypatch.setattr(w, "send_email", amb)
    _, d = _held(sm, client, fake_llm, factory)
    assert _release(client, digest=d["content_sha256"]).status_code == 200

    assert _send(factory).outcome == "unknown"
    assert _row(sm)["state"] == S.UNKNOWN.value
    for _ in range(3):
        assert _send(factory) is None
    assert len(amb.calls) == 1, "an ambiguous outcome must never be automatically resent"
    assert client.get(f"{DRAFTS}/{KEY}").json()["state"] == "unknown"
    # and release can no longer be replayed into a second send
    assert _release(client, digest=d["content_sha256"]).status_code == 200
    assert _send(factory) is None and len(amb.calls) == 1


# =========================================================== 20: native guard unchanged; legacy-vocabulary


def test_native_approve_endpoint_still_refuses_a_held_managed_reply(sm, client, fake_llm, factory, smtp):
    _held(sm, client, fake_llm, factory)
    row = _row(sm)
    before = _msg(sm, row["message_id"])
    r = client.post(f"/messages/{row['message_id']}/approve")
    assert r.status_code == 409 and OWNED_DETAIL in r.json()["detail"]
    assert _msg(sm, row["message_id"]) == before and before["status"] == MessageStatus.DRAFT.value
    assert _row(sm)["state"] == S.AWAITING_APPROVAL.value and smtp.calls == []


def test_held_reply_reads_as_queued_on_the_unchanged_reconciliation_endpoint(sm, client, fake_llm, factory):
    _held(sm, client, fake_llm, factory)
    body = client.get(f"/integrations/leadboost/outreach-actions/{KEY}").json()
    assert body["state"] == "queued", "an un-upgraded caller must see only the vocabulary it knows"


# =========================================================== preview size: fail closed, never truncate


def test_oversized_draft_is_neither_previewable_nor_releasable(sm, client, fake_llm, factory, smtp):
    _held(sm, client, fake_llm, factory)
    _tamper(sm, _row(sm)["message_id"], body="x" * (cd.MAX_BODY_CHARS + 1))
    before = _row(sm)
    r = client.get(f"{DRAFTS}/{KEY}")
    assert r.status_code == 422 and r.json()["detail"] == "draft_too_large"
    assert "xxxx" not in r.text, "no truncated body is ever returned"
    r2 = _release(client, digest=cd.content_sha256("s", "b"))
    assert r2.status_code == 422 and r2.json()["detail"] == "draft_too_large"
    assert _row(sm) == before and smtp.calls == []


# =========================================================== 21: legacy C9.4 replies (post-migration rows)
#
# A reply accepted BEFORE migration 013 has content_approval_required = FALSE (the column default) but still carries
# the immutable ``reply_parent`` snapshot in grounding_context. LeadBoost reads an HTTP 404 from the cancel endpoint
# as "the Mailer has no such operation" and may then cancel its own action. For a legacy reply that is unsafe: the
# message may be queued, sending, sent, or in an ambiguous SMTP state. These rows are therefore shaped exactly like
# post-migration production rows -- flag false, reply_parent present -- and must be a 409, never a 404, never written.

LEGACY_KEY = "legacy-reply-1"
LEGACY_CODE = "legacy_reply_requires_reconciliation"
LEGACY_CONTEXT = {
    "reply_parent": {
        "message_id": 41,
        "subject": "Re: Hello",
        "in_reply_to_header": REAL_IN,
        "references_header": REAL_OUT + " " + REAL_IN,
    }
}


def _legacy(sm, state, *, key=LEGACY_KEY, org=ORG_A, with_message=True, context=LEGACY_CONTEXT, flag=False):
    """One pre-013 reply as it exists after migration 013 ran: the new flag at its default (false), no recorded
    digest, reply_parent in grounding_context. A SENDING row looks claimed by an old worker."""
    from tests.dispatch_support import age

    claimed = state in (S.SENDING.value, S.GENERATING.value)
    with sm() as s:
        d = seed_dispatch(
            s, org=org, idem=key, state=state, with_message=with_message,
            sender_email="outreach@mailer.example.com",
            claimed_by="old-worker" if claimed else None, claimed_at=age(5) if claimed else None,
            content_approval_required=flag, grounding_context=context,
        )
        return d.id


def _snapshot(sm, key):
    with sm() as s:
        d = s.query(ExternalDispatch).filter_by(idempotency_key=key).one()
        return (
            d.state, d.message_id, d.content_approval_required, d.approved_content_sha256, d.approved_at,
            d.claimed_by, d.claimed_at, d.error_message, d.updated_at, d.grounding_context,
        )


# (label, state, has a generated message). The five the rollout can leave behind, plus two defensive shapes.
LEGACY_UNSAFE = [
    ("sent", S.SENT.value, True),
    ("unknown", S.UNKNOWN.value, True),
    ("sending", S.SENDING.value, True),
    ("queued-with-message", S.QUEUED.value, True),
    ("queued-without-message", S.QUEUED.value, False),
    ("generating", S.GENERATING.value, False),
    ("awaiting-approval-with-flag-cleared", S.AWAITING_APPROVAL.value, True),
]


@pytest.mark.parametrize("label,state,has_message", LEGACY_UNSAFE, ids=[c[0] for c in LEGACY_UNSAFE])
def test_legacy_reply_cancel_is_a_409_never_a_404_and_writes_nothing(sm, client, smtp, label, state, has_message):
    _legacy(sm, state, with_message=has_message)
    before, n_msgs = _snapshot(sm, LEGACY_KEY), _n_messages(sm)

    r = client.post(f"{DRAFTS}/{LEGACY_KEY}/cancel")

    assert r.status_code == 409, f"{label}: {r.status_code} {r.text}"
    assert r.json() == {"detail": LEGACY_CODE}
    assert r.status_code != 404, "an existing operation must never read as 'not found'"
    assert _snapshot(sm, LEGACY_KEY) == before, "a legacy row must not be modified by the cancel endpoint"
    assert _n_messages(sm) == n_msgs and smtp.calls == []


@pytest.mark.parametrize("label,state,has_message", LEGACY_UNSAFE, ids=[c[0] for c in LEGACY_UNSAFE])
def test_legacy_reply_cannot_be_read_or_released_through_the_approval_flow(sm, client, smtp, label, state, has_message):
    _legacy(sm, state, with_message=has_message)
    before = _snapshot(sm, LEGACY_KEY)

    got = client.get(f"{DRAFTS}/{LEGACY_KEY}")
    rel = _release(client, LEGACY_KEY, "a" * 64)

    for r in (got, rel):
        assert r.status_code == 409 and r.json() == {"detail": LEGACY_CODE}, f"{label}: {r.status_code} {r.text}"
    assert _snapshot(sm, LEGACY_KEY) == before, "release must never approve a legacy row"
    assert smtp.calls == []


@pytest.mark.parametrize("state", [S.FAILED.value, S.CANCELLED.value])
def test_legacy_reply_that_provably_never_sent_answers_200_without_content_and_writes_nothing(sm, client, smtp, state):
    _legacy(sm, state, with_message=False)
    before = _snapshot(sm, LEGACY_KEY)

    r = client.post(f"{DRAFTS}/{LEGACY_KEY}/cancel")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == state
    assert body["subject"] is None and body["body"] is None and body["content_sha256"] is None
    assert _snapshot(sm, LEGACY_KEY) == before and smtp.calls == []


def test_legacy_failed_reply_with_a_message_also_returns_no_content(sm, client):
    """A failed legacy dispatch that had a message (a definite SMTP failure) still exposes none of it."""
    _legacy(sm, S.FAILED.value, with_message=True)
    r = client.post(f"{DRAFTS}/{LEGACY_KEY}/cancel")
    assert r.status_code == 200 and r.json()["state"] == "failed"
    assert r.json()["body"] is None and r.json()["subject"] is None


def test_the_atomic_cancel_still_cannot_touch_a_legacy_row(sm):
    """Defence in depth under the endpoint: the one UPDATE that revokes a held draft is restricted to rows that
    require content approval, so even a future caller cannot cancel a legacy queued row through it."""
    did = _legacy(sm, S.QUEUED.value, with_message=False)
    with sm() as s:
        assert wc.cancel_unreleased_dispatch(s, organization_id=ORG_A, dispatch_id=did) is False
        s.rollback()
    assert _row(sm, LEGACY_KEY)["state"] == S.QUEUED.value


def test_legacy_reply_of_another_tenant_is_indistinguishable_from_nonexistent(sm, client, who, smtp):
    _legacy(sm, S.SENT.value)
    before = _snapshot(sm, LEGACY_KEY)
    who.org = ORG_B
    missing = client.post(f"{DRAFTS}/never-existed/cancel")
    for r in (
        client.get(f"{DRAFTS}/{LEGACY_KEY}"),
        _release(client, LEGACY_KEY, "a" * 64),
        client.post(f"{DRAFTS}/{LEGACY_KEY}/cancel"),
    ):
        assert r.status_code == 404, "tenant scoping must not weaken: another tenant never learns the row exists"
        assert r.json() == missing.json()
    assert _snapshot(sm, LEGACY_KEY) == before and smtp.calls == []


def test_the_same_key_in_two_tenants_resolves_to_each_tenants_own_row(sm, client, who):
    """Keys are unique per organization, so a legacy row in A and a new-style row in B may share a key; each
    tenant must only ever see its own."""
    _legacy(sm, S.SENT.value, key="shared-key", org=ORG_A)
    with sm() as s:
        seed_dispatch(s, org=ORG_B, idem="shared-key", state=S.AWAITING_APPROVAL.value, content_approval_required=True,
                      sender_email="outreach@mailer.example.com")
    who.org = ORG_A
    assert client.post(f"{DRAFTS}/shared-key/cancel").json() == {"detail": LEGACY_CODE}
    who.org = ORG_B
    r = client.post(f"{DRAFTS}/shared-key/cancel")
    assert r.status_code == 200 and r.json()["state"] == "cancelled"


NOT_A_REPLY_CONTEXTS = [
    pytest.param(None, id="no-grounding-context"),
    pytest.param({}, id="empty-object"),
    pytest.param({"proof_points": ["x"]}, id="ordinary-outreach-context"),
    pytest.param({"reply_parent": None}, id="reply_parent-null"),
    pytest.param({"reply_parent": "reply_parent"}, id="reply_parent-string"),
    pytest.param({"reply_parent": []}, id="reply_parent-array"),
    pytest.param({"note": "reply_parent"}, id="word-only-in-a-value"),
]


@pytest.mark.parametrize("context", NOT_A_REPLY_CONTEXTS)
def test_ordinary_non_reply_dispatch_keeps_the_plain_404(sm, client, smtp, context):
    """A dispatch with the flag false that is NOT a reply (no reply_parent OBJECT) is not a draft and not a legacy
    reply: the 404 is unchanged, and identical to a nonexistent key."""
    _legacy(sm, S.QUEUED.value, key="ordinary-2", context=context)
    before = _snapshot(sm, "ordinary-2")
    missing = client.post(f"{DRAFTS}/never-existed/cancel")
    for r in (
        client.get(f"{DRAFTS}/ordinary-2"),
        _release(client, "ordinary-2", "a" * 64),
        client.post(f"{DRAFTS}/ordinary-2/cancel"),
    ):
        assert r.status_code == 404 and r.json() == missing.json()
    assert _snapshot(sm, "ordinary-2") == before and smtp.calls == []


def test_a_genuinely_nonexistent_operation_is_still_a_404(client):
    for r in (
        client.get(f"{DRAFTS}/never-existed"),
        _release(client, "never-existed", "a" * 64),
        client.post(f"{DRAFTS}/never-existed/cancel"),
    ):
        assert r.status_code == 404 and r.json() == {"detail": "Reply draft not found"}


def test_a_new_c95_reply_is_unaffected_and_still_cancellable(sm, client, fake_llm, factory, smtp):
    """The new flow keeps its exact behaviour even though its row also carries reply_parent."""
    _held(sm, client, fake_llm, factory)
    assert _row(sm)["flag"] is True
    r = client.post(f"{DRAFTS}/{KEY}/cancel")
    assert r.status_code == 200 and r.json()["state"] == "cancelled"
    assert smtp.calls == []


def test_a_legacy_queued_row_is_still_sent_by_the_unchanged_send_path(sm, client, factory, smtp):
    """This patch must not change what the SMTP worker does with a legacy row (that is the old C9.4 behaviour the
    rollout gate exists to drain before the new workers start). Documented here so a future change is deliberate."""
    _legacy(sm, S.QUEUED.value, with_message=True)
    assert _send(factory) is not None
    assert len(smtp.calls) == 1 and _row(sm, LEGACY_KEY)["state"] == S.SENT.value
