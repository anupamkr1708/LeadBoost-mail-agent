"""
M2-B -- durable generation behind acceptance (SQLite, offline, deterministic).

    accept (QUEUED, message_id NULL) -> GENERATING -> QUEUED (Message set) | FAILED
                                     -> (C6-C8 worker) SENDING -> SENT/FAILED/UNKNOWN

Covers the claim, the fenced write, "no session open across the LLM call",
lease recovery (GENERATING -> QUEUED, the opposite of SENDING -> UNKNOWN), a
crash matrix, and the full accept -> generate -> mailbox send chain.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailer_agent.api.deps import get_current_org_id, get_integration_org_id, require_api_key
from mailer_agent.api.main import app
from mailer_agent.config import get_settings
from mailer_agent.db import get_db
from mailer_agent.followup import work_claiming as wc
from mailer_agent.mail import external_dispatch_worker as w
from mailer_agent.mail import outreach_generation_worker as gw
from mailer_agent.models import (
    Base,
    ExternalDispatch,
    ExternalDispatchState as S,
    Mailbox,
    MailboxStatus,
    Message,
    MessageStatus,
)
from tests.dispatch_support import FakeSender, TrackingFactory, age, seed_dispatch, seed_org_mailboxes

ORG = "org-a"
URL = "/integrations/leadboost/outreach-requests"
VP = "We cut manual invoice reconciliation time by 40% for finance teams."
BODY = (
    "Hi Jane,\n\nWe cut manual invoice reconciliation time by 40% for finance teams.\n\n"
    "Worth a quick chat?\n\nBest,\nTest Sender"
)


class Dead(BaseException):
    """Simulated process death: not an Exception, so no handler can swallow it."""


# ------------------------------------------------------------------ fixtures

@pytest.fixture(autouse=True)
def _sender_identity(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "leadboost_integration_sender_email", "outreach@mailer.example.com")
    monkeypatch.setattr(s, "leadboost_integration_sender_name", "Test Sender")
    monkeypatch.setattr(s, "leadboost_integration_sender_org", "Test Org")


@pytest.fixture()
def sm(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path/'m2b.db'}", connect_args={"check_same_thread": False, "timeout": 15})
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
def client(sm):
    def _get_db():
        s = sm()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_integration_org_id] = lambda: ORG
    app.dependency_overrides[get_current_org_id] = lambda: ORG
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def send_spy(monkeypatch):
    monkeypatch.setattr(w.settings, "live_sending_enabled", True)
    f = FakeSender()
    monkeypatch.setattr(w, "send_email", f)
    return f


def _req(key="idem-1", action="481", email="jane@acme.example.com"):
    return {
        "external_action_id": action,
        "idempotency_key": key,
        "recipient": {"email": email, "name": "Jane Doe", "title": "VP Finance", "company": "Acme"},
        "context": {"value_proposition": VP, "recipient_facts": []},
    }


def _accept(client, fake_llm=None, key="idem-1", action="481", email="jane@acme.example.com", draft=True):
    if draft and fake_llm is not None:
        fake_llm.queue_response({"subject": "Quick question", "body": BODY, "reasoning": "t"})
    r = client.post(URL, json=_req(key, action, email))
    assert r.status_code == 202, r.text
    return r.json()["mailing_agent_reference"]


def _gen_one(factory, worker="gen-1", runtime=None):
    return gw.process_next_generation(session_factory=factory, worker_id=worker, runtime=runtime or gw.DispatchRuntime())


def _send_one(factory, worker="snd-1"):
    return w.process_next_external_dispatch(session_factory=factory, worker_id=worker, runtime=w.DispatchRuntime())


def _d(sm, key="idem-1"):
    with sm() as s:
        d = s.query(ExternalDispatch).filter_by(idempotency_key=key).one()
        return {
            "state": d.state, "message_id": d.message_id, "claimed_by": d.claimed_by,
            "claimed_at": d.claimed_at, "error": d.error_message, "id": d.id,
        }


def _n_messages(sm):
    with sm() as s:
        return s.query(Message).count()


def _expire(sm, key="idem-1", seconds=10_000):
    with sm() as s:
        s.query(ExternalDispatch).filter_by(idempotency_key=key).update({"claimed_at": age(seconds)})
        s.commit()


def _recover(factory):
    return gw.run_generation_lease_recovery(session_factory=factory)


# ------------------------------------------------------------- claim shapes

def test_generation_claims_only_queued_rows_without_a_message(sm, factory, client, fake_llm):
    _accept(client, fake_llm, key="pending")                       # QUEUED, message NULL
    with sm() as s:
        seed_dispatch(s, org=ORG, idem="has-msg", email="x1@example.com")      # QUEUED, message present (exact route shape)
        seed_dispatch(s, org=ORG, idem="sent", state=S.SENT.value, email="x2@example.com")
    res = _gen_one(factory)
    assert res.outcome == "generated" and res.dispatch_id == _d(sm, "pending")["id"]
    assert _gen_one(factory) is None                                # nothing else is eligible
    assert _d(sm, "has-msg")["state"] == S.QUEUED.value and _d(sm, "sent")["state"] == S.SENT.value


def test_generation_claim_is_oldest_first_and_single_owner(sm, factory, client, fake_llm):
    _accept(client, fake_llm, key="A", action="1")
    _accept(client, fake_llm, key="B", action="2", email="sam@globex.example.com")
    with factory.sm() as s:
        c1 = wc.claim_next_generation_dispatch(s, "gen-x")
        s.commit()
    with factory.sm() as s:
        c2 = wc.claim_next_generation_dispatch(s, "gen-y")
        c3 = wc.claim_next_generation_dispatch(s, "gen-z")
        s.commit()
    assert c1.dispatch_id != c2.dispatch_id and c3 is None          # nobody claims a claimed row
    assert _d(sm, "A")["state"] == _d(sm, "B")["state"] == S.GENERATING.value
    assert c1.dispatch_id < c2.dispatch_id


def test_send_claim_never_takes_a_row_awaiting_generation(sm, factory, client, fake_llm, send_spy):
    _accept(client, fake_llm)
    assert _send_one(factory) is None and send_spy.calls == []
    with factory.sm() as s:
        assert wc.claim_next_external_dispatch(s, "snd-x") is None
    assert _d(sm)["state"] == S.QUEUED.value and _d(sm)["message_id"] is None


def test_a_generating_row_is_not_claimable_for_send_or_generation(sm, factory, client, fake_llm, send_spy):
    _accept(client, fake_llm)
    with factory.sm() as s:
        assert wc.claim_next_generation_dispatch(s, "gen-1")
        s.commit()
    assert _gen_one(factory, "gen-2") is None and _send_one(factory) is None and send_spy.calls == []


# ----------------------------------------------- transactions / the LLM boundary

def test_no_session_or_transaction_is_open_while_the_llm_runs(sm, factory, client, fake_llm, monkeypatch):
    _accept(client, fake_llm)
    seen = []
    real = gw.draft_message

    def spying_draft(**kw):
        seen.append(len(factory.open_transactions()))
        return real(**kw)

    monkeypatch.setattr(gw, "draft_message", spying_draft)
    assert _gen_one(factory).outcome == "generated"
    assert seen == [0]
    assert _d(sm)["state"] == S.QUEUED.value                       # GENERATING was already committed + closed


def test_generating_is_durable_while_the_llm_runs(sm, factory, client, fake_llm, monkeypatch):
    _accept(client, fake_llm)
    during = {}
    real = gw.draft_message

    def peek(**kw):
        during.update(_d(sm))
        during["messages"] = _n_messages(sm)
        return real(**kw)

    monkeypatch.setattr(gw, "draft_message", peek)
    _gen_one(factory, worker="gen-peek")
    assert (during["state"], during["claimed_by"], during["message_id"], during["messages"]) == (
        S.GENERATING.value, "gen-peek", None, 0)
    assert during["claimed_at"] is not None


def test_llm_is_fed_only_the_durable_snapshot_not_shared_rows(sm, factory, client, fake_llm):
    _accept(client, fake_llm)
    with sm() as s:                                                 # shared rows change after acceptance
        from mailer_agent.models import Campaign, Contact
        s.query(Campaign).update({"value_prop": "SOMETHING ELSE ENTIRELY"})
        s.query(Contact).update({"title": "Intern", "company": "Other Co"})
        s.commit()
    _gen_one(factory)
    prompt = "\n".join(fake_llm.last_prompts[-1])
    assert VP in prompt and "VP Finance" in prompt and "Acme" in prompt
    assert "SOMETHING ELSE ENTIRELY" not in prompt and "Intern" not in prompt


# ---------------------------------------------------------- outcomes / failures

def test_success_attaches_exactly_one_message_and_releases_the_claim(sm, factory, client, fake_llm):
    _accept(client, fake_llm)
    res = _gen_one(factory)
    d = _d(sm)
    assert (res.outcome, res.persisted) == ("generated", True)
    assert d["state"] == S.QUEUED.value and d["message_id"] and (d["claimed_by"], d["claimed_at"]) == (None, None)
    with sm() as s:
        m = s.get(Message, d["message_id"])
        assert (m.status, m.body) == (MessageStatus.DRAFT.value, BODY)
    assert _n_messages(sm) == 1 and _gen_one(factory) is None


def test_llm_exception_is_failed_with_no_message(sm, factory, client, fake_llm):
    fake_llm.queue_error(RuntimeError("provider exploded"))
    _accept(client, fake_llm, draft=False)
    res = _gen_one(factory)
    d = _d(sm)
    assert res.outcome == "failed" and d["state"] == S.FAILED.value and d["error"].startswith("generation_failed")
    assert d["message_id"] is None and (d["claimed_by"], d["claimed_at"]) == (None, None) and _n_messages(sm) == 0


def test_missing_snapshot_is_failed_before_any_llm_call(sm, factory, client, fake_llm):
    _accept(client, fake_llm)
    with sm() as s:
        from sqlalchemy import text
        s.execute(text("UPDATE external_dispatches SET grounding_context = NULL"))   # core UPDATE: bypasses ORM guard
        s.commit()
    res = _gen_one(factory)
    assert res.outcome == "failed" and fake_llm.call_count == 0
    assert _d(sm)["error"].startswith("generation_context_missing") and _n_messages(sm) == 0


def test_campaign_of_another_org_is_refused_before_any_llm_call(sm, factory, client, fake_llm):
    from mailer_agent.models import Campaign
    _accept(client, fake_llm)
    with sm() as s:
        s.query(Campaign).update({"organization_id": "org-evil"})
        s.commit()
    res = _gen_one(factory)
    assert res.outcome == "failed" and fake_llm.call_count == 0
    assert _d(sm)["error"].startswith("tenant_mismatch")


def test_grounding_hard_block_is_failed_and_nothing_is_ever_sent(sm, factory, client, fake_llm, send_spy):
    fake_llm.queue_response({"subject": "S", "body": "Hi Jane,\n\nWe saved teams $90,000 last year.\n\nBest", "reasoning": "t"})
    client.post(URL, json={**_req(), "context": {"value_proposition": "We help finance teams.", "recipient_facts": []}})
    assert _gen_one(factory).outcome == "failed"
    assert _send_one(factory) is None and send_spy.calls == [] and _n_messages(sm) == 0


# ------------------------------------------------------------- crash matrix

def test_G1_crash_before_the_claim_commits_leaves_the_row_queued_and_claimable(sm, factory, client, fake_llm, monkeypatch):
    _accept(client, fake_llm)

    def die(*a, **k):
        raise Dead()

    monkeypatch.setattr(gw, "_prepare", die)                        # after claim UPDATE, before commit
    with pytest.raises(Dead):
        _gen_one(factory, "gen-dead")
    d = _d(sm)
    assert (d["state"], d["claimed_by"], d["message_id"]) == (S.QUEUED.value, None, None)   # rolled back
    monkeypatch.undo()
    assert _gen_one(factory).outcome == "generated"


def test_crash_during_the_llm_leaves_generating_then_lease_recovery_requeues_without_a_message(
    sm, factory, client, fake_llm, monkeypatch
):
    _accept(client, fake_llm)

    def die(**k):
        raise Dead()

    monkeypatch.setattr(gw, "draft_message", die)
    with pytest.raises(Dead):
        _gen_one(factory, "gen-dead")
    d = _d(sm)
    assert (d["state"], d["claimed_by"], d["message_id"]) == (S.GENERATING.value, "gen-dead", None)
    assert _n_messages(sm) == 0

    assert _recover(factory) == 0                                   # lease not expired yet: untouched
    _expire(sm)
    assert _recover(factory) == 1
    d = _d(sm)
    assert (d["state"], d["claimed_by"], d["claimed_at"], d["message_id"]) == (S.QUEUED.value, None, None, None)
    monkeypatch.undo()
    assert _gen_one(factory, "gen-2").outcome == "generated" and _n_messages(sm) == 1   # regenerated once


def test_crash_between_llm_and_G2_commit_leaves_no_message_and_recovers(sm, factory, client, fake_llm, monkeypatch):
    _accept(client, fake_llm)

    def die(*a, **k):
        raise Dead()

    monkeypatch.setattr(gw, "_transaction_g2", die)                 # LLM already returned a draft
    with pytest.raises(Dead):
        _gen_one(factory, "gen-dead")
    assert _d(sm)["state"] == S.GENERATING.value and _n_messages(sm) == 0
    _expire(sm)
    assert _recover(factory) == 1
    monkeypatch.undo()
    assert _gen_one(factory).outcome == "generated" and _n_messages(sm) == 1


def test_late_worker_with_a_lost_lease_cannot_create_a_second_message(sm, factory, client, fake_llm, monkeypatch):
    """Worker 1 stalls in the LLM; its lease expires; recovery requeues; worker 2
    generates and completes; worker 1 then wakes and tries to write. Exactly one
    Message ever exists and worker 1's result is discarded."""
    _accept(client, fake_llm)
    fake_llm.queue_response({"subject": "Second draft", "body": BODY, "reasoning": "t"})
    outcome = {}
    real = gw.draft_message
    calls = {"n": 0}

    def stall_then_race(**kw):
        calls["n"] += 1
        if calls["n"] == 1:                                         # worker 1 is inside the LLM...
            _expire(sm)
            assert _recover(factory) == 1                           # ...its lease is recovered...
            outcome["w2"] = _gen_one(factory, "gen-2")              # ...and worker 2 finishes the job
        return real(**kw)

    monkeypatch.setattr(gw, "draft_message", stall_then_race)
    res1 = _gen_one(factory, "gen-1")
    assert outcome["w2"].outcome == "generated"
    assert (res1.outcome, res1.persisted) == ("lease_lost", False)
    assert _n_messages(sm) == 1
    d = _d(sm)
    assert d["state"] == S.QUEUED.value and d["message_id"] is not None


def test_stale_worker_cannot_write_while_another_worker_holds_a_fresh_generating_claim(sm, factory, client, fake_llm):
    """The row is GENERATING again (worker 2, new claim token) when stale worker 1
    writes. State alone would match; owner + exact claimed_at must reject it."""
    _accept(client, fake_llm)
    with factory.sm() as s:
        c1 = wc.claim_next_generation_dispatch(s, "gen-1")
        s.commit()
    _expire(sm)
    assert _recover(factory) == 1                                   # worker 1's lease is recovered
    with factory.sm() as s:
        c2 = wc.claim_next_generation_dispatch(s, "gen-2")          # worker 2 now owns it
        s.commit()
    assert c2.dispatch_id == c1.dispatch_id and c2.worker_id != c1.worker_id

    with factory.sm() as s:                                         # stale worker 1 tries both outcomes
        m = Message(contact_id=1, direction="outbound", subject="stale", body="stale", status="draft")
        s.add(m)
        s.flush()
        assert wc.complete_generation(s, c1, m.id) is False
        assert wc.fail_generation(s, c1, "stale failure") is False
        s.rollback()
    d = _d(sm)
    assert (d["state"], d["claimed_by"], d["message_id"], d["error"]) == (S.GENERATING.value, "gen-2", None, None)

    with factory.sm() as s:                                         # the real owner still can
        m = Message(contact_id=1, direction="outbound", subject="ok", body="ok", status="draft")
        s.add(m)
        s.flush()
        assert wc.complete_generation(s, c2, m.id) is True
        s.commit()
    assert _d(sm)["state"] == S.QUEUED.value and _n_messages(sm) == 1


def test_G2_database_failure_is_retried_then_left_for_recovery(sm, factory, client, fake_llm, monkeypatch):
    from sqlalchemy.exc import OperationalError
    _accept(client, fake_llm)
    monkeypatch.setattr(gw, "_TXN_G2_RETRY_DELAY_SECONDS", 0)
    real_complete = gw.complete_generation
    attempts = {"n": 0}

    def flaky(db, claim, message_id):
        attempts["n"] += 1
        if attempts["n"] < gw._TXN_G2_ATTEMPTS:
            raise OperationalError("stmt", {}, Exception("db blip"))
        return real_complete(db, claim, message_id)

    monkeypatch.setattr(gw, "complete_generation", flaky)
    assert _gen_one(factory).outcome == "generated" and _n_messages(sm) == 1 and attempts["n"] == 3   # transient: retried

    # persistent failure: never persisted, never a stray Message, recoverable
    _accept(client, fake_llm, key="idem-2", action="2", email="b@acme.example.com")
    monkeypatch.setattr(gw, "complete_generation", lambda *a, **k: (_ for _ in ()).throw(OperationalError("s", {}, Exception("down"))))
    res = _gen_one(factory, "gen-dead")
    assert (res.outcome, res.persisted) == ("lease_lost", False)
    assert _d(sm, "idem-2")["state"] == S.GENERATING.value and _n_messages(sm) == 1
    _expire(sm, "idem-2")
    assert _recover(factory) == 1 and _d(sm, "idem-2")["state"] == S.QUEUED.value


def test_shutdown_requested_means_no_claim_and_no_llm(sm, factory, client, fake_llm):
    _accept(client, fake_llm)
    rt = gw.DispatchRuntime()
    rt.request_shutdown()
    assert _gen_one(factory, runtime=rt) is None and fake_llm.call_count == 0
    assert _d(sm)["state"] == S.QUEUED.value


# ------------------------------------------------ recovery scope (C6-C8 intact)

def test_recovery_touches_only_expired_generating_rows(sm, factory, client, fake_llm):
    _accept(client, fake_llm, key="gen-exp", action="1")
    _accept(client, fake_llm, key="gen-live", action="2", email="b@acme.example.com")
    with sm() as s:
        seed_dispatch(s, org=ORG, idem="snd-exp", state=S.SENDING.value, claimed_by="w", claimed_at=age(10_000),
                      email="x3@example.com")
        seed_dispatch(s, org=ORG, idem="queued", state=S.QUEUED.value, email="x4@example.com")
    with factory.sm() as s:
        wc.claim_next_generation_dispatch(s, "g1")
        wc.claim_next_generation_dispatch(s, "g2")
        s.commit()
    with sm() as s:
        s.query(ExternalDispatch).filter_by(idempotency_key="gen-exp").update({"claimed_at": age(10_000)})
        s.commit()
    assert _recover(factory) == 1
    assert _d(sm, "gen-exp")["state"] == S.QUEUED.value
    assert _d(sm, "gen-live")["state"] == S.GENERATING.value
    assert _d(sm, "snd-exp")["state"] == S.SENDING.value            # generation sweep never resolves SENDING
    assert _d(sm, "queued")["state"] == S.QUEUED.value


def test_sending_recovery_is_unchanged_and_ignores_generating_rows(sm, factory, client, fake_llm):
    _accept(client, fake_llm, key="gen", action="1")
    with sm() as s:
        seed_dispatch(s, org=ORG, idem="snd", state=S.SENDING.value, claimed_by="w", claimed_at=age(10_000),
                      email="x5@example.com")
    with factory.sm() as s:
        wc.claim_next_generation_dispatch(s, "g1")
        s.commit()
    _expire(sm, "gen")
    with factory.sm() as s:
        recovered = wc.recover_expired_external_dispatches(s)
        s.commit()
    assert [r.dispatch_id for r in recovered] == [_d(sm, "snd")["id"]]
    assert _d(sm, "snd")["state"] == S.UNKNOWN.value                # C6-C8 rule intact: SENDING -> UNKNOWN
    assert _d(sm, "gen")["state"] == S.GENERATING.value             # the SENDING sweep never touches GENERATING


def test_generating_never_resolves_to_unknown_or_failed_by_recovery(sm, factory, client, fake_llm):
    _accept(client, fake_llm)
    with factory.sm() as s:
        wc.claim_next_generation_dispatch(s, "g1")
        s.commit()
    _expire(sm)
    _recover(factory)
    assert _d(sm)["state"] == S.QUEUED.value                        # requeued, not UNKNOWN/FAILED: nothing was sent


# ----------------------------------------------------------- full chain

def test_accept_generate_then_send_through_the_mailbox(sm, factory, client, fake_llm, send_spy):
    ref = _accept(client, fake_llm)
    assert _send_one(factory) is None and send_spy.calls == []      # nothing sendable before generation
    assert _gen_one(factory).outcome == "generated"
    assert send_spy.calls == []                                     # generation never sends
    res = _send_one(factory)
    assert res.outcome == "sent" and len(send_spy.calls) == 1
    call = send_spy.calls[0]
    assert call["body_text"] == BODY and call["to_email"] == "jane@acme.example.com"
    assert call["from_email"] == "outreach@mailer.example.com" and call["smtp_config"].host == "smtp.mailbox.example"
    d = _d(sm)
    assert d["state"] == S.SENT.value
    assert client.get(f"/integrations/leadboost/outreach-actions/idem-1").json()["mailing_agent_reference"] == ref


def test_reconciliation_reads_queued_for_the_whole_pre_send_phase(sm, factory, client, fake_llm, monkeypatch):
    _accept(client, fake_llm)
    read = lambda: client.get("/integrations/leadboost/outreach-actions/idem-1").json()["state"]
    states = [read()]                                               # accepted, awaiting generation
    real = gw.draft_message
    monkeypatch.setattr(gw, "draft_message", lambda **kw: (states.append(read()), real(**kw))[1])   # GENERATING
    _gen_one(factory)
    states.append(read())                                           # generated, awaiting send
    assert states == ["queued", "queued", "queued"]


def test_mailbox_disabled_between_generation_and_send_fails_pre_smtp_keeping_the_message(
    sm, factory, client, fake_llm, send_spy
):
    _accept(client, fake_llm)
    _gen_one(factory)
    with sm() as s:
        s.query(Mailbox).filter_by(organization_id=ORG).update({"status": MailboxStatus.DISABLED.value})
        s.commit()
    res = _send_one(factory)
    d = _d(sm)
    assert res.outcome == "failed" and send_spy.calls == []
    assert d["state"] == S.FAILED.value and d["error"].startswith("mailbox_disabled") and d["message_id"]


def test_second_generation_sweep_and_worker_cannot_duplicate_work(sm, factory, client, fake_llm, send_spy):
    _accept(client, fake_llm)
    assert _gen_one(factory, "g1").outcome == "generated"
    assert _gen_one(factory, "g2") is None and _recover(factory) == 0
    assert _send_one(factory, "s1").outcome == "sent" and _send_one(factory, "s2") is None
    assert len(send_spy.calls) == 1 and _n_messages(sm) == 1 and fake_llm.call_count == 1


# ------------------------------------------------- boundaries and wiring

ROOT = Path(__file__).resolve().parent.parent


def _imports(rel):
    tree = ast.parse((ROOT / rel).read_text())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


def test_generation_worker_has_no_smtp_or_sender_dependency():
    imports = _imports("mailer_agent/mail/outreach_generation_worker.py")
    assert not {i for i in imports if i.startswith(("mailer_agent.mail.sender", "smtplib", "mailer_agent.mailbox_secrets"))}
    assert "mailer_agent.llm.agent" in imports                       # the one place the LLM is called


def test_dispatch_worker_still_imports_no_llm_or_generation_code():
    imports = _imports("mailer_agent/mail/external_dispatch_worker.py")
    assert not {i for i in imports if i.startswith(("mailer_agent.llm", "mailer_agent.memory", "mailer_agent.mail.outreach_generation_worker"))}


def test_intake_module_no_longer_imports_the_llm_or_draft_path():
    imports = _imports("mailer_agent/api/integrations_generated.py")
    assert not {i for i in imports if i.startswith(("mailer_agent.llm", "mailer_agent.mail.exact_message"))}


def test_scheduler_registers_generation_and_recovery_jobs():
    from mailer_agent.followup import scheduler

    src = Path(scheduler.__file__).read_text()
    for job in ("generate_external_dispatches_job", "recover_external_generation_leases_job"):
        assert hasattr(scheduler, job) and f"{job},\n" in src


def test_migration_008_is_a_noop_on_sqlite_and_models_are_nullable():
    import importlib.util

    spec = importlib.util.spec_from_file_location("m008", ROOT / "migrations" / "008_external_dispatch_deferred_message.py")
    mig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mig)
    assert ExternalDispatch.__table__.c.message_id.nullable is True
    assert "make_message_id_nullable" in dir(mig) and "make_message_id_required" in dir(mig)
