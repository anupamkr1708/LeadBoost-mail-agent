"""
C7/C8 -- the async ExternalDispatch pipeline, end to end against SQLite with
deterministic doubles (no real SMTP, no LLM, no network).

Real-SMTP behaviour is exercised only against the in-process FakeSMTPServer;
PostgreSQL locking is in test_external_dispatch_postgres.py.
"""

from __future__ import annotations

import email
import sqlite3

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from mailer_agent.followup import work_claiming as wc
from mailer_agent.mail import external_dispatch_worker as w
from mailer_agent.mail import sender
from mailer_agent.mail.sender import SendOutcome
from mailer_agent.models import (
    Base,
    Campaign,
    Contact,
    ExternalDispatch,
    ExternalDispatchState as S,
    Message,
    MessageStatus,
    SuppressionEntry,
)
from tests.dispatch_support import FakeSender, FakeSMTPServer, TrackingFactory, age, seed_dispatch

CLEAN_BODY = "Hi Jane,\n\nWorth a quick chat?\n\nBest"
WORKER = "w-test"


@pytest.fixture()
def dbfile(tmp_path):
    return tmp_path / "w.db"


@pytest.fixture()
def factory(dbfile):
    eng = create_engine(f"sqlite:///{dbfile}", connect_args={"check_same_thread": False, "timeout": 15})
    Base.metadata.create_all(bind=eng)
    yield TrackingFactory(sessionmaker(bind=eng))
    eng.dispose()


@pytest.fixture()
def runtime():
    return w.DispatchRuntime()


@pytest.fixture()
def live(monkeypatch):
    monkeypatch.setattr(w.settings, "live_sending_enabled", True)


@pytest.fixture()
def fake(monkeypatch, live):
    f = FakeSender()
    monkeypatch.setattr(w, "send_email", f)
    return f


def seed(factory, **kw):
    with factory.sm() as s:
        d = seed_dispatch(s, **kw)
        return d.id


def run(factory, runtime, **kw):
    return w.process_next_external_dispatch(session_factory=factory, worker_id=WORKER, runtime=runtime, **kw)


def row(factory, did):
    with factory.sm() as s:
        d = s.get(ExternalDispatch, did)
        m = s.get(Message, d.message_id)
        s.expunge_all()
        return d, m


# ------------------------------------------------------------------ happy path

def test_queued_to_sent_with_exact_message_and_claim_released(factory, runtime, fake):
    body = "Line one  \r\n\r\n  indented ünïcode ✓ {{not_a_template}}\n"
    did = seed(factory, subject="Exact  subject ✓", body=body)

    res = run(factory, runtime)

    assert (res.outcome, res.persisted, res.smtp_attempted) == ("sent", True, True)
    d, m = row(factory, did)
    assert d.state == S.SENT.value
    assert d.claimed_by is None and d.claimed_at is None and d.error_message is None
    assert m.status == MessageStatus.SENT.value
    assert (m.subject, m.body) == ("Exact  subject ✓", body)          # never modified
    call = fake.calls[0]
    assert (call["subject"], call["body_text"]) == ("Exact  subject ✓", body)
    assert call["to_email"] == "lead@example.com"


def test_exact_bytes_on_the_wire_through_the_real_sender(factory, runtime, live, monkeypatch):
    with FakeSMTPServer() as srv:
        # The Mailer-owned mailbox carries the transport; global settings.smtp_* are NOT patched.
        did = seed(factory, subject="Wire subject", body=CLEAN_BODY,
                   smtp_host="127.0.0.1", smtp_port=srv.port)

        res = run(factory, runtime)

        assert res.outcome == "sent" and len(srv.accepted) == 1     # exactly one delivery
        _, m = row(factory, did)
        parsed = email.message_from_string(srv.accepted[0])
        assert parsed["Subject"] == "Wire subject"
        assert parsed["Message-ID"] == m.message_id_header == res.message_id_header
        text = parsed.get_payload()[0].get_payload(decode=True).decode()
        assert text.replace("\r\n", "\n").strip() == CLEAN_BODY.strip()


def test_no_llm_drafting_and_no_new_message_rows(factory, runtime, fake, monkeypatch):
    import mailer_agent.llm.agent as agent

    def boom(*a, **k):
        raise AssertionError("LLM drafting must never run on the exact-message path")

    monkeypatch.setattr(agent, "draft_message", boom)
    seed(factory)
    with factory.sm() as s:
        before = s.scalar(select(func.count()).select_from(Message))
    run(factory, runtime)
    with factory.sm() as s:
        assert s.scalar(select(func.count()).select_from(Message)) == before


# --------------------------------------------------- transaction boundary + ID

def test_message_id_committed_and_no_transaction_open_during_smtp(factory, runtime, live, monkeypatch, dbfile):
    did = seed(factory)
    seen = {}

    def during_smtp(kw):
        # 1. committed & visible to an independent connection
        with factory.sm() as other:
            d = other.get(ExternalDispatch, did)
            m = other.get(Message, d.message_id)
            seen["state"], seen["claimed_by"] = d.state, d.claimed_by
            seen["db_id"], seen["msg_status"] = m.message_id_header, m.status
        # 2. no session of ours has a transaction open
        seen["open_txns"] = factory.open_transactions()
        # 3. no write lock is held: an independent writer gets in immediately
        con = sqlite3.connect(dbfile, timeout=0.2)
        try:
            con.execute("BEGIN IMMEDIATE")
            seen["writer_ok"] = True
            con.rollback()
        except sqlite3.OperationalError:
            seen["writer_ok"] = False
        finally:
            con.close()
        seen["kw_id"] = kw["message_id_header"]

    f = FakeSender(on_call=during_smtp)
    monkeypatch.setattr(w, "send_email", f)
    res = run(factory, runtime)

    assert seen["state"] == S.SENDING.value and seen["claimed_by"] == WORKER
    assert seen["msg_status"] == MessageStatus.SENDING.value
    assert seen["db_id"] is not None                              # persisted BEFORE smtp
    assert seen["kw_id"] == seen["db_id"] == res.message_id_header  # DB id == transmitted id
    assert seen["open_txns"] == []                                # no DB transaction during SMTP
    assert seen["writer_ok"] is True                              # and no lock either
    assert f.calls[0]["message_id_header"] == seen["db_id"]


def test_persisted_id_is_generated_with_the_sender_domain(factory, runtime, fake):
    did = seed(factory, sender_email="outreach@sender.example.org")
    run(factory, runtime)
    _, m = row(factory, did)
    assert m.message_id_header.startswith("<") and m.message_id_header.endswith("@sender.example.org>")


# ------------------------------------------------------------- LIVE_SENDING

def test_live_sending_disabled_never_calls_smtp_and_never_records_sent(factory, runtime, monkeypatch):
    monkeypatch.setattr(w.settings, "live_sending_enabled", False)
    f = FakeSender()
    monkeypatch.setattr(w, "send_email", f)
    did = seed(factory)

    res = run(factory, runtime)

    assert f.calls == []
    assert (res.outcome, res.smtp_attempted) == ("failed", False)
    d, m = row(factory, did)
    assert d.state == S.FAILED.value and "live_sending_disabled" in d.error_message
    assert d.claimed_by is None and d.claimed_at is None
    assert m.status == MessageStatus.FAILED.value and m.message_id_header is None


def test_live_sending_disabled_with_the_real_sender_cannot_produce_false_sent(factory, runtime, monkeypatch):
    """Regression: sender.send_email() in dry-run returns SENT without
    transmitting. The worker must never reach it."""
    monkeypatch.setattr(w.settings, "live_sending_enabled", False)
    with FakeSMTPServer() as srv:
        monkeypatch.setattr(sender.settings, "smtp_host", "127.0.0.1")
        monkeypatch.setattr(sender.settings, "smtp_port", srv.port)
        did = seed(factory)
        run(factory, runtime)
        assert srv.connections == 0 and srv.accepted == []
    d, m = row(factory, did)
    assert d.state != S.SENT.value and m.status != MessageStatus.SENT.value


def test_live_sending_enabled_reaches_the_send_path(factory, runtime, fake):
    seed(factory)
    run(factory, runtime)
    assert len(fake.calls) == 1


# ---------------------------------------------------------------- suppression

def test_suppressed_recipient_blocks_smtp_and_fails(factory, runtime, fake):
    with factory.sm() as s:
        s.add(SuppressionEntry(email="lead@example.com", organization_id="org-a", reason="unsubscribed"))
        s.commit()
    did = seed(factory)
    res = run(factory, runtime)
    assert fake.calls == [] and res.outcome == "failed"
    d, m = row(factory, did)
    assert d.state == S.FAILED.value and d.error_message.startswith("suppressed")
    assert m.status == MessageStatus.FAILED.value


def test_suppression_from_another_org_also_blocks_like_the_legacy_default(factory, runtime, fake):
    with factory.sm() as s:
        s.add(SuppressionEntry(email="lead@example.com", organization_id="org-b", reason="bounced"))
        s.commit()
    seed(factory)
    assert run(factory, runtime).outcome == "failed" and fake.calls == []


def test_unsuppressed_recipient_proceeds(factory, runtime, fake):
    with factory.sm() as s:
        s.add(SuppressionEntry(email="someone-else@example.com", organization_id="org-a"))
        s.commit()
    seed(factory)
    assert run(factory, runtime).outcome == "sent" and len(fake.calls) == 1


def test_suppression_helper_matches_legacy_engine_semantics(factory):
    from mailer_agent.followup.engine import is_suppressed as legacy
    from mailer_agent.suppression import is_email_suppressed

    with factory.sm() as s:
        s.add_all([
            SuppressionEntry(email="a@x.com", organization_id="org-a"),
            SuppressionEntry(email="b@x.com", organization_id="org-b"),
            SuppressionEntry(email="c@x.com", organization_id=None),
        ])
        s.commit()
        for email_addr in ("a@x.com", "b@x.com", "c@x.com", "A@x.com", "none@x.com"):
            for org in (None, "org-a", "org-b", "org-z"):
                assert is_email_suppressed(s, email_addr, org) == legacy(s, email_addr, org), (email_addr, org)


# ------------------------------------------------------------------ grounding

def test_grounding_hard_block_prevents_smtp_fails_and_never_edits(factory, runtime, fake):
    body = "Our customers see a 40% lift in reply rates."
    did = seed(factory, body=body)
    res = run(factory, runtime)
    assert fake.calls == [] and res.outcome == "failed"
    d, m = row(factory, did)
    assert d.state == S.FAILED.value and "grounding" in d.error_message
    assert d.claimed_by is None
    assert m.body == body and m.status == MessageStatus.FAILED.value  # text untouched


def test_clean_body_passes_grounding(factory, runtime, fake):
    seed(factory, body=CLEAN_BODY)
    assert run(factory, runtime).outcome == "sent"


def test_review_required_is_preserved_as_pass_through(factory, runtime, fake):
    body = "Happy to share pricing if useful."
    did = seed(factory, body=body)
    assert run(factory, runtime).outcome == "sent"
    assert fake.calls[0]["body_text"] == body


# ------------------------------------------------------------- SMTP outcomes

@pytest.mark.parametrize(
    "outcome, state, msg_status",
    [
        (SendOutcome.SENT, S.SENT, MessageStatus.SENT),
        (SendOutcome.FAILED, S.FAILED, MessageStatus.FAILED),
        (SendOutcome.UNKNOWN, S.UNKNOWN, MessageStatus.UNKNOWN),
    ],
)
def test_outcome_mapping_and_claim_release(factory, runtime, live, monkeypatch, outcome, state, msg_status):
    f = FakeSender(outcome=outcome, error=None if outcome is SendOutcome.SENT else "boom")
    monkeypatch.setattr(w, "send_email", f)
    did = seed(factory)
    res = run(factory, runtime)
    assert res.persisted is True
    d, m = row(factory, did)
    assert d.state == state.value and m.status == msg_status.value
    assert d.claimed_by is None and d.claimed_at is None
    assert (d.error_message is None) == (outcome is SendOutcome.SENT)
    assert m.message_id_header == res.message_id_header               # ID kept for every outcome


def test_unknown_is_never_resent_or_requeued(factory, runtime, live, monkeypatch):
    f = FakeSender(outcome=SendOutcome.UNKNOWN, error="ambiguous")
    monkeypatch.setattr(w, "send_email", f)
    did = seed(factory)
    run(factory, runtime)
    for _ in range(3):
        assert run(factory, runtime) is None                           # nothing claimable
        w.run_external_dispatch_lease_recovery(session_factory=factory)
    assert len(f.calls) == 1
    assert row(factory, did)[0].state == S.UNKNOWN.value


def test_send_email_raising_is_treated_as_unknown_not_failed(factory, runtime, live, monkeypatch):
    f = FakeSender(raises=RuntimeError("socket exploded"))
    monkeypatch.setattr(w, "send_email", f)
    did = seed(factory)
    res = run(factory, runtime)
    assert res.outcome == "unknown"
    d, m = row(factory, did)
    assert d.state == S.UNKNOWN.value and "RuntimeError" in d.error_message


# -------------------------------------------------------------------- tenancy

def test_cross_tenant_campaign_binding_is_refused(factory, runtime, fake):
    a = seed(factory, org="org-a", email="a@example.com")
    b = seed(factory, org="org-b", email="b@example.com")
    with factory.sm() as s:                         # corrupt: org-a's dispatch -> org-b's campaign
        camp_b = s.get(ExternalDispatch, b).campaign_id
        s.get(ExternalDispatch, a).campaign_id = camp_b
        s.commit()
    with factory.sm() as s:                         # process org-a's row only
        s.get(ExternalDispatch, b).state = S.SENT.value
        s.commit()
    res = run(factory, runtime)
    assert res.dispatch_id == a and res.outcome == "failed" and fake.calls == []
    assert "tenant_mismatch" in row(factory, a)[0].error_message


def test_contact_from_a_different_campaign_is_refused(factory, runtime, fake):
    a = seed(factory, org="org-a", email="a@example.com")
    b = seed(factory, org="org-b", email="b@example.com")
    with factory.sm() as s:
        s.get(ExternalDispatch, a).contact_id = s.get(ExternalDispatch, b).contact_id
        s.get(ExternalDispatch, b).state = S.SENT.value
        s.commit()
    assert run(factory, runtime).outcome == "failed" and fake.calls == []


def test_message_belonging_to_another_contact_is_refused(factory, runtime, fake):
    a = seed(factory, org="org-a", email="a@example.com")
    b = seed(factory, org="org-a", email="b@example.com")
    with factory.sm() as s:
        s.get(ExternalDispatch, a).message_id = s.get(ExternalDispatch, b).message_id
        s.get(ExternalDispatch, b).state = S.SENT.value
        s.commit()
    assert run(factory, runtime).outcome == "failed" and fake.calls == []


def test_non_draft_message_is_a_wiring_failure_not_a_send(factory, runtime, fake):
    did = seed(factory)
    with factory.sm() as s:
        s.get(Message, s.get(ExternalDispatch, did).message_id).status = MessageStatus.SENT.value
        s.commit()
    res = run(factory, runtime)
    assert res.outcome == "failed" and fake.calls == []
    assert "wiring_error" in row(factory, did)[0].error_message


def test_integration_contact_never_gains_a_follow_up_schedule(factory, runtime, fake):
    did = seed(factory)
    run(factory, runtime)
    with factory.sm() as s:
        contact = s.get(Contact, s.get(ExternalDispatch, did).contact_id)
        assert contact.next_action_at is None


# --------------------------------------------------- fencing / lease interplay

def test_late_smtp_success_after_lease_loss_does_not_rewrite_state(factory, runtime, live, monkeypatch):
    """Lease expires while SMTP is still running; recovery resolves the row to
    UNKNOWN; the send then succeeds. The fence must refuse the late write."""
    did = seed(factory)

    def expire_lease_mid_smtp(kw):
        with factory.sm() as s:
            assert len(wc.recover_expired_external_dispatches(s, lease_seconds=-1)) == 1
            s.commit()

    f = FakeSender(on_call=expire_lease_mid_smtp)
    monkeypatch.setattr(w, "send_email", f)
    res = run(factory, runtime)

    assert res.outcome == "sent" and res.persisted is False
    d, m = row(factory, did)
    assert d.state == S.UNKNOWN.value                                   # not rewritten to SENT
    assert m.status == MessageStatus.UNKNOWN.value
    assert "late_outcome_after_lease_loss" in d.error_message
    assert res.message_id_header in d.error_message                     # evidence preserved
    assert d.claimed_by is None
    assert len(f.calls) == 1


def test_stale_claim_token_cannot_finish_a_row_reclaimed_state(factory):
    did = seed(factory, state=S.SENDING.value, claimed_by="w-1", claimed_at=age(5))
    with factory.sm() as s:
        d = s.get(ExternalDispatch, did)
        stale = wc.ClaimedDispatch(did, d.organization_id, "w-1", age(999))     # wrong token
        assert wc.finish_external_dispatch(s, stale, new_state=S.SENT) is False
        wrong_worker = wc.ClaimedDispatch(did, d.organization_id, "w-2", d.claimed_at)
        assert wc.finish_external_dispatch(s, wrong_worker, new_state=S.SENT) is False
        wrong_org = wc.ClaimedDispatch(did, "org-zzz", "w-1", d.claimed_at)
        assert wc.finish_external_dispatch(s, wrong_org, new_state=S.SENT) is False
        s.commit()
    assert row(factory, did)[0].state == S.SENDING.value


def test_finish_can_never_target_queued(factory):
    did = seed(factory, state=S.SENDING.value, claimed_by="w-1", claimed_at=age(5))
    with factory.sm() as s:
        d = s.get(ExternalDispatch, did)
        claim = wc.ClaimedDispatch(did, d.organization_id, "w-1", d.claimed_at)
        with pytest.raises(ValueError):
            wc.finish_external_dispatch(s, claim, new_state=S.QUEUED)


def test_transaction_c_retries_a_transient_db_error(factory, runtime, fake, monkeypatch):
    real = w.finish_external_dispatch
    calls = {"n": 0}

    def flaky(db, claim, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OperationalError("stmt", {}, Exception("connection reset"))
        return real(db, claim, **kw)

    monkeypatch.setattr(w, "finish_external_dispatch", flaky)
    monkeypatch.setattr(w, "_TXN_C_RETRY_DELAY_SECONDS", 0)
    did = seed(factory)
    res = run(factory, runtime)
    assert res.persisted is True and row(factory, did)[0].state == S.SENT.value
    assert len(fake.calls) == 1                                       # retrying C never resends


def test_transaction_c_exhausted_leaves_sending_for_the_lease_sweep_to_make_unknown(factory, runtime, fake, monkeypatch):
    def always_fail(db, claim, **kw):
        raise OperationalError("stmt", {}, Exception("db down"))

    monkeypatch.setattr(w, "finish_external_dispatch", always_fail)
    monkeypatch.setattr(w, "_TXN_C_RETRY_DELAY_SECONDS", 0)
    did = seed(factory)
    res = run(factory, runtime)
    assert res.persisted is False and len(fake.calls) == 1
    assert row(factory, did)[0].state == S.SENDING.value              # Case H shape
    with factory.sm() as s:
        assert len(wc.recover_expired_external_dispatches(s, lease_seconds=-1)) == 1
        s.commit()
    assert row(factory, did)[0].state == S.UNKNOWN.value
    assert len(fake.calls) == 1                                       # and still one send


# ---------------------------------------------------------------------- cycle

def test_claims_are_one_at_a_time_so_leases_do_not_age(factory, runtime, live, monkeypatch):
    ids = [seed(factory, email=f"l{i}@example.com") for i in range(3)]
    snapshots = []

    def during(kw):
        with factory.sm() as s:
            snapshots.append([s.get(ExternalDispatch, i).state for i in ids])

    monkeypatch.setattr(w, "send_email", FakeSender(on_call=during))
    results = w.run_external_dispatch_cycle(session_factory=factory, worker_id=WORKER, runtime=runtime)

    assert [r.outcome for r in results] == ["sent"] * 3
    assert snapshots[0] == ["sending", "queued", "queued"]              # others still QUEUED
    assert snapshots[1] == ["sent", "sending", "queued"]


def test_cycle_respects_the_per_cycle_cap(factory, runtime, fake):
    for i in range(4):
        seed(factory, email=f"l{i}@example.com")
    got = w.run_external_dispatch_cycle(session_factory=factory, worker_id=WORKER, runtime=runtime, max_items=2)
    assert len(got) == 2


def test_two_workers_never_send_the_same_dispatch_twice(factory, live, monkeypatch):
    f = FakeSender()
    monkeypatch.setattr(w, "send_email", f)
    ids = [seed(factory, email=f"l{i}@example.com") for i in range(4)]
    r1, r2 = w.DispatchRuntime(), w.DispatchRuntime()
    out = []
    for _ in range(3):
        for wid, rt in (("w-1", r1), ("w-2", r2)):
            out += w.run_external_dispatch_cycle(session_factory=factory, worker_id=wid, runtime=rt, max_items=1)
    assert sorted(r.dispatch_id for r in out) == sorted(ids)
    assert len(f.calls) == 4 and len({c["to_email"] for c in f.calls}) == 4
