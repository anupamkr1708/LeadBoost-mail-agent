"""
Failure matrix (spec cases A-H) for the async ExternalDispatch path.

A "crash" is simulated by raising a BaseException (SystemExit) at the exact
seam: unlike Exception, it is not caught by the worker's error handling, so
it unwinds like a dying process -- only the database state that was already
COMMITTED survives, which is precisely what a real crash leaves behind.

Recovery then runs with an artificially expired lease, and the assertion in
every ambiguous case is the same: the row ends UNKNOWN, never QUEUED, and
no further SMTP attempt is ever made.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from mailer_agent.followup import work_claiming as wc
from mailer_agent.mail import external_dispatch_worker as w
from mailer_agent.mail import sender
from mailer_agent.mail.sender import SendOutcome
from mailer_agent.models import Base, ExternalDispatch, ExternalDispatchState as S, Message, MessageStatus
from tests.dispatch_support import FakeSender, FakeSMTPServer, TrackingFactory, age, seed_dispatch


@pytest.fixture()
def factory(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path/'c.db'}", connect_args={"check_same_thread": False, "timeout": 15})
    Base.metadata.create_all(bind=eng)
    yield TrackingFactory(sessionmaker(bind=eng))
    eng.dispose()


@pytest.fixture(autouse=True)
def live(monkeypatch):
    monkeypatch.setattr(w.settings, "live_sending_enabled", True)


def _row(factory, did):
    with factory.sm() as s:
        d = s.get(ExternalDispatch, did)
        m = s.get(Message, d.message_id)
        return d.state, d.claimed_by, d.claimed_at, m.status, m.message_id_header


def _seed(factory, **kw):
    with factory.sm() as s:
        return seed_dispatch(s, **kw).id


def _cycle(factory, n=1):
    rt = w.DispatchRuntime()
    return [w.run_external_dispatch_cycle(session_factory=factory, worker_id="w-next", runtime=rt) for _ in range(n)]


def _expire_and_recover(factory):
    with factory.sm() as s:
        out = wc.recover_expired_external_dispatches(s, lease_seconds=-1)
        s.commit()
    return out


class Dead(BaseException):
    """Stands in for the process dying (not an Exception)."""


# A -- before the HTTP acceptance commit nothing exists; retry is safe. That
# boundary belongs to the HTTP endpoint and is proven in
# tests/test_leadboost_integration.py (idempotency / replay / race tests).


def test_B_queued_row_is_claimed_and_processed_by_a_later_worker(factory, monkeypatch):
    f = FakeSender()
    monkeypatch.setattr(w, "send_email", f)
    did = _seed(factory)
    assert _row(factory, did)[0] == S.QUEUED.value
    _cycle(factory)
    assert _row(factory, did)[0] == S.SENT.value and len(f.calls) == 1


def test_C_crash_before_claim_transaction_commits_leaves_row_queued_and_claimable(factory, monkeypatch):
    f = FakeSender()
    monkeypatch.setattr(w, "send_email", f)
    did = _seed(factory)

    def die(*a, **k):
        raise Dead()

    monkeypatch.setattr(w, "evaluate_exact_message_grounding", die)   # after claim, before commit
    with pytest.raises(Dead):
        w.process_next_external_dispatch(session_factory=factory, worker_id="w-dead", runtime=w.DispatchRuntime())

    assert _row(factory, did)[:2] == (S.QUEUED.value, None)            # claim rolled back
    assert _row(factory, did)[4] is None                                # no Message-ID either
    monkeypatch.undo()
    monkeypatch.setattr(w.settings, "live_sending_enabled", True)
    monkeypatch.setattr(w, "send_email", f)
    _cycle(factory)
    assert _row(factory, did)[0] == S.SENT.value and len(f.calls) == 1


def test_C_db_error_before_commit_also_leaves_queued(factory, monkeypatch):
    f = FakeSender()
    monkeypatch.setattr(w, "send_email", f)
    did = _seed(factory)
    monkeypatch.setattr(w, "generate_message_id",
                        lambda e: (_ for _ in ()).throw(OperationalError("s", {}, Exception("gone"))))
    assert w.run_external_dispatch_cycle(session_factory=factory, worker_id="w", runtime=w.DispatchRuntime()) == []
    assert _row(factory, did)[0] == S.QUEUED.value and f.calls == []


def test_D_crash_after_sending_commit_before_smtp_ends_unknown_never_queued(factory, monkeypatch):
    f = FakeSender()
    monkeypatch.setattr(w, "send_email", f)
    did = _seed(factory)

    def die(*a, **k):
        raise Dead()

    monkeypatch.setattr(w, "_execute_and_finalize", die)               # B has committed
    with pytest.raises(Dead):
        w.process_next_external_dispatch(session_factory=factory, worker_id="w-dead", runtime=w.DispatchRuntime())
    monkeypatch.undo()
    monkeypatch.setattr(w.settings, "live_sending_enabled", True)
    monkeypatch.setattr(w, "send_email", f)

    state, claimed_by, claimed_at, msg_status, msg_id = _row(factory, did)
    assert state == S.SENDING.value and claimed_by == "w-dead" and claimed_at is not None
    assert msg_id is not None and msg_status == MessageStatus.SENDING.value    # ID durable

    assert _cycle(factory, 2) == [[], []]                              # SENDING is not claimable
    assert len(_expire_and_recover(factory)) == 1
    assert _row(factory, did)[0] == S.UNKNOWN.value
    assert _cycle(factory, 3) == [[], [], []]
    assert f.calls == []                                               # never sent by anyone
    assert _row(factory, did)[0] == S.UNKNOWN.value                    # and never QUEUED


def test_E_ambiguous_smtp_outcome_is_unknown_and_never_resent(factory, monkeypatch):
    f = FakeSender(outcome=SendOutcome.UNKNOWN, error="connection dropped after DATA")
    monkeypatch.setattr(w, "send_email", f)
    did = _seed(factory)
    _cycle(factory)
    for _ in range(3):
        _cycle(factory)
        _expire_and_recover(factory)
    assert _row(factory, did)[0] == S.UNKNOWN.value and len(f.calls) == 1


def test_F_smtp_success_commits_sent_and_releases_the_claim(factory, monkeypatch):
    with FakeSMTPServer() as srv:
        s = sender.settings
        for k, v in dict(smtp_host="127.0.0.1", smtp_port=srv.port, smtp_use_tls=False,
                         smtp_username="u", smtp_password="p").items():
            monkeypatch.setattr(s, k, v)
        did = _seed(factory)
        _cycle(factory)
        state, claimed_by, claimed_at, msg_status, msg_id = _row(factory, did)
        assert (state, claimed_by, claimed_at, msg_status) == (S.SENT.value, None, None, MessageStatus.SENT.value)
        assert len(srv.accepted) == 1 and msg_id in srv.accepted[0]


def test_G_definite_failure_is_failed_and_not_blindly_retried(factory, monkeypatch):
    f = FakeSender(outcome=SendOutcome.FAILED, error="connection refused")
    monkeypatch.setattr(w, "send_email", f)
    did = _seed(factory)
    _cycle(factory, 4)
    _expire_and_recover(factory)
    state, claimed_by, *_ = _row(factory, did)
    assert state == S.FAILED.value and claimed_by is None and len(f.calls) == 1


def test_H_smtp_accepted_then_crash_before_transaction_c_ends_unknown_with_one_delivery(factory, monkeypatch):
    with FakeSMTPServer() as srv:
        s = sender.settings
        for k, v in dict(smtp_host="127.0.0.1", smtp_port=srv.port, smtp_use_tls=False,
                         smtp_username="u", smtp_password="p").items():
            monkeypatch.setattr(s, k, v)
        did = _seed(factory)

        def die(*a, **k):
            raise Dead()

        monkeypatch.setattr(w, "_transaction_c", die)                  # SMTP already accepted it
        with pytest.raises(Dead):
            w.process_next_external_dispatch(session_factory=factory, worker_id="w-dead", runtime=w.DispatchRuntime())
        monkeypatch.undo()
        monkeypatch.setattr(w.settings, "live_sending_enabled", True)
        for k, v in dict(smtp_host="127.0.0.1", smtp_port=srv.port, smtp_use_tls=False,
                         smtp_username="u", smtp_password="p").items():
            monkeypatch.setattr(sender.settings, k, v)

        assert len(srv.accepted) == 1
        assert _row(factory, did)[0] == S.SENDING.value                # indistinguishable from D
        assert len(_expire_and_recover(factory)) == 1
        _cycle(factory, 3)
        assert _row(factory, did)[0] == S.UNKNOWN.value
        assert len(srv.accepted) == 1                                  # no duplicate, ever
        assert _row(factory, did)[4] in srv.accepted[0]                # stored ID is the transmitted ID


def test_expired_sending_never_produces_another_smtp_attempt(factory, monkeypatch):
    f = FakeSender(raises=AssertionError("SMTP must never be attempted for an expired SENDING row"))
    monkeypatch.setattr(w, "send_email", f)
    did = _seed(factory, state=S.SENDING.value, claimed_by="w-dead", claimed_at=age(99999))
    for _ in range(3):
        _cycle(factory)
        w.run_external_dispatch_lease_recovery(session_factory=factory)
    assert f.calls == [] and _row(factory, did)[0] == S.UNKNOWN.value


def test_fresh_sending_row_is_neither_claimed_nor_recovered(factory, monkeypatch):
    f = FakeSender()
    monkeypatch.setattr(w, "send_email", f)
    did = _seed(factory, state=S.SENDING.value, claimed_by="w-live", claimed_at=age(30))
    _cycle(factory)
    assert w.run_external_dispatch_lease_recovery(session_factory=factory) == 0
    assert _row(factory, did)[:2] == (S.SENDING.value, "w-live") and f.calls == []


def test_recovery_job_helper_reports_count_and_is_idempotent(factory):
    _seed(factory, state=S.SENDING.value, claimed_by="d", claimed_at=age(99999))
    assert w.run_external_dispatch_lease_recovery(session_factory=factory) == 1
    assert w.run_external_dispatch_lease_recovery(session_factory=factory) == 0
