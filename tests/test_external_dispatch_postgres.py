"""
Real-PostgreSQL concurrency tests for ExternalDispatch claiming/recovery.

Skipped (never faked with SQLite) unless POSTGRES_TEST_URL points at a
reachable PostgreSQL, e.g.
    POSTGRES_TEST_URL=postgresql+psycopg2://user:pw@localhost:5432/mailer_test
Use the explicit +psycopg2 driver: with SQLAlchemy >= 2.1 a bare
``postgresql://`` resolves to psycopg3, which requirements.txt does not pin.

Each worker is its own thread with its own engine/session, and claims are
held UNCOMMITTED across a barrier -- that is what proves FOR UPDATE SKIP
LOCKED gives concurrent transactions different rows instead of blocking or
double-claiming. Dedicated test database only: it is TRUNCATEd.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

pytest.importorskip("psycopg2", reason="psycopg2 not installed -- cannot test real PostgreSQL")

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

from mailer_agent.followup import work_claiming as wc  # noqa: E402
from mailer_agent.models import Base, ExternalDispatch, ExternalDispatchState as S  # noqa: E402
from tests.dispatch_support import age, seed_dispatch  # noqa: E402


@pytest.fixture()
def factory():
    eng = create_engine(POSTGRES_TEST_URL, pool_size=20)
    Base.metadata.create_all(bind=eng)
    with eng.begin() as c:
        c.execute(text(
            "TRUNCATE external_dispatches, mailboxes, messages, contacts, campaigns, "
            "suppression_list RESTART IDENTITY CASCADE"
        ))
    yield sessionmaker(bind=eng)
    with eng.begin() as c:
        c.execute(text(
            "TRUNCATE external_dispatches, mailboxes, messages, contacts, campaigns, "
            "suppression_list RESTART IDENTITY CASCADE"
        ))
    eng.dispose()


def _run_threads(fn, n):
    barrier = threading.Barrier(n)
    results, errors = [None] * n, []

    def target(i):
        try:
            results[i] = fn(i, barrier)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=target, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    return results


def test_concurrent_uncommitted_claims_get_distinct_rows(factory):
    n = 8
    with factory() as s:
        ids = sorted(seed_dispatch(s, email=f"l{i}@example.com").id for i in range(n))

    def worker(i, barrier):
        s = factory()
        try:
            claim = wc.claim_next_external_dispatch(s, f"w-{i}")
            barrier.wait(timeout=30)   # every claim is held UNCOMMITTED here
            s.commit()
            return claim.dispatch_id if claim else None
        finally:
            s.close()

    got = _run_threads(worker, n)
    assert None not in got                      # nobody blocked out or starved
    assert sorted(got) == ids                   # each row claimed exactly once
    with factory() as s:
        rows = s.query(ExternalDispatch).all()
        assert {r.state for r in rows} == {S.SENDING.value}
        assert len({r.claimed_by for r in rows}) == n


def test_single_row_is_claimed_by_exactly_one_of_many(factory):
    with factory() as s:
        did = seed_dispatch(s).id

    def worker(i, barrier):
        s = factory()
        try:
            barrier.wait(timeout=30)
            t0 = time.monotonic()
            claim = wc.claim_next_external_dispatch(s, f"w-{i}")
            waited = time.monotonic() - t0
            time.sleep(0.3)            # winner holds its lock; losers must not have waited on it
            s.commit()
            return (claim.dispatch_id if claim else None, waited)
        finally:
            s.close()

    got = _run_threads(worker, 6)
    winners = [r for r, _ in got if r is not None]
    assert winners == [did]
    assert max(w for _, w in got) < 0.25       # SKIP LOCKED: losers returned, did not block


def test_only_queued_claimable_on_postgres(factory):
    with factory() as s:
        seed_dispatch(s, email="a@example.com", state=S.SENDING.value, claimed_by="x", claimed_at=age(99999))
        seed_dispatch(s, email="b@example.com", state=S.UNKNOWN.value)
        seed_dispatch(s, email="c@example.com", state=S.SENT.value)
    with factory() as s:
        assert wc.claim_next_external_dispatch(s, "w") is None


def test_concurrent_recovery_has_one_terminal_result(factory):
    with factory() as s:
        did = seed_dispatch(s, state=S.SENDING.value, claimed_by="dead", claimed_at=age(5000)).id

    def worker(i, barrier):
        s = factory()
        try:
            barrier.wait(timeout=30)
            out = wc.recover_expired_external_dispatches(s)
            time.sleep(0.3)            # first sweeper holds the row lock, uncommitted
            s.commit()
            return [r.dispatch_id for r in out]
        finally:
            s.close()

    got = _run_threads(worker, 4)
    assert sorted(x for r in got for x in r) == [did]     # recovered exactly once in total
    with factory() as s:
        row = s.get(ExternalDispatch, did)
        assert row.state == S.UNKNOWN.value and row.claimed_by is None
        assert wc.claim_next_external_dispatch(s, "w") is None    # never back to QUEUED


# ---------------------------------------------------------------------------
# Worker pipeline on real PostgreSQL
# ---------------------------------------------------------------------------

from mailer_agent.mail import external_dispatch_worker as w  # noqa: E402
from mailer_agent.models import Message, MessageStatus  # noqa: E402
from tests.dispatch_support import FakeSender  # noqa: E402


@pytest.fixture()
def live(monkeypatch):
    monkeypatch.setattr(w.settings, "live_sending_enabled", True)


def test_concurrent_workers_send_each_dispatch_exactly_once(factory, live, monkeypatch):
    n_rows, n_workers = 24, 4
    with factory() as s:
        for i in range(n_rows):
            seed_dispatch(s, email=f"lead{i}@example.com")
    f = FakeSender()
    monkeypatch.setattr(w, "send_email", f)

    def worker(i, barrier):
        rt = w.DispatchRuntime()
        barrier.wait(timeout=30)
        done = 0
        while True:
            got = w.run_external_dispatch_cycle(
                session_factory=factory, worker_id=f"w-{i}", runtime=rt, max_items=3)
            if not got:
                return done
            done += len(got)

    counts = _run_threads(worker, n_workers)

    assert sum(counts) == n_rows
    recipients = [c["to_email"] for c in f.calls]
    assert len(recipients) == n_rows and len(set(recipients)) == n_rows      # no duplicate send
    with factory() as s:
        rows = s.query(ExternalDispatch).all()
        assert {r.state for r in rows} == {S.SENT.value}
        assert all(r.claimed_by is None and r.claimed_at is None for r in rows)
        ids = [m.message_id_header for m in s.query(Message).all()]
        assert None not in ids and len(set(ids)) == n_rows
        assert {c["message_id_header"] for c in f.calls} == set(ids)          # stored == transmitted


def test_no_transaction_is_open_on_postgres_while_smtp_runs(factory, live, monkeypatch):
    with factory() as s:
        did = seed_dispatch(s).id
    seen = {}

    def during_smtp(kw):
        with factory() as other:
            d = other.get(ExternalDispatch, did)
            m = other.get(Message, d.message_id)
            seen["state"], seen["msg_status"], seen["db_id"] = d.state, m.status, m.message_id_header
            seen["idle_in_txn"] = other.execute(text(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_backend_pid() "
                "AND state LIKE 'idle in transaction%'"
            )).scalar()
            other.rollback()
        seen["kw_id"] = kw["message_id_header"]

    monkeypatch.setattr(w, "send_email", FakeSender(on_call=during_smtp))
    res = w.process_next_external_dispatch(session_factory=factory, worker_id="w-1", runtime=w.DispatchRuntime())

    assert res.outcome == "sent"
    assert seen["state"] == S.SENDING.value and seen["msg_status"] == MessageStatus.SENDING.value
    assert seen["idle_in_txn"] == 0                       # no DB transaction open during SMTP
    assert seen["db_id"] == seen["kw_id"] == res.message_id_header


def test_fence_refuses_a_late_success_after_lease_recovery_on_postgres(factory, live, monkeypatch):
    with factory() as s:
        did = seed_dispatch(s).id

    def expire_mid_smtp(kw):
        with factory() as s:
            assert len(wc.recover_expired_external_dispatches(s, lease_seconds=-1)) == 1
            s.commit()

    f = FakeSender(on_call=expire_mid_smtp)
    monkeypatch.setattr(w, "send_email", f)
    res = w.process_next_external_dispatch(session_factory=factory, worker_id="w-1", runtime=w.DispatchRuntime())

    assert res.outcome == "sent" and res.persisted is False
    with factory() as s:
        d = s.get(ExternalDispatch, did)
        assert d.state == S.UNKNOWN.value and "late_outcome_after_lease_loss" in d.error_message
        assert wc.claim_next_external_dispatch(s, "w-2") is None
    assert len(f.calls) == 1


def test_gate_failure_is_committed_as_failed_on_postgres(factory, live, monkeypatch):
    f = FakeSender()
    monkeypatch.setattr(w, "send_email", f)
    with factory() as s:
        did = seed_dispatch(s, body="Our customers see a 40% lift in reply rates.").id
    res = w.process_next_external_dispatch(session_factory=factory, worker_id="w-1", runtime=w.DispatchRuntime())
    assert res.outcome == "failed" and f.calls == []
    with factory() as s:
        d = s.get(ExternalDispatch, did)
        assert d.state == S.FAILED.value and d.claimed_by is None
