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
            "TRUNCATE external_dispatches, messages, contacts, campaigns, "
            "suppression_list RESTART IDENTITY CASCADE"
        ))
    yield sessionmaker(bind=eng)
    with eng.begin() as c:
        c.execute(text(
            "TRUNCATE external_dispatches, messages, contacts, campaigns, "
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
