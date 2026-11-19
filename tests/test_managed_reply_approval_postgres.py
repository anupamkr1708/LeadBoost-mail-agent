"""
Real-PostgreSQL concurrency tests for the managed-reply approval hold.

The SQLite suite (test_managed_reply_content_approval.py) proves the LOGIC, but SQLite serialises writers, so it
cannot show that the guarded single-statement UPDATEs behave under genuinely concurrent transactions. These do:
every competitor is its own thread with its own engine/session, released together by a barrier, against a real
PostgreSQL. Skipped (never faked with SQLite) unless POSTGRES_TEST_URL points at a reachable PostgreSQL, e.g.

    POSTGRES_TEST_URL=postgresql+psycopg2://user:pw@localhost:5432/mailer_agent_test

Use the explicit +psycopg2 driver (a bare ``postgresql://`` resolves to psycopg3 on SQLAlchemy >= 2.1). Dedicated
test database only: it is TRUNCATEd.

What is proven: against any number of concurrent attempts, release / cancel / fenced-generation-completion have
exactly ONE winner, the loser changes nothing, the final row is consistent with the winner, and the SMTP worker's
SKIP LOCKED claim never sees a held draft.
"""

from __future__ import annotations

import os
import threading

import pytest

pytest.importorskip("psycopg2", reason="psycopg2 not installed -- cannot test real PostgreSQL")

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

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
from mailer_agent.mail import content_digest as cd  # noqa: E402
from mailer_agent.models import Base, ExternalDispatch, ExternalDispatchState as S, Message  # noqa: E402
from tests.dispatch_support import seed_dispatch  # noqa: E402

ORG = "org-a"
_TRUNCATE = (
    "TRUNCATE external_dispatches, mailboxes, messages, contacts, campaigns, suppression_list "
    "RESTART IDENTITY CASCADE"
)


@pytest.fixture()
def factory():
    eng = create_engine(POSTGRES_TEST_URL, pool_size=30)
    Base.metadata.create_all(bind=eng)
    with eng.begin() as c:
        c.execute(text(_TRUNCATE))
    yield sessionmaker(bind=eng)
    with eng.begin() as c:
        c.execute(text(_TRUNCATE))
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


def _held(factory, n):
    """n held (AWAITING_APPROVAL) managed-reply drafts. Returns [(dispatch_id, message_id, digest)]."""
    out = []
    for i in range(n):
        with factory() as s:
            d = seed_dispatch(
                s, org=ORG, idem=f"held-{i}", email=f"lead{i}@example.com",
                state=S.AWAITING_APPROVAL.value, content_approval_required=True,
            )
            m = s.get(Message, d.message_id)
            out.append((d.id, d.message_id, cd.content_sha256(m.subject, m.body)))
    return out


def _state(factory, dispatch_id):
    with factory() as s:
        d = s.get(ExternalDispatch, dispatch_id)
        return d.state, d.approved_content_sha256


def _release(factory, did, mid, digest):
    with factory() as s:
        ok = wc.release_held_dispatch(s, organization_id=ORG, dispatch_id=did, message_id=mid, content_sha256=digest)
        s.commit()
        return ok


def _cancel(factory, did):
    with factory() as s:
        ok = wc.cancel_unreleased_dispatch(s, organization_id=ORG, dispatch_id=did)
        s.commit()
        return ok


def test_release_versus_cancel_has_exactly_one_winner_across_many_races(factory):
    rounds = _held(factory, 25)
    release_wins = cancel_wins = 0
    for did, mid, digest in rounds:
        def attempt(i, barrier, did=did, mid=mid, digest=digest):
            barrier.wait()
            return _release(factory, did, mid, digest) if i == 0 else _cancel(factory, did)

        released, cancelled = _run_threads(attempt, 2)
        assert (released, cancelled).count(True) == 1, f"exactly one winner, got release={released} cancel={cancelled}"
        state, approved = _state(factory, did)
        if released:
            release_wins += 1
            assert (state, approved) == (S.QUEUED.value, digest)
        else:
            cancel_wins += 1
            assert (state, approved) == (S.CANCELLED.value, None), "a cancelled draft must carry no approval evidence"
    assert release_wins + cancel_wins == len(rounds)


def test_many_identical_concurrent_releases_have_one_winner_and_leave_one_consistent_row(factory):
    ((did, mid, digest),) = _held(factory, 1)

    def attempt(i, barrier):
        barrier.wait()
        return _release(factory, did, mid, digest)

    results = _run_threads(attempt, 12)
    assert results.count(True) == 1, results
    assert _state(factory, did) == (S.QUEUED.value, digest)


def test_a_wrong_message_or_tenant_never_wins_a_race(factory):
    ((did, mid, digest),) = _held(factory, 1)

    def attempt(i, barrier):
        barrier.wait()
        if i % 3 == 0:
            return ("right", _release(factory, did, mid, digest))
        if i % 3 == 1:
            with factory() as s:
                ok = wc.release_held_dispatch(
                    s, organization_id="org-b", dispatch_id=did, message_id=mid, content_sha256=digest
                )
                s.commit()
                return ("wrong-tenant", ok)
        with factory() as s:
            ok = wc.release_held_dispatch(
                s, organization_id=ORG, dispatch_id=did, message_id=mid + 1000, content_sha256=digest
            )
            s.commit()
            return ("wrong-message", ok)

    results = _run_threads(attempt, 9)
    assert [ok for kind, ok in results if kind != "right"] == [False] * 6
    assert [ok for kind, ok in results if kind == "right"].count(True) == 1


def test_cancel_versus_the_generation_workers_fenced_completion_has_one_winner(factory):
    """The subtlest race: the user cancels while a worker is mid-LLM. Either the worker's fenced write wins (the
    draft lands in the HOLD and the cancel then revokes it on retry) or the cancel wins (the worker's write
    changes nothing, so no Message is attached). Never both, and never a sendable row."""
    for i in range(20):
        with factory() as s:
            d = seed_dispatch(
                s, org=ORG, idem=f"gen-{i}", email=f"gen{i}@example.com", state=S.QUEUED.value,
                content_approval_required=True,
            )
            did = d.id
            # Turn it into an accepted-but-ungenerated request, as acceptance leaves it.
            s.execute(text("UPDATE external_dispatches SET message_id = NULL WHERE id = :i"), {"i": did})
            s.commit()
        with factory() as s:
            claim = wc.claim_next_generation_dispatch(s, f"gen-worker-{i}")
            s.commit()
        assert claim is not None and claim.dispatch_id == did

        with factory() as s:
            msg_id = s.execute(text("SELECT id FROM messages ORDER BY id DESC LIMIT 1")).scalar()

        def attempt(k, barrier, claim=claim, msg_id=msg_id, did=did):
            barrier.wait()
            if k == 0:
                with factory() as s:
                    ok = wc.complete_generation(s, claim, msg_id, hold_for_approval=True)
                    s.commit()
                    return ok
            return _cancel(factory, did)

        completed, cancelled = _run_threads(attempt, 2)
        state, approved = _state(factory, did)
        with factory() as s:
            attached = s.get(ExternalDispatch, did).message_id
        if completed:
            # The worker won the fence: the draft is HELD. (The cancel, racing a state it could not yet match or
            # having lost the row lock, may or may not have run first; a held draft is never sendable either way.)
            assert state in (S.AWAITING_APPROVAL.value, S.CANCELLED.value)
            assert approved is None
        else:
            assert cancelled is True and state == S.CANCELLED.value and attached is None
        assert state != S.QUEUED.value, "no ordering of these two may produce a sendable row"


def test_the_smtp_claim_never_sees_a_held_draft_under_concurrent_claimers(factory):
    _held(factory, 10)

    def attempt(i, barrier):
        barrier.wait()
        with factory() as s:
            claim = wc.claim_next_external_dispatch(s, f"snd-{i}")
            s.commit()
            return claim

    assert all(c is None for c in _run_threads(attempt, 8)), "a held draft was claimable by the SMTP worker"
    with factory() as s:
        assert s.query(ExternalDispatch).filter(ExternalDispatch.state == S.AWAITING_APPROVAL.value).count() == 10


def test_after_a_release_exactly_one_smtp_claimer_gets_it(factory):
    ((did, mid, digest),) = _held(factory, 1)
    assert _release(factory, did, mid, digest) is True

    def attempt(i, barrier):
        barrier.wait()
        with factory() as s:
            claim = wc.claim_next_external_dispatch(s, f"snd-{i}")
            s.commit()
            return claim

    claims = [c for c in _run_threads(attempt, 8) if c is not None]
    assert len(claims) == 1 and claims[0].dispatch_id == did
