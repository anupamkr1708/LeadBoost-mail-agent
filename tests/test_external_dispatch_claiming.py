"""
C6 -- ExternalDispatch worker claiming, lease recovery, worker identity.

SQLite covers the guarded-UPDATE logic and state transitions. It does NOT
prove PostgreSQL row-locking; that is tests/test_external_dispatch_postgres.py,
run against a real PostgreSQL.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from mailer_agent.config import get_settings
from mailer_agent.followup import work_claiming as wc
from mailer_agent.models import (
    Base,
    ExternalDispatch,
    ExternalDispatchState as S,
    Message,
    MessageStatus,
)
from tests.dispatch_support import age, naive_now, seed_dispatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def factory(tmp_path):
    eng = create_engine(
        f"sqlite:///{tmp_path/'t.db'}", connect_args={"check_same_thread": False, "timeout": 15}
    )
    Base.metadata.create_all(bind=eng)
    yield sessionmaker(bind=eng)
    eng.dispose()


@pytest.fixture()
def db(factory):
    s = factory()
    yield s
    s.close()


def _row(factory, dispatch_id):
    with factory() as s:
        return s.get(ExternalDispatch, dispatch_id)


# ----------------------------------------------------------------- claiming

def test_claims_one_queued_dispatch_and_populates_lease_fields(db, factory):
    d = seed_dispatch(db)
    claim = wc.claim_next_external_dispatch(db, "w-1")
    db.commit()

    assert claim is not None and claim.dispatch_id == d.id
    assert claim.organization_id == "org-a" and claim.worker_id == "w-1"
    row = _row(factory, d.id)
    assert row.state == S.SENDING.value
    assert row.claimed_by == "w-1"
    assert row.claimed_at == claim.claimed_at  # the fencing token is what was written


def test_claim_returns_none_when_nothing_queued(db):
    assert wc.claim_next_external_dispatch(db, "w-1") is None


def test_oldest_queued_is_claimed_first_and_one_at_a_time(db):
    a = seed_dispatch(db, email="a@example.com")
    b = seed_dispatch(db, email="b@example.com")
    first = wc.claim_next_external_dispatch(db, "w-1")
    db.commit()
    second = wc.claim_next_external_dispatch(db, "w-1")
    db.commit()
    assert (first.dispatch_id, second.dispatch_id) == (a.id, b.id)
    assert wc.claim_next_external_dispatch(db, "w-1") is None


def test_two_workers_cannot_claim_the_same_dispatch(factory):
    with factory() as s:
        did = seed_dispatch(s).id
    s1, s2 = factory(), factory()
    try:
        c1 = wc.claim_next_external_dispatch(s1, "w-1")
        s1.commit()
        c2 = wc.claim_next_external_dispatch(s2, "w-2")
        s2.commit()
    finally:
        s1.close(); s2.close()
    assert c1 is not None and c1.dispatch_id == did
    assert c2 is None
    assert _row(factory, did).claimed_by == "w-1"


@pytest.mark.parametrize("state", [S.SENDING, S.SENT, S.FAILED, S.UNKNOWN])
def test_only_queued_is_claimable(db, state):
    # includes an EXPIRED sending row: stale SENDING is never claimed.
    seed_dispatch(
        db, state=state.value,
        claimed_by="old" if state is S.SENDING else None,
        claimed_at=age(99999) if state is S.SENDING else None,
    )
    assert wc.claim_next_external_dispatch(db, "w-1") is None


def test_uncommitted_claim_rolls_back_to_queued(factory):
    """Case C: crash before the claim transaction commits."""
    with factory() as s:
        did = seed_dispatch(s).id
    crashed = factory()
    assert wc.claim_next_external_dispatch(crashed, "w-1") is not None
    crashed.rollback(); crashed.close()  # process dies, txn never committed

    row = _row(factory, did)
    assert row.state == S.QUEUED.value and row.claimed_by is None and row.claimed_at is None
    with factory() as s:
        assert wc.claim_next_external_dispatch(s, "w-2") is not None  # claimable again


# ------------------------------------------------------------------ recovery

def _recover(factory, **kw):
    with factory() as s:
        out = wc.recover_expired_external_dispatches(s, **kw)
        s.commit()
    return out


def test_fresh_sending_is_not_recovered(db, factory):
    d = seed_dispatch(db, state=S.SENDING.value, claimed_by="w", claimed_at=age(10))
    assert _recover(factory) == []
    assert _row(factory, d.id).state == S.SENDING.value


def test_expired_sending_becomes_unknown_never_queued(db, factory):
    d = seed_dispatch(db, state=S.SENDING.value, claimed_by="w-dead", claimed_at=age(1000))
    with factory() as s:  # mirror what Transaction B leaves behind
        s.get(Message, d.message_id).status = MessageStatus.SENDING.value
        s.commit()

    out = _recover(factory)

    assert [r.dispatch_id for r in out] == [d.id]
    row = _row(factory, d.id)
    assert row.state == S.UNKNOWN.value
    assert row.claimed_by is None and row.claimed_at is None
    assert "lease_expired" in row.error_message and "NOT be retried" in row.error_message
    with factory() as s:
        m = s.get(Message, d.message_id)
        assert m.status == MessageStatus.UNKNOWN.value
    # and it is not claimable afterwards
    with factory() as s:
        assert wc.claim_next_external_dispatch(s, "w-2") is None


def test_recovery_is_idempotent_and_unknown_stays_unknown(db, factory):
    d = seed_dispatch(db, state=S.SENDING.value, claimed_by="w", claimed_at=age(5000))
    assert len(_recover(factory)) == 1
    assert _recover(factory) == []
    assert _recover(factory) == []
    assert _row(factory, d.id).state == S.UNKNOWN.value


def test_already_unknown_is_left_alone(db, factory):
    d = seed_dispatch(db, state=S.UNKNOWN.value)
    assert _recover(factory) == []
    assert _row(factory, d.id).state == S.UNKNOWN.value


def test_recovery_uses_dedicated_lease_not_claim_lease_seconds(db, factory):
    """Aged past the 300 s contact lease but inside the 900 s dispatch lease."""
    assert wc.CLAIM_LEASE_SECONDS == 300
    assert get_settings().external_dispatch_lease_seconds == 900
    d = seed_dispatch(db, state=S.SENDING.value, claimed_by="w", claimed_at=age(400))
    assert _recover(factory) == []
    assert _row(factory, d.id).state == S.SENDING.value
    assert len(_recover(factory, lease_seconds=300)) == 1  # explicit override still works


def test_sending_with_null_claimed_at_is_recovered(db, factory):
    d = seed_dispatch(db, state=S.SENDING.value, claimed_by=None, claimed_at=None)
    assert len(_recover(factory)) == 1
    assert _row(factory, d.id).state == S.UNKNOWN.value


def test_recovery_only_touches_expired_sending_rows(db, factory):
    q = seed_dispatch(db, email="q@example.com")
    sent = seed_dispatch(db, email="s@example.com", state=S.SENT.value)
    failed = seed_dispatch(db, email="f@example.com", state=S.FAILED.value)
    _recover(factory)
    assert _row(factory, q.id).state == S.QUEUED.value
    assert _row(factory, sent.id).state == S.SENT.value
    assert _row(factory, failed.id).state == S.FAILED.value


def test_two_recovery_sweeps_yield_one_terminal_result(db, factory):
    d = seed_dispatch(db, state=S.SENDING.value, claimed_by="w", claimed_at=age(2000))
    s1, s2 = factory(), factory()
    try:
        r1 = wc.recover_expired_external_dispatches(s1)
        s1.commit()
        r2 = wc.recover_expired_external_dispatches(s2)
        s2.commit()
    finally:
        s1.close(); s2.close()
    assert (len(r1), len(r2)) == (1, 0)
    assert _row(factory, d.id).state == S.UNKNOWN.value


def test_recovery_module_cannot_send(db):
    """Recovery lives in work_claiming, which must not load the sender."""
    code = (
        "import sys, mailer_agent.followup.work_claiming;"
        "bad=[m for m in sys.modules if m in ('mailer_agent.mail.sender','smtplib')"
        " or m.startswith(('mailer_agent.llm.agent','mailer_agent.llm.provider'))];"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


# ------------------------------------------------------------- worker identity

def test_worker_id_is_process_unique_and_stable(monkeypatch):
    monkeypatch.delenv("WORKER_ID", raising=False)
    a = wc.make_worker_id()
    assert a == wc.make_worker_id()  # stable within a process
    monkeypatch.setattr(os, "getpid", lambda: 424242)
    b = wc.make_worker_id()
    assert b != a and "424242" in b  # a sibling process on the same host differs


def test_explicit_worker_id_env_still_wins(monkeypatch):
    monkeypatch.setenv("WORKER_ID", "render-instance-7")
    assert wc.make_worker_id() == "render-instance-7"
