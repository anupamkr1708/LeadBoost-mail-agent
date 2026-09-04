"""
PostgreSQL concurrency suite (spec §32/§24-26).

Honesty check, stated as plainly as everywhere else in this repository:
this file has never been executed. There is no PostgreSQL instance in
the sandbox these changes were made in. Every test below was written by
tracing the real implementation (followup/work_claiming.py,
mail/reply_handler_v2.py, models.py's new unique constraint) by hand,
the same standard as the rest of this session's work -- but concurrency
correctness is exactly the property that CANNOT be verified by tracing
alone. Do not read this file's existence as evidence the concurrency
guarantees hold. Run it (see docs/FINAL_PRODUCTION_READINESS.md) and
report back what actually happens.

Setup
-----
Requires POSTGRES_TEST_URL, e.g.:
    postgresql://user:password@localhost:5432/mailer_agent_test

The whole module is skipped (not failed) if this isn't set or the
database isn't reachable -- per this repository's standing rule, a
skip is reported as NOT VERIFIED, never silently treated as a pass.

Each test uses its own thread with its own SQLAlchemy engine/session
(sessions are not thread-safe; sharing one across threads would not
test real concurrent transactions, it would just serialize on Python's
GIL and prove nothing about database-level locking). A
threading.Barrier synchronizes threads to maximize the chance of
genuine transaction overlap, since the whole point is to exercise the
race window, not avoid it.
"""

from __future__ import annotations

import os
import threading

import pytest

pytest.importorskip("psycopg2", reason="psycopg2 not installed -- cannot test real PostgreSQL")

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

POSTGRES_TEST_URL = os.environ.get("POSTGRES_TEST_URL")


def _pg_available() -> bool:
    if not POSTGRES_TEST_URL:
        return False
    try:
        engine = create_engine(POSTGRES_TEST_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


pytestmark.append(
    pytest.mark.skipif(
        not _pg_available(),
        reason="POSTGRES_TEST_URL not set or PostgreSQL unreachable -- see module docstring",
    )
)


@pytest.fixture
def pg_engine():
    from mailer_agent.models import Base
    engine = create_engine(POSTGRES_TEST_URL)
    Base.metadata.create_all(bind=engine)
    yield engine
    # Clean up between tests -- this is a dedicated test database, safe
    # to truncate. Never point POSTGRES_TEST_URL at a real deployment.
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE messages, contacts, campaigns, suppression_list CASCADE"))
        conn.commit()
    engine.dispose()


def _new_session(engine):
    return sessionmaker(bind=engine)()


def _seed_campaign_and_contact(engine, *, status="active", next_action_at_past=True):
    from datetime import datetime, timedelta, timezone

    from mailer_agent.models import Campaign, Contact

    session = _new_session(engine)
    campaign = Campaign(
        name="Concurrency Test Campaign", organization_id="org-concurrency",
        sender_name="Test", sender_org="TestCo", sender_email="test@testco.example.com",
        value_prop="test",
    )
    session.add(campaign)
    session.flush()
    contact = Contact(
        campaign_id=campaign.id, name="Test Contact", email="concurrency-test@example.com",
        status=status,
        next_action_at=(datetime.now(timezone.utc) - timedelta(minutes=1)) if next_action_at_past else None,
    )
    session.add(contact)
    session.commit()
    contact_id, campaign_id = contact.id, campaign.id
    session.close()
    return contact_id, campaign_id


# ---------------------------------------------------------------------------
# Work claiming: two workers, one contact, only one may win
# ---------------------------------------------------------------------------

def test_two_workers_claiming_same_contact_only_one_wins(pg_engine):
    from mailer_agent.followup.work_claiming import claim_due_contacts

    contact_id, _ = _seed_campaign_and_contact(pg_engine)

    results = {}
    barrier = threading.Barrier(2)

    def worker(worker_id):
        session = _new_session(pg_engine)
        try:
            barrier.wait(timeout=5)  # maximize overlap
            claimed = claim_due_contacts(session, worker_id=worker_id, limit=10)
            session.commit()
            results[worker_id] = [c.id for c in claimed]
        finally:
            session.close()

    t1 = threading.Thread(target=worker, args=("worker-a",))
    t2 = threading.Thread(target=worker, args=("worker-b",))
    t1.start(); t2.start()
    t1.join(timeout=10); t2.join(timeout=10)

    a_got_it = contact_id in results.get("worker-a", [])
    b_got_it = contact_id in results.get("worker-b", [])
    assert a_got_it != b_got_it, (
        f"Exactly one worker must claim the contact, not both or neither. "
        f"worker-a: {results.get('worker-a')}, worker-b: {results.get('worker-b')}"
    )


def test_work_claim_lease_recovery_after_expiry(pg_engine):
    """A claim whose lease has expired (worker crashed mid-processing)
    must become claimable again -- work is not permanently lost."""
    from datetime import datetime, timedelta, timezone

    from mailer_agent.followup.work_claiming import claim_due_contacts
    from mailer_agent.models import Contact

    contact_id, _ = _seed_campaign_and_contact(pg_engine)

    session = _new_session(pg_engine)
    contact = session.query(Contact).get(contact_id)
    # Simulate a crashed worker: claimed long enough ago that the lease
    # has definitely expired (CLAIM_LEASE_SECONDS is well under an hour).
    contact.claimed_by = "crashed-worker"
    contact.claimed_at = datetime.now(timezone.utc) - timedelta(hours=2)
    session.commit()
    session.close()

    session2 = _new_session(pg_engine)
    claimed = claim_due_contacts(session2, worker_id="recovery-worker", limit=10)
    session2.commit()
    session2.close()

    assert contact_id in [c.id for c in claimed], (
        "A contact with an expired claim lease must be claimable again -- "
        "work must not be permanently lost when a worker crashes mid-processing."
    )


# ---------------------------------------------------------------------------
# Duplicate Message-ID under real concurrency
# ---------------------------------------------------------------------------

def test_concurrent_duplicate_message_id_insert_only_one_succeeds(pg_engine):
    """
    Regression test for the check-then-insert race this session found
    and closed with a real unique constraint (see models.py,
    migrations/003_message_id_unique_constraint.py). Two threads both
    attempt to insert a Message with the identical message_id_header at
    nearly the same instant -- exactly one must succeed; the other must
    fail with IntegrityError (which mail/reply_handler_v2.py's
    process_inbound_email_v2 catches and treats as a graceful duplicate,
    not a crash -- see that function for the production-path version of
    this same handling).
    """
    from mailer_agent.models import Contact, Message, MessageDirection, MessageStatus

    contact_id, _ = _seed_campaign_and_contact(pg_engine)
    shared_message_id = "<concurrency-test-race@example.com>"

    results = {}
    barrier = threading.Barrier(2)

    def worker(name):
        session = _new_session(pg_engine)
        try:
            msg = Message(
                contact_id=contact_id,
                direction=MessageDirection.INBOUND.value,
                subject="test", body="test",
                status=MessageStatus.RECEIVED.value,
                message_id_header=shared_message_id,
            )
            session.add(msg)
            barrier.wait(timeout=5)
            try:
                session.commit()
                results[name] = "success"
            except IntegrityError:
                session.rollback()
                results[name] = "integrity_error"
        finally:
            session.close()

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start(); t2.start()
    t1.join(timeout=10); t2.join(timeout=10)

    outcomes = sorted(results.values())
    assert outcomes == ["integrity_error", "success"], (
        f"Exactly one insert must succeed and the other must hit the unique "
        f"constraint -- got {results}. If both succeeded, the constraint "
        f"from migrations/003_message_id_unique_constraint.py is not applied "
        f"or not working; if both failed, something else is wrong."
    )

    count = (
        _new_session(pg_engine)
        .query(Message)
        .filter(Message.message_id_header == shared_message_id)
        .count()
    )
    assert count == 1, f"Exactly one message row should exist for this Message-ID, found {count}"


# ---------------------------------------------------------------------------
# Concurrent approval: two simultaneous approval requests for one draft
# ---------------------------------------------------------------------------

def test_concurrent_approval_only_sends_once(pg_engine):
    """
    Two SIMULTANEOUS approval requests for the same draft message (not a
    sequential retry -- tests/test_approval_safety_gate.py's
    test_already_sent_message_cannot_be_approved_again already covers
    the sequential case). Exercises the real production function,
    api.messages.approve_and_send_draft, directly -- not a standalone
    lock pattern -- so this proves the actual code path, not just that
    SELECT ... FOR UPDATE works in the abstract.

    This is a regression test for a gap this session found and fixed:
    _get_message_or_404 (the helper approve_and_send_draft uses to fetch
    the message) previously had no row lock, so two concurrent requests
    could both pass the DRAFT-status check before either committed, and
    both attempt to send. Fixed with .with_for_update() -- see
    api/messages.py.
    """
    from mailer_agent.api.messages import approve_and_send_draft
    from mailer_agent.models import Message, MessageDirection, MessageStatus, MessageType

    contact_id, campaign_id = _seed_campaign_and_contact(pg_engine)

    session = _new_session(pg_engine)
    msg = Message(
        contact_id=contact_id, direction=MessageDirection.OUTBOUND.value,
        message_type=MessageType.INITIAL.value, subject="test",
        body="Hello, following up on our conversation.",
        status=MessageStatus.DRAFT.value,
    )
    session.add(msg)
    session.commit()
    message_id = msg.id
    session.close()

    results = {}
    barrier = threading.Barrier(2)

    def approve(name):
        session = _new_session(pg_engine)
        try:
            barrier.wait(timeout=5)
            try:
                approve_and_send_draft(message_id=message_id, org_id="org-concurrency", db=session)
                results[name] = "sent"
            except Exception as e:
                results[name] = f"rejected: {type(e).__name__}"
        finally:
            session.close()

    t1 = threading.Thread(target=approve, args=("a",))
    t2 = threading.Thread(target=approve, args=("b",))
    t1.start(); t2.start()
    t1.join(timeout=10); t2.join(timeout=10)

    sent_count = sum(1 for v in results.values() if v == "sent")
    assert sent_count == 1, (
        f"Exactly one concurrent approval request should succeed; the other "
        f"must be rejected (blocked by the row lock until the first commits, "
        f"then correctly sees status != DRAFT). Got: {results}"
    )

    final_status = (
        _new_session(pg_engine).query(Message).filter(Message.id == message_id).first().status
    )
    assert final_status == MessageStatus.SENT.value
