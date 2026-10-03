"""
Distributed work claiming for the follow-up scheduler.

Problem this solves
-------------------
When the Mailer Agent runs as a Render Web Service + Background Worker
(or any multi-process/multi-instance setup), multiple processes could
simultaneously query for "due" contacts and process the same row twice
-- resulting in duplicate emails.

Solution
--------
Before any outbound send, the worker atomically claims the contact row
using a database-level lock.  No in-memory locks are used anywhere;
all safety guarantees come from the database.

For PostgreSQL (production):
  Uses ``SELECT … FOR UPDATE SKIP LOCKED`` inside an UPDATE statement.
  SKIP LOCKED means competing workers silently skip rows that are
  already locked by another transaction rather than blocking -- so a
  slow worker doesn't cause a pile-up waiting for its lock to release.

For SQLite (local dev / tests):
  Falls back to a single-row UPDATE + rowcount check (optimistic
  locking via `updated_at` compare).  SQLite serializes all writes so
  there is no actual concurrency risk in single-process dev use.

Lease expiry / recovery
-----------------------
A "lease" is the combination of (claimed_by, claimed_at) on a Contact.
If the worker that claimed a contact crashes or hangs, no other worker
will ever touch it again — unless the lease expires.

``CLAIM_LEASE_SECONDS`` (default: 300 s / 5 min) is the grace window.
Any worker can re-claim a contact whose ``claimed_at`` is older than
that.  The reclaim is also atomic (same FOR UPDATE SKIP LOCKED), so two
workers racing to recover an expired lease can't both win.

After successfully processing a contact (send or skip), the worker
MUST call ``release_claim(db, contact)`` to clear the lease.  If the
worker crashes before release, the lease expires naturally.

Usage
-----
In an engine / scheduler job::

    from mailer_agent.followup.work_claiming import claim_due_contacts, release_claim

    contacts = claim_due_contacts(db, worker_id="worker-abc", status="active", limit=20)
    for contact in contacts:
        try:
            process(contact)
        finally:
            release_claim(db, contact)

"""

from __future__ import annotations

import logging
import os
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import and_, func, or_, select, text, update
from sqlalchemy.orm import Session

from mailer_agent.config import get_settings
from mailer_agent.models import (
    Contact,
    ContactStatus,
    ExternalDispatch,
    ExternalDispatchState,
    Message,
    MessageStatus,
)

logger = logging.getLogger("mailer_agent.followup.work_claiming")
settings = get_settings()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# How long a claim is considered valid.  After this many seconds a worker
# is assumed dead/stalled and any other worker may re-claim the contact.
CLAIM_LEASE_SECONDS: int = int(os.environ.get("CLAIM_LEASE_SECONDS", "300"))

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_postgres(db: Session) -> bool:
    """Return True if the underlying engine is PostgreSQL."""
    dialect = db.get_bind().dialect.name  # type: ignore[union-attr]
    return dialect == "postgresql"


def _utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def _lease_cutoff() -> datetime:
    """Claims older than this timestamp are considered expired."""
    return _utcnow() - timedelta(seconds=CLAIM_LEASE_SECONDS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def claim_due_contacts(
    db: Session,
    worker_id: str,
    status: str = ContactStatus.ACTIVE.value,
    limit: Optional[int] = None,
) -> List[Contact]:
    """
    Atomically claim up to *limit* due contacts for *worker_id*.

    "Due" means:
      - ``status`` matches the requested status value
      - ``next_action_at`` is set and <= now
      - Either not claimed (``claimed_by IS NULL``) **or** the current
        claim has expired (``claimed_at < now() - CLAIM_LEASE_SECONDS``)
      - the contact's Campaign is NOT integration-managed
        (``integration_source IS NULL``): LeadBoost-owned contacts are
        authorized per action by LeadBoost and delivered via
        ExternalDispatch, so the native initial/follow-up schedulers -- the
        only callers of this function -- must never claim them, whatever
        their status or next_action_at.

    Returns a list of Contact ORM objects whose ``claimed_by`` has been
    updated to *worker_id* in the same transaction.  The caller is
    responsible for calling ``release_claim()`` after processing each
    contact.
    """
    effective_limit = limit or settings.max_sends_per_cycle
    now = _utcnow()
    lease_cutoff = _lease_cutoff()
    # Store as naive UTC for SQLite / legacy columns that expect naive datetimes
    now_naive = now.replace(tzinfo=None)
    cutoff_naive = lease_cutoff.replace(tzinfo=None)

    if _is_postgres(db):
        contact_ids = _claim_postgres(db, worker_id, status, effective_limit, now_naive, cutoff_naive)
    else:
        contact_ids = _claim_sqlite(db, worker_id, status, effective_limit, now_naive, cutoff_naive)

    if not contact_ids:
        return []

    contacts = db.query(Contact).filter(Contact.id.in_(contact_ids)).all()
    logger.info(
        "Worker %s claimed %d contact(s) with status=%s",
        worker_id, len(contacts), status,
    )
    return contacts


def release_claim(db: Session, contact: Contact) -> None:
    """
    Release the work claim on a contact after processing.

    Sets ``claimed_by = NULL`` and ``claimed_at = NULL``.  Should be
    called in a ``finally`` block so a failed send still releases the
    lease and allows retry by the next poll cycle.
    """
    contact.claimed_by = None
    contact.claimed_at = None
    db.add(contact)
    # Flush immediately so the release is visible to other workers even
    # before the outer transaction commits.
    db.flush()
    logger.debug("Released claim on contact %d", contact.id)


def release_all_claims(db: Session, worker_id: str) -> int:
    """
    Release all claims held by *worker_id*.

    Called on graceful shutdown so the next worker can pick up any
    contacts that were claimed but not yet processed.  Returns the count
    of rows updated.
    """
    count = (
        db.query(Contact)
        .filter(Contact.claimed_by == worker_id)
        .update({"claimed_by": None, "claimed_at": None}, synchronize_session="fetch")
    )
    db.flush()
    if count:
        logger.info("Released %d stale claim(s) held by worker %s on shutdown", count, worker_id)
    return count


def recover_expired_claims(db: Session, dry_run: bool = False) -> int:
    """
    Find and release all expired leases regardless of which worker holds them.

    Useful as a startup sweep to clear any leases left by a previous
    crashed process.  Returns the count of recovered rows.

    Pass ``dry_run=True`` to count without releasing (for monitoring/logging).
    """
    cutoff_naive = _lease_cutoff().replace(tzinfo=None)
    stale = (
        db.query(Contact)
        .filter(
            Contact.claimed_by.isnot(None),
            Contact.claimed_at < cutoff_naive,
        )
        .all()
    )

    if not stale:
        return 0

    count = len(stale)
    logger.warning(
        "Found %d expired claim(s) (lease cutoff: %s)",
        count,
        cutoff_naive.isoformat(),
    )

    if not dry_run:
        for c in stale:
            logger.warning(
                "Recovering expired claim: contact_id=%d, held_by=%s since %s",
                c.id,
                c.claimed_by,
                c.claimed_at,
            )
            c.claimed_by = None
            c.claimed_at = None
            db.add(c)
        db.flush()
        logger.info("Recovered %d expired claim(s)", count)

    return count


def get_active_claims(db: Session) -> List[Contact]:
    """
    Return all contacts currently held by a (non-expired) claim.

    Useful for monitoring/admin tooling.
    """
    cutoff_naive = _lease_cutoff().replace(tzinfo=None)
    return (
        db.query(Contact)
        .filter(
            Contact.claimed_by.isnot(None),
            Contact.claimed_at >= cutoff_naive,
        )
        .all()
    )


# ---------------------------------------------------------------------------
# PostgreSQL implementation — SELECT FOR UPDATE SKIP LOCKED
# ---------------------------------------------------------------------------

def _claim_postgres(
    db: Session,
    worker_id: str,
    status: str,
    limit: int,
    now_naive: datetime,
    cutoff_naive: datetime,
) -> List[int]:
    """
    PostgreSQL-native atomic claim using a single UPDATE … FROM (SELECT … FOR UPDATE SKIP LOCKED).

    This pattern is the gold-standard for PostgreSQL job queues:
      1. The inner SELECT acquires row-level locks with SKIP LOCKED so
         competing workers never block each other — they each get
         different rows.
      2. The outer UPDATE atomically writes claimed_by + claimed_at only
         on rows the inner SELECT locked.
      3. RETURNING id lets us know which rows we actually claimed.

    The WHERE clause in the inner SELECT covers both:
      - Unclaimed rows  (claimed_by IS NULL)
      - Expired leases  (claimed_at < cutoff)

    Note: next_action_at is stored as a naive UTC datetime in the DB.
    We compare it against ``now_naive`` (also stripped of tzinfo) to
    avoid any implicit cast errors.
    """
    result = db.execute(
        text("""
            UPDATE contacts
               SET claimed_by  = :worker_id,
                   claimed_at  = :now,
                   updated_at  = :now
             WHERE id IN (
                     SELECT c.id
                       FROM contacts c
                      WHERE c.status         = :status
                        AND c.next_action_at IS NOT NULL
                        AND c.next_action_at <= :now
                        AND (
                              c.claimed_by IS NULL
                              OR c.claimed_at < :cutoff
                            )
                        AND NOT EXISTS (
                              SELECT 1 FROM campaigns cm
                               WHERE cm.id = c.campaign_id
                                 AND cm.integration_source IS NOT NULL
                            )
                      ORDER BY c.next_action_at
                      LIMIT :limit
                         FOR UPDATE SKIP LOCKED
                   )
            RETURNING id
        """),
        {
            "worker_id": worker_id,
            "now":       now_naive,
            "cutoff":    cutoff_naive,
            "status":    status,
            "limit":     limit,
        },
    )
    return [row[0] for row in result.fetchall()]


# ---------------------------------------------------------------------------
# SQLite fallback — optimistic UPDATE with rowcount check
# ---------------------------------------------------------------------------

def _claim_sqlite(
    db: Session,
    worker_id: str,
    status: str,
    limit: int,
    now_naive: datetime,
    cutoff_naive: datetime,
) -> List[int]:
    """
    SQLite-compatible work claiming via optimistic locking.

    SQLite serialises all writes at the file level, so there is no true
    concurrent write risk in single-process local dev.  We still use the
    same two-phase (SELECT candidates → UPDATE one-by-one) approach so
    the logic is exercisable in tests.

    Each candidate is claimed with an UPDATE … WHERE claimed_by IS NULL
    (or claimed_at < cutoff) check in the WHERE clause.  If two threads
    somehow race (shouldn't happen with SQLite's write serialisation),
    one UPDATE will affect 0 rows and we simply skip that candidate.
    """
    # Step 1: Find candidate IDs (read-only SELECT, no lock needed on SQLite)
    rows = db.execute(
        text("""
            SELECT id
              FROM contacts
             WHERE status         = :status
               AND next_action_at IS NOT NULL
               AND next_action_at <= :now
               AND (
                     claimed_by IS NULL
                     OR claimed_at < :cutoff
                   )
               AND NOT EXISTS (
                     SELECT 1 FROM campaigns cm
                      WHERE cm.id = contacts.campaign_id
                        AND cm.integration_source IS NOT NULL
                   )
             ORDER BY next_action_at
             LIMIT :limit
        """),
        {"status": status, "now": now_naive, "cutoff": cutoff_naive, "limit": limit},
    ).fetchall()
    candidates = [row[0] for row in rows]

    if not candidates:
        return []

    # Step 2: Attempt to claim each row; skip if another writer beat us.
    claimed_ids: List[int] = []
    for cid in candidates:
        result = db.execute(
            text("""
                UPDATE contacts
                   SET claimed_by = :worker_id,
                       claimed_at = :now,
                       updated_at = :now
                 WHERE id = :id
                   AND (claimed_by IS NULL OR claimed_at < :cutoff)
            """),
            {
                "worker_id": worker_id,
                "now":       now_naive,
                "cutoff":    cutoff_naive,
                "id":        cid,
            },
        )
        if result.rowcount == 1:
            claimed_ids.append(cid)

    return claimed_ids


# ---------------------------------------------------------------------------
# ExternalDispatch claiming (Phase C6) -- LeadBoost async dispatch
# ---------------------------------------------------------------------------
#
# Same philosophy as the Contact claim above -- the database is the queue and
# the only arbiter -- with three deliberate differences:
#
#  1. ONLY state=QUEUED is claimable. A SENDING row is never re-claimed, not
#     even after its lease expires: an expired SENDING lease resolves to
#     UNKNOWN (recover_expired_external_dispatches), never back to QUEUED and
#     never to another SMTP attempt. See models.resolve_expired_sending_lease
#     for why the two indistinguishable crash shapes force that.
#  2. ONE row per claim. The caller claims, processes, then claims again, so a
#     claim never idles behind other sends and ages toward its lease (the
#     Contact path claims a batch and sleeps between sends).
#  3. The claim does NOT commit. It runs inside the caller's "Transaction B"
#     together with the pre-SMTP gates and the Message-ID persistence, and the
#     caller commits once. A crash anywhere before that commit rolls the claim
#     back and the row is simply QUEUED again -- no SMTP has happened.
#
# The lease used for recovery is settings.external_dispatch_lease_seconds
# (default 900), NOT CLAIM_LEASE_SECONDS (300, contacts only).

@dataclass(frozen=True)
class ClaimedDispatch:
    """
    Identity of one successful claim. ``claimed_at`` is the exact naive-UTC
    value written to the row and doubles as a FENCING TOKEN: the outcome
    write (Transaction C) matches on (id, organization_id, state=SENDING,
    claimed_by, claimed_at), so a worker that lost its lease -- or whose row
    was moved on by recovery/shutdown -- cannot overwrite the newer state.
    """
    dispatch_id: int
    organization_id: str
    worker_id: str
    claimed_at: datetime


@dataclass(frozen=True)
class RecoveredDispatch:
    dispatch_id: int
    organization_id: str
    external_action_id: Optional[str]
    correlation_id: Optional[str]
    public_reference: str
    previous_worker: Optional[str]
    previous_claimed_at: Optional[datetime]


def _naive_utc(dt: Optional[datetime] = None) -> datetime:
    dt = dt or _utcnow()
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def claim_next_external_dispatch(
    db: Session,
    worker_id: str,
    now: Optional[datetime] = None,
) -> Optional[ClaimedDispatch]:
    """
    Atomically claim the oldest QUEUED ExternalDispatch for *worker_id*.

    Sets state=SENDING, claimed_by, claimed_at in the caller's open
    transaction and returns the claim, or None if nothing is claimable.
    Does not commit and performs no I/O beyond the database.

    PostgreSQL: ``SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1`` then a guarded
    UPDATE of that locked row. (Two statements on purpose: the
    UPDATE ... WHERE id IN (SELECT ... LIMIT n FOR UPDATE SKIP LOCKED) form
    used for contacts can, under some plans, evaluate the subquery more than
    once and claim more than n rows. With LIMIT 1 and an explicit id there is
    nothing for the planner to get wrong.) The row lock is held until the
    caller's commit/rollback; competing workers skip it rather than block.

    SQLite (dev/tests): candidate SELECT, then ``UPDATE ... WHERE id=? AND
    state='queued'`` with a rowcount check. This exercises the guarded-update
    logic only; it does NOT prove PostgreSQL locking -- that is covered
    separately against a real PostgreSQL (tests/test_external_dispatch_postgres.py).
    """
    claimed_at = _naive_utc(now)
    queued = ExternalDispatchState.QUEUED.value
    sending = ExternalDispatchState.SENDING.value

    def _guarded_claim(dispatch_id: int) -> bool:
        result = db.execute(
            update(ExternalDispatch)
            .where(
                ExternalDispatch.id == dispatch_id,
                ExternalDispatch.state == queued,
                ExternalDispatch.message_id.is_not(None),   # M2-B: never send a row still awaiting generation
            )
            .values(
                state=sending,
                claimed_by=worker_id,
                claimed_at=claimed_at,
                updated_at=_utcnow(),
            )
            .execution_options(synchronize_session=False)
        )
        return result.rowcount == 1

    if _is_postgres(db):
        row = db.execute(
            select(ExternalDispatch.id, ExternalDispatch.organization_id)
            .where(ExternalDispatch.state == queued, ExternalDispatch.message_id.is_not(None))
            .order_by(ExternalDispatch.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        ).first()
        if row is None or not _guarded_claim(row.id):
            return None
        chosen = row
    else:
        candidates = db.execute(
            select(ExternalDispatch.id, ExternalDispatch.organization_id)
            .where(ExternalDispatch.state == queued, ExternalDispatch.message_id.is_not(None))
            .order_by(ExternalDispatch.id)
            .limit(10)
        ).all()
        chosen = next((c for c in candidates if _guarded_claim(c.id)), None)
        if chosen is None:
            return None

    logger.info(
        "Worker %s claimed external_dispatch id=%d org=%s (QUEUED -> SENDING)",
        worker_id, chosen.id, chosen.organization_id,
    )
    return ClaimedDispatch(
        dispatch_id=chosen.id,
        organization_id=chosen.organization_id,
        worker_id=worker_id,
        claimed_at=claimed_at,
    )


def recover_expired_external_dispatches(
    db: Session,
    *,
    now: Optional[datetime] = None,
    lease_seconds: Optional[int] = None,
    limit: int = 100,
) -> List[RecoveredDispatch]:
    """
    Resolve expired SENDING leases: ``SENDING -> UNKNOWN``. Never QUEUED.

    A row is expired when state=SENDING and claimed_at is older than the
    lease (or claimed_at is NULL, an anomalous shape that can never be a live
    claim). On expiry: state=UNKNOWN, claimed_by/claimed_at cleared, a clear
    reason on the dispatch and on its Message (SENDING/DRAFT -> UNKNOWN).

    Deliberately performs NO SMTP and imports no sender: it cannot resend.
    Safe to run repeatedly and concurrently: every UPDATE re-checks the
    expiry predicate, a second run finds nothing, and on PostgreSQL rows are
    taken with FOR UPDATE SKIP LOCKED so two sweeps split the work. Does not
    commit; the caller owns the transaction.
    """
    lease = (
        settings.external_dispatch_lease_seconds
        if lease_seconds is None else lease_seconds
    )
    now_naive = _naive_utc(now)
    cutoff = now_naive - timedelta(seconds=lease)
    sending = ExternalDispatchState.SENDING.value
    expired = and_(
        ExternalDispatch.state == sending,
        or_(ExternalDispatch.claimed_at.is_(None), ExternalDispatch.claimed_at < cutoff),
    )

    query = (
        select(
            ExternalDispatch.id,
            ExternalDispatch.organization_id,
            ExternalDispatch.external_action_id,
            ExternalDispatch.correlation_id,
            ExternalDispatch.public_reference,
            ExternalDispatch.claimed_by,
            ExternalDispatch.claimed_at,
            ExternalDispatch.message_id,
        )
        .where(expired)
        .order_by(ExternalDispatch.id)
        .limit(limit)
    )
    if _is_postgres(db):
        query = query.with_for_update(skip_locked=True)

    recovered: List[RecoveredDispatch] = []
    for row in db.execute(query).all():
        reason = (
            "lease_expired: dispatch was SENDING and its claim "
            f"(worker={row.claimed_by}, claimed_at={row.claimed_at}) expired "
            f"after {lease}s. Whether SMTP accepted the message is unknown; "
            "it will NOT be retried automatically."
        )
        result = db.execute(
            update(ExternalDispatch)
            .where(ExternalDispatch.id == row.id, expired)
            .values(
                state=ExternalDispatchState.UNKNOWN.value,
                claimed_by=None,
                claimed_at=None,
                error_message=reason,
                updated_at=_utcnow(),
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            continue  # another sweep (or a late outcome) got there first
        db.execute(
            update(Message)
            .where(
                Message.id == row.message_id,
                Message.status.in_([MessageStatus.SENDING.value, MessageStatus.DRAFT.value]),
            )
            .values(status=MessageStatus.UNKNOWN.value, error_message=reason)
            .execution_options(synchronize_session=False)
        )
        logger.warning(
            "external_dispatch lease recovery: SENDING -> UNKNOWN id=%d org=%s "
            "external_action_id=%s correlation_id=%s public_reference=%s "
            "previous_worker=%s previous_claimed_at=%s lease=%ds (no resend)",
            row.id, row.organization_id, row.external_action_id,
            row.correlation_id, row.public_reference, row.claimed_by,
            row.claimed_at, lease,
        )
        recovered.append(
            RecoveredDispatch(
                dispatch_id=row.id,
                organization_id=row.organization_id,
                external_action_id=row.external_action_id,
                correlation_id=row.correlation_id,
                public_reference=row.public_reference,
                previous_worker=row.claimed_by,
                previous_claimed_at=row.claimed_at,
            )
        )
    return recovered


def finish_external_dispatch(
    db: Session,
    claim: ClaimedDispatch,
    *,
    new_state: ExternalDispatchState,
    error_message: Optional[str] = None,
) -> bool:
    """
    OWNERSHIP-FENCED outcome write ("Transaction C", also used for
    pre-SMTP gate failures and for shutdown's SENDING -> UNKNOWN).

    Moves SENDING -> ``new_state`` and clears claimed_by/claimed_at only if
    the row is STILL the one this worker claimed: id, organization_id,
    state=SENDING, claimed_by and the exact claimed_at token must all match.
    Returns False (and writes nothing) otherwise -- e.g. lease recovery or a
    shutdown sweep already resolved it to UNKNOWN. A worker that lost its
    claim therefore can never overwrite a newer state, in either direction.

    On success the linked Message is moved to the matching terminal status
    (sent/failed/unknown). Never targets QUEUED. Does not commit.
    """
    if new_state not in (
        ExternalDispatchState.SENT,
        ExternalDispatchState.FAILED,
        ExternalDispatchState.UNKNOWN,
    ):
        raise ValueError(f"finish_external_dispatch cannot move to {new_state!r}")

    result = db.execute(
        update(ExternalDispatch)
        .where(
            ExternalDispatch.id == claim.dispatch_id,
            ExternalDispatch.organization_id == claim.organization_id,
            ExternalDispatch.state == ExternalDispatchState.SENDING.value,
            ExternalDispatch.claimed_by == claim.worker_id,
            ExternalDispatch.claimed_at == claim.claimed_at,
        )
        .values(
            state=new_state.value,
            claimed_by=None,
            claimed_at=None,
            error_message=error_message,
            updated_at=_utcnow(),
        )
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        return False
    message_pk = db.execute(
        select(ExternalDispatch.message_id).where(ExternalDispatch.id == claim.dispatch_id)
    ).scalar_one()
    db.execute(
        update(Message)
        .where(Message.id == message_pk)
        .values(status=MessageStatus(new_state.value).value, error_message=error_message)
        .execution_options(synchronize_session=False)
    )
    return True


# ---------------------------------------------------------------------------
# M2-B: generation claiming (QUEUED, message_id NULL -> GENERATING -> QUEUED/FAILED)
# ---------------------------------------------------------------------------

def claim_next_generation_dispatch(
    db: Session,
    worker_id: str,
    now: Optional[datetime] = None,
) -> Optional[ClaimedDispatch]:
    """
    Atomically claim the oldest accepted-but-ungenerated dispatch
    (state=QUEUED AND message_id IS NULL) for *worker_id*: state=GENERATING,
    claimed_by, claimed_at (the fencing token), in the caller's open
    transaction. Same two-statement SKIP LOCKED shape as
    claim_next_external_dispatch; does not commit, no I/O beyond the database.
    """
    claimed_at = _naive_utc(now)
    queued = ExternalDispatchState.QUEUED.value
    generating = ExternalDispatchState.GENERATING.value

    def _guarded_claim(dispatch_id: int) -> bool:
        result = db.execute(
            update(ExternalDispatch)
            .where(
                ExternalDispatch.id == dispatch_id,
                ExternalDispatch.state == queued,
                ExternalDispatch.message_id.is_(None),
            )
            .values(
                state=generating,
                claimed_by=worker_id,
                claimed_at=claimed_at,
                updated_at=_utcnow(),
            )
            .execution_options(synchronize_session=False)
        )
        return result.rowcount == 1

    base = (
        select(ExternalDispatch.id, ExternalDispatch.organization_id)
        .where(ExternalDispatch.state == queued, ExternalDispatch.message_id.is_(None))
        .order_by(ExternalDispatch.id)
    )
    if _is_postgres(db):
        row = db.execute(base.limit(1).with_for_update(skip_locked=True)).first()
        if row is None or not _guarded_claim(row.id):
            return None
        chosen = row
    else:
        chosen = next((c for c in db.execute(base.limit(10)).all() if _guarded_claim(c.id)), None)
        if chosen is None:
            return None

    logger.info(
        "Worker %s claimed external_dispatch id=%d org=%s for generation (QUEUED -> GENERATING)",
        worker_id, chosen.id, chosen.organization_id,
    )
    return ClaimedDispatch(
        dispatch_id=chosen.id,
        organization_id=chosen.organization_id,
        worker_id=worker_id,
        claimed_at=claimed_at,
    )


def _generation_fence(claim: ClaimedDispatch):
    return and_(
        ExternalDispatch.id == claim.dispatch_id,
        ExternalDispatch.organization_id == claim.organization_id,
        ExternalDispatch.state == ExternalDispatchState.GENERATING.value,
        ExternalDispatch.claimed_by == claim.worker_id,
        ExternalDispatch.claimed_at == claim.claimed_at,
        ExternalDispatch.message_id.is_(None),
    )


def complete_generation(db: Session, claim: ClaimedDispatch, message_id: int) -> bool:
    """
    OWNERSHIP-FENCED GENERATING -> QUEUED: attach the freshly flushed Message
    and clear the claim, only if the row is STILL the one this worker claimed
    (id, org, state, claimed_by, exact claimed_at, message_id NULL). False
    writes nothing -- the caller must then roll back its Message insert, so a
    worker that lost its lease can never leave a second Message behind.
    Does not commit.
    """
    result = db.execute(
        update(ExternalDispatch)
        .where(_generation_fence(claim))
        .values(
            state=ExternalDispatchState.QUEUED.value,
            message_id=message_id,
            claimed_by=None,
            claimed_at=None,
            updated_at=_utcnow(),
        )
        .execution_options(synchronize_session=False)
    )
    return result.rowcount == 1


def fail_generation(db: Session, claim: ClaimedDispatch, error_message: str) -> bool:
    """
    OWNERSHIP-FENCED GENERATING -> FAILED. No Message exists (message_id stays
    NULL); nothing was or will be sent. Same fence as complete_generation.
    Does not commit.
    """
    result = db.execute(
        update(ExternalDispatch)
        .where(_generation_fence(claim))
        .values(
            state=ExternalDispatchState.FAILED.value,
            claimed_by=None,
            claimed_at=None,
            error_message=error_message,
            updated_at=_utcnow(),
        )
        .execution_options(synchronize_session=False)
    )
    return result.rowcount == 1


def recover_expired_generations(
    db: Session,
    *,
    now: Optional[datetime] = None,
    lease_seconds: Optional[int] = None,
    limit: int = 100,
) -> List[int]:
    """
    Expired GENERATING lease -> QUEUED (message_id still NULL), claim cleared.

    Safe -- and deliberately the opposite of SENDING recovery -- because
    generation has no external side effect: nothing was transmitted and no
    Message was committed (the Message and the GENERATING -> QUEUED move are
    one fenced commit), so the row is simply regenerated. A late worker's
    complete_generation then matches nothing and its Message is rolled back.
    Safe to run repeatedly/concurrently: every UPDATE re-checks the expiry
    predicate; SKIP LOCKED on PostgreSQL. Does not commit. Returns the ids.
    """
    lease = (
        settings.external_dispatch_generation_lease_seconds
        if lease_seconds is None else lease_seconds
    )
    cutoff = _naive_utc(now) - timedelta(seconds=lease)
    expired = and_(
        ExternalDispatch.state == ExternalDispatchState.GENERATING.value,
        ExternalDispatch.message_id.is_(None),
        or_(ExternalDispatch.claimed_at.is_(None), ExternalDispatch.claimed_at < cutoff),
    )
    query = select(ExternalDispatch.id, ExternalDispatch.organization_id, ExternalDispatch.claimed_by).where(expired).order_by(ExternalDispatch.id).limit(limit)
    if _is_postgres(db):
        query = query.with_for_update(skip_locked=True)

    recovered: List[int] = []
    for row in db.execute(query).all():
        result = db.execute(
            update(ExternalDispatch)
            .where(ExternalDispatch.id == row.id, expired)
            .values(
                state=ExternalDispatchState.QUEUED.value,
                claimed_by=None,
                claimed_at=None,
                updated_at=_utcnow(),
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            continue
        logger.warning(
            "external_dispatch generation lease recovery: GENERATING -> QUEUED id=%d org=%s "
            "previous_worker=%s lease=%ds (nothing sent; will regenerate)",
            row.id, row.organization_id, row.claimed_by, lease,
        )
        recovered.append(row.id)
    return recovered


def annotate_unknown_dispatch(
    db: Session, *, dispatch_id: int, organization_id: str, note: str
) -> bool:
    """
    Append an informational note to an UNKNOWN dispatch's error_message.

    Used when a worker learns a definite outcome AFTER it lost its claim
    (the row was already resolved to UNKNOWN). It never changes state --
    UNKNOWN is not rewritten to SENT/FAILED and never to QUEUED -- it only
    leaves the fact for whoever reconciles. Matches only state=UNKNOWN.
    """
    result = db.execute(
        update(ExternalDispatch)
        .where(
            ExternalDispatch.id == dispatch_id,
            ExternalDispatch.organization_id == organization_id,
            ExternalDispatch.state == ExternalDispatchState.UNKNOWN.value,
        )
        .values(error_message=func.coalesce(ExternalDispatch.error_message, "") + " | " + note)
        .execution_options(synchronize_session=False)
    )
    return result.rowcount == 1


# ---------------------------------------------------------------------------
# Worker identity helper
# ---------------------------------------------------------------------------

# Fixed once per interpreter. Combined with the live PID it makes the default
# worker identity unique per PROCESS, not per host: two worker.py processes on
# one machine (a rolling deploy, a local dev box) previously both became
# "worker-<hostname>", so anything keyed on worker_id alone (release_all_claims,
# an ownership check) could not tell them apart.
_PROCESS_INSTANCE = uuid.uuid4().hex[:8]


def make_worker_id(prefix: str = "worker") -> str:
    """
    Build a process-unique worker identity string for lease attribution.

    An explicit ``WORKER_ID`` env var wins (worker.py exports the id it
    generated so scheduler-job threads in the same process share it).
    Otherwise: ``<prefix>-<hostname>-<pid>-<8 hex>``. Stable for the life of
    the process, distinct across processes -- including two processes on the
    same host. Note an operator who sets the SAME WORKER_ID on several
    processes defeats that; nothing ownership-critical here relies on
    worker_id alone (ExternalDispatch fences on claimed_at as well).
    """
    from_env = os.environ.get("WORKER_ID", "")
    if from_env:
        return from_env
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"
    return f"{prefix}-{hostname}-{os.getpid()}-{_PROCESS_INSTANCE}"
