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
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from mailer_agent.config import get_settings
from mailer_agent.models import Contact, ContactStatus

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
# Worker identity helper
# ---------------------------------------------------------------------------

def make_worker_id(prefix: str = "worker") -> str:
    """
    Build a stable-enough worker identity string for lease attribution.

    Uses ``WORKER_ID`` env var if set (recommended for Render / Docker
    deployments where each container has a unique instance name), otherwise
    falls back to ``<prefix>-<hostname>``.

    For local dev, ``prefix`` defaults to ``"worker"`` giving ``"worker-mymachine"``.
    """
    from_env = os.environ.get("WORKER_ID", "")
    if from_env:
        return from_env
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"
    return f"{prefix}-{hostname}"
