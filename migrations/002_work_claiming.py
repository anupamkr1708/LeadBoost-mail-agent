"""
Migration 002: Safe work claiming fields (Phase 8).

Adds ``claimed_by`` and ``claimed_at`` to the ``contacts`` table so
the distributed scheduler can use database-level row locking instead of
in-memory locks.

Schema changes
--------------
contacts:
  + claimed_by  VARCHAR   -- NULL when not claimed
  + claimed_at  TIMESTAMP -- NULL when not claimed

Indexes
-------
  idx_contacts_claimed_by   (claimed_by)
    -- lets the expired-lease sweep (recover_expired_claims) run an
       efficient scan instead of a full table scan

  idx_contacts_work_queue   (status, next_action_at, claimed_at)
    -- covers the hot WHERE clause used by claim_due_contacts:
       WHERE status = ? AND next_action_at <= ? AND (claimed_by IS NULL OR claimed_at < ?)

Run with::

    python migrations/002_work_claiming.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from mailer_agent.db import session_scope

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (same pattern as 001_production_hardening.py)
# ---------------------------------------------------------------------------

def _add_column_if_missing(db, table: str, column: str, definition: str) -> bool:
    """Add ``column`` to ``table`` if it does not already exist."""
    # PostgreSQL: check information_schema
    try:
        result = db.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                f"WHERE table_name='{table}' AND column_name='{column}'"
            )
        )
        if result.fetchone():
            logger.info("Column %s.%s already exists -- skipping", table, column)
            return False
        db.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))
        logger.info("✓ Added %s.%s", table, column)
        return True
    except Exception:
        # Fallback for SQLite (no information_schema)
        try:
            db.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))
            logger.info("✓ Added %s.%s", table, column)
            return True
        except Exception as e2:
            if "duplicate column" in str(e2).lower() or "already exists" in str(e2).lower():
                logger.info("Column %s.%s already exists -- skipping", table, column)
                return False
            raise


def _create_index_if_missing(db, index_name: str, sql: str) -> bool:
    try:
        db.execute(text(sql))
        logger.info("✓ Created index %s", index_name)
        return True
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info("Index %s already exists -- skipping", index_name)
            return False
        raise


# ---------------------------------------------------------------------------
# Migration steps
# ---------------------------------------------------------------------------

def add_claiming_columns(db) -> None:
    logger.info("\n=== Adding work-claiming columns to contacts ===")
    _add_column_if_missing(db, "contacts", "claimed_by", "VARCHAR")
    _add_column_if_missing(db, "contacts", "claimed_at", "TIMESTAMP")


def add_claiming_indexes(db) -> None:
    logger.info("\n=== Adding work-claiming indexes ===")

    # Index for expired-lease sweep
    _create_index_if_missing(
        db,
        "idx_contacts_claimed_by",
        "CREATE INDEX IF NOT EXISTS idx_contacts_claimed_by ON contacts(claimed_by)",
    )

    # Composite index covering the hot work-queue WHERE clause
    # PostgreSQL supports this cleanly; SQLite's query planner also benefits.
    _create_index_if_missing(
        db,
        "idx_contacts_work_queue",
        "CREATE INDEX IF NOT EXISTS idx_contacts_work_queue "
        "ON contacts(status, next_action_at, claimed_at)",
    )


def verify(db) -> bool:
    logger.info("\n=== Verifying migration 002 ===")
    checks = []
    try:
        db.execute(text("SELECT claimed_by, claimed_at FROM contacts LIMIT 1"))
        checks.append("✓ contacts.claimed_by and contacts.claimed_at exist")
    except Exception as e:
        checks.append(f"✗ contacts claiming columns missing: {e}")

    for check in checks:
        logger.info(check)
    return all("✓" in c for c in checks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> bool:
    logger.info("Starting migration 002: safe work claiming fields")
    with session_scope() as db:
        try:
            add_claiming_columns(db)
            add_claiming_indexes(db)
            db.commit()
            logger.info("\n✓ Migration 002 committed")

            if verify(db):
                logger.info("\n✅ Migration 002 completed and verified!")
                return True
            else:
                logger.warning("\n⚠️ Migration 002 committed but verification failed")
                return False
        except Exception:
            logger.exception("Migration 002 failed")
            db.rollback()
            raise


if __name__ == "__main__":
    try:
        ok = run()
        sys.exit(0 if ok else 1)
    except Exception:
        logger.exception("Unhandled error during migration 002")
        sys.exit(1)
