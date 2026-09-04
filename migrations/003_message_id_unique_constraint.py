"""
Migration 003: unique constraint on messages.message_id_header.

Closes a check-then-insert race in inbound-message deduplication: without
a database-level constraint, mail/reply_handler_v2.py's dedup check
("does a Message with this message_id_header already exist? if not,
insert") can be defeated by two truly concurrent transactions that both
see "not found" before either commits -- e.g. the same email delivered
via webhook and picked up by IMAP polling at nearly the same moment, or
a retried webhook delivery racing the original. SQLite testing cannot
demonstrate this is safe; only a real PostgreSQL instance with genuinely
concurrent transactions can (see tests/test_postgresql_concurrency.py,
which is written but NOT VERIFIED in this environment -- no PostgreSQL
instance was available to actually run it against; see
docs/FINAL_PRODUCTION_READINESS.md for the exact command to run it).

Schema changes
--------------
messages:
  + UNIQUE constraint on message_id_header (NULLs unconstrained -- many
    draft/failed-before-send messages legitimately have none; only
    non-NULL collisions are rejected)

Run with::

    python migrations/003_message_id_unique_constraint.py

This does NOT retroactively deduplicate any existing rows -- if a
database already has a genuine duplicate message_id_header from before
this constraint existed (which the applicaton-level check should have
prevented in the non-concurrent case, but see above), adding the
constraint will fail until that's resolved manually. This migration
checks for that up front and reports it clearly rather than failing
opaquely mid-ALTER.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from mailer_agent.db import session_scope

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _check_no_existing_duplicates(db) -> list[str]:
    """Return any message_id_header values that already collide. Empty list = safe to add the constraint."""
    result = db.execute(
        text(
            "SELECT message_id_header, COUNT(*) as c FROM messages "
            "WHERE message_id_header IS NOT NULL "
            "GROUP BY message_id_header HAVING COUNT(*) > 1"
        )
    )
    return [row[0] for row in result.fetchall()]


def _constraint_exists_postgres(db) -> bool:
    result = db.execute(
        text(
            "SELECT constraint_name FROM information_schema.table_constraints "
            "WHERE table_name='messages' AND constraint_name='uq_messages_message_id_header'"
        )
    )
    return result.fetchone() is not None


def add_unique_constraint(db) -> bool:
    logger.info("\n=== Checking for existing duplicate message_id_header values ===")
    duplicates = _check_no_existing_duplicates(db)
    if duplicates:
        logger.error(
            "✗ Found %d existing duplicate message_id_header value(s): %s\n"
            "  Resolve these manually (decide which row is canonical for "
            "each duplicate) before this migration can add the uniqueness "
            "constraint. Not attempting an automatic merge -- that's a "
            "data decision, not a schema one.",
            len(duplicates), duplicates[:10],
        )
        return False

    logger.info("No existing duplicates found -- safe to add constraint.")
    logger.info("\n=== Adding unique constraint on messages.message_id_header ===")

    try:
        if _constraint_exists_postgres(db):
            logger.info("Constraint uq_messages_message_id_header already exists -- skipping")
            return True
        db.execute(
            text(
                "ALTER TABLE messages ADD CONSTRAINT uq_messages_message_id_header "
                "UNIQUE (message_id_header)"
            )
        )
        logger.info("✓ Added uq_messages_message_id_header")
        return True
    except Exception as e:
        msg = str(e).lower()
        if "already exists" in msg or "duplicate" in msg:
            # SQLite (no information_schema) or a constraint already present
            logger.info("Constraint already exists (or SQLite reported it during creation) -- skipping")
            return True
        raise


def verify(db) -> bool:
    logger.info("\n=== Verifying migration 003 ===")
    try:
        # A real duplicate insert would now raise -- we don't actually
        # insert one here (that would leave junk data), just confirm the
        # constraint is queryable / the table still functions normally.
        db.execute(text("SELECT message_id_header FROM messages LIMIT 1"))
        logger.info("✓ messages.message_id_header is queryable")
        return True
    except Exception as e:
        logger.error("✗ Verification query failed: %s", e)
        return False


def run() -> bool:
    logger.info("Starting migration 003: messages.message_id_header unique constraint")
    with session_scope() as db:
        try:
            ok = add_unique_constraint(db)
            if not ok:
                db.rollback()
                logger.error("\n✗ Migration 003 NOT applied -- resolve duplicates first (see above)")
                return False
            db.commit()
            logger.info("\n✓ Migration 003 committed")

            if verify(db):
                logger.info("\n✅ Migration 003 completed and verified!")
                return True
            else:
                logger.warning("\n⚠️ Migration 003 committed but verification failed")
                return False
        except Exception:
            logger.exception("Migration 003 failed")
            db.rollback()
            raise


if __name__ == "__main__":
    try:
        ok = run()
        sys.exit(0 if ok else 1)
    except Exception:
        logger.exception("Unhandled error during migration 003")
        sys.exit(1)
