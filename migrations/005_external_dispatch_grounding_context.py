"""
Migration 005: external_dispatches.grounding_context (C9.2).

Adds the immutable per-dispatch grounding snapshot used by
POST /integrations/leadboost/outreach-requests (see
mailer_agent/api/integrations.py and models.py::ExternalDispatch.grounding_context).

Schema change
-------------
external_dispatches:
  + grounding_context  JSON, NULL
    NULL for every existing row and for every dispatch created by the
    exact-message route (POST /outreach-actions); the worker treats NULL as
    "ground against the integration Campaign/Contact fields exactly as
    before". Purely additive: no existing column, constraint or row is
    touched, and nothing reads the column until the C9.2 code is deployed.

Same conventions as 004: a standalone, idempotent script (re-running is a
safe no-op). Unlike 001-004 this one also ships ``downgrade()`` because the
C9.2 stage asks for upgrade/downgrade/re-upgrade to be verifiable. Downgrade
DROPs the column and therefore discards any snapshots written since the
upgrade -- only run it before the C9.2 endpoint has accepted real traffic, or
accept that those dispatches fall back to legacy grounding (which would
re-validate their messages against the shared integration Campaign/Contact
rows -- the exact lineage the snapshot exists to prevent).

Run with::

    python migrations/005_external_dispatch_grounding_context.py            # upgrade
    python migrations/005_external_dispatch_grounding_context.py --downgrade

Apply BEFORE deploying the C9.2 application code: the ORM selects this column
on every ExternalDispatch query, including the existing worker's claim path,
so code deployed ahead of the column fails on first use (see migrations/README.md).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import inspect, text

from mailer_agent.db import session_scope

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TABLE = "external_dispatches"
COLUMN = "grounding_context"


def _column_exists(db) -> bool:
    # inspect() on the session's connection works on both PostgreSQL and SQLite
    # (no information_schema dependency).
    cols = inspect(db.connection()).get_columns(TABLE)
    return any(c["name"] == COLUMN for c in cols)


def add_grounding_context_column(db) -> bool:
    logger.info("\n=== Adding %s.%s ===", TABLE, COLUMN)
    if _column_exists(db):
        logger.info("Column %s.%s already exists -- skipping", TABLE, COLUMN)
        return False
    # "JSON" is what the ORM's Column(JSON) emits on both engines.
    db.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} JSON NULL"))
    logger.info("✓ Added %s.%s", TABLE, COLUMN)
    return True


def drop_grounding_context_column(db) -> bool:
    logger.info("\n=== Dropping %s.%s ===", TABLE, COLUMN)
    if not _column_exists(db):
        logger.info("Column %s.%s does not exist -- skipping", TABLE, COLUMN)
        return False
    db.execute(text(f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}"))
    logger.info("✓ Dropped %s.%s", TABLE, COLUMN)
    return True


def verify(db) -> bool:
    try:
        db.execute(text(f"SELECT {COLUMN} FROM {TABLE} LIMIT 1"))
        logger.info("✓ %s.%s is queryable", TABLE, COLUMN)
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("✗ %s.%s missing/broken: %s", TABLE, COLUMN, e)
        return False


def run() -> bool:
    logger.info("Starting migration 005: external_dispatches.grounding_context")
    with session_scope() as db:
        try:
            add_grounding_context_column(db)
            db.commit()
            ok = verify(db)
            logger.info("✅ Migration 005 completed" if ok else "⚠️ Migration 005 verification failed")
            return ok
        except Exception:
            logger.exception("Migration 005 failed")
            db.rollback()
            raise


def downgrade() -> bool:
    logger.info("Starting migration 005 DOWNGRADE")
    with session_scope() as db:
        try:
            drop_grounding_context_column(db)
            db.commit()
            return True
        except Exception:
            logger.exception("Migration 005 downgrade failed")
            db.rollback()
            raise


if __name__ == "__main__":
    try:
        ok = downgrade() if "--downgrade" in sys.argv[1:] else run()
        sys.exit(0 if ok else 1)
    except Exception:
        logger.exception("Unhandled error during migration 005")
        sys.exit(1)
