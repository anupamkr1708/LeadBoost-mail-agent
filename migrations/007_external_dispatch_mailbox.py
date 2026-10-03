"""
Migration 007: external_dispatches.mailbox_id (M2-A).

Adds the reference from a dispatch to the Mailer-owned Mailbox it executes
through (see models.py::ExternalDispatch.mailbox_id).

Schema change
-------------
external_dispatches:
  + mailbox_id  INTEGER, NULL, FK -> mailboxes.id ON DELETE RESTRICT
    NULL for every existing row. The M2-A worker refuses a dispatch with a
    NULL mailbox_id before SMTP (FAILED, "no_mailbox"), so rows still QUEUED
    when this code is deployed will fail rather than send through the old
    global SMTP identity. Drain the queue (or accept those failures) before
    deploying; a failed dispatch is retried by the caller with a new
    idempotency key, as for any FAILED dispatch.

Requires migration 006 (mailboxes) first. Same conventions as 004/005/006:
standalone, idempotent, upgrade and downgrade. Downgrade drops the column and
the dispatch->mailbox link history.

Apply BEFORE deploying the M2-A code: the ORM selects this column on every
ExternalDispatch query.

Run:        python migrations/007_external_dispatch_mailbox.py
Downgrade:  python migrations/007_external_dispatch_mailbox.py --downgrade
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
COLUMN = "mailbox_id"


def _column_exists(db) -> bool:
    return any(c["name"] == COLUMN for c in inspect(db.connection()).get_columns(TABLE))


def add_mailbox_id_column(db) -> bool:
    if _column_exists(db):
        logger.info("Column %s.%s already exists -- skipping", TABLE, COLUMN)
        return False
    db.execute(text(
        f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} INTEGER NULL "
        "REFERENCES mailboxes (id) ON DELETE RESTRICT"
    ))
    logger.info("✓ Added %s.%s", TABLE, COLUMN)
    return True


def drop_mailbox_id_column(db) -> bool:
    if not _column_exists(db):
        logger.info("Column %s.%s does not exist -- skipping", TABLE, COLUMN)
        return False
    db.execute(text(f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}"))
    logger.info("✓ Dropped %s.%s", TABLE, COLUMN)
    return True


def verify(db) -> bool:
    try:
        db.execute(text(f"SELECT {COLUMN} FROM {TABLE} LIMIT 1"))
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("✗ %s.%s missing/broken: %s", TABLE, COLUMN, e)
        return False


def run() -> bool:
    with session_scope() as db:
        try:
            add_mailbox_id_column(db)
            db.commit()
            return verify(db)
        except Exception:
            logger.exception("Migration 007 failed")
            db.rollback()
            raise


def downgrade() -> bool:
    with session_scope() as db:
        try:
            drop_mailbox_id_column(db)
            db.commit()
            return True
        except Exception:
            logger.exception("Migration 007 downgrade failed")
            db.rollback()
            raise


if __name__ == "__main__":
    try:
        ok = downgrade() if "--downgrade" in sys.argv[1:] else run()
        sys.exit(0 if ok else 1)
    except Exception:
        logger.exception("Unhandled error during migration 007")
        sys.exit(1)
