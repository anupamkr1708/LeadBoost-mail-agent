"""
Migration 008: external_dispatches.message_id becomes NULLABLE (M2-B).

Generated outreach is now accepted BEFORE its message exists: the dispatch is
committed with message_id NULL and the generation worker later creates the one
Message and sets message_id (state GENERATING -> QUEUED). No column is added;
the FK to messages.id (ON DELETE RESTRICT) is unchanged. Existing rows all
have a message_id and are untouched. Nothing sends a row whose message_id is
NULL (the send claim requires it). The new internal state value "generating"
needs no DDL (state is a plain string column).

PostgreSQL only: it relaxes NOT NULL with ALTER COLUMN ... DROP NOT NULL.
SQLite cannot alter nullability in place; SQLite databases are dev/test only
and are created from models.py (create_all), which already has the nullable
column -- on SQLite this script reports that and does nothing.

Apply BEFORE deploying the M2-B code (an INSERT with message_id NULL fails
against the old constraint), and after 007.

Downgrade restores NOT NULL and therefore REFUSES to run while any row has
message_id NULL (accepted-but-ungenerated or failed-before-generation rows):
drain/fail those first, or it will error out rather than lose data.

Run:        python migrations/008_external_dispatch_deferred_message.py
Downgrade:  python migrations/008_external_dispatch_deferred_message.py --downgrade
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
COLUMN = "message_id"


def _is_postgres(db) -> bool:
    return db.connection().dialect.name == "postgresql"


def _is_nullable(db) -> bool:
    cols = {c["name"]: c for c in inspect(db.connection()).get_columns(TABLE)}
    return bool(cols[COLUMN]["nullable"])


def make_message_id_nullable(db) -> bool:
    if not _is_postgres(db):
        logger.info("Not PostgreSQL: %s.%s nullability comes from models.py/create_all -- skipping", TABLE, COLUMN)
        return False
    if _is_nullable(db):
        logger.info("%s.%s is already nullable -- skipping", TABLE, COLUMN)
        return False
    db.execute(text(f"ALTER TABLE {TABLE} ALTER COLUMN {COLUMN} DROP NOT NULL"))
    logger.info("✓ %s.%s is now nullable", TABLE, COLUMN)
    return True


def make_message_id_required(db) -> bool:
    if not _is_postgres(db):
        logger.info("Not PostgreSQL -- skipping")
        return False
    if not _is_nullable(db):
        logger.info("%s.%s is already NOT NULL -- skipping", TABLE, COLUMN)
        return False
    pending = db.execute(text(f"SELECT COUNT(*) FROM {TABLE} WHERE {COLUMN} IS NULL")).scalar()
    if pending:
        raise RuntimeError(
            f"{pending} dispatch row(s) have message_id NULL (awaiting or failed before generation); "
            "resolve them before downgrading"
        )
    db.execute(text(f"ALTER TABLE {TABLE} ALTER COLUMN {COLUMN} SET NOT NULL"))
    logger.info("✓ %s.%s is NOT NULL again", TABLE, COLUMN)
    return True


def run() -> bool:
    with session_scope() as db:
        try:
            make_message_id_nullable(db)
            db.commit()
            return True
        except Exception:
            logger.exception("Migration 008 failed")
            db.rollback()
            raise


def downgrade() -> bool:
    with session_scope() as db:
        try:
            make_message_id_required(db)
            db.commit()
            return True
        except Exception:
            logger.exception("Migration 008 downgrade failed")
            db.rollback()
            raise


if __name__ == "__main__":
    try:
        ok = downgrade() if "--downgrade" in sys.argv[1:] else run()
        sys.exit(0 if ok else 1)
    except Exception:
        logger.exception("Unhandled error during migration 008")
        sys.exit(1)
