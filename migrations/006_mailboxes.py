"""
Migration 006: mailboxes table (M1, Mailer-owned Mailbox foundation).

Creates `mailboxes` via Mailbox.__table__.create(checkfirst=True) -- the same
pattern as 004, so the DDL cannot drift from models.py and coexists with
db.py::init_db()'s create_all (whichever runs first wins; the other no-ops).
Purely additive: nothing existing reads or writes this table.

Credentials in this table are Fernet ciphertext produced with
MAILBOX_ENCRYPTION_KEY (deployment configuration, not stored in the database).

Run:        python migrations/006_mailboxes.py
Downgrade:  python migrations/006_mailboxes.py --downgrade
            DESTRUCTIVE: drops the mailboxes table and every stored mailbox
            record and encrypted credential.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import inspect, text

from mailer_agent.db import session_scope
from mailer_agent.models import Mailbox

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TABLE = Mailbox.__tablename__


def _table_exists(db) -> bool:
    return inspect(db.connection()).has_table(TABLE)


def create_mailboxes_table(db) -> bool:
    if _table_exists(db):
        logger.info("Table %s already exists -- skipping", TABLE)
        return False
    Mailbox.__table__.create(bind=db.connection(), checkfirst=True)
    logger.info("✓ Created %s", TABLE)
    return True


def drop_mailboxes_table(db) -> bool:
    if not _table_exists(db):
        logger.info("Table %s does not exist -- skipping", TABLE)
        return False
    count = db.execute(text(f"SELECT COUNT(*) FROM {TABLE}")).scalar()
    logger.warning("Dropping %s: discarding %s mailbox record(s) and their encrypted credentials", TABLE, count)
    Mailbox.__table__.drop(bind=db.connection(), checkfirst=True)
    logger.info("✓ Dropped %s", TABLE)
    return True


def verify(db) -> bool:
    insp = inspect(db.connection())
    if not insp.has_table(TABLE):
        logger.error("✗ %s missing", TABLE)
        return False
    uniques = {tuple(u["column_names"]) for u in insp.get_unique_constraints(TABLE)}
    uniques |= {tuple(i["column_names"]) for i in insp.get_indexes(TABLE) if i.get("unique")}
    ok = {("organization_id", "email_address"), ("public_reference",)} <= uniques
    logger.info("✓ %s present with expected unique constraints" if ok else "✗ %s unique constraints missing", TABLE)
    return ok


def run() -> bool:
    logger.info("Starting migration 006: mailboxes")
    with session_scope() as db:
        try:
            create_mailboxes_table(db)
            db.commit()
            ok = verify(db)
            logger.info("✅ Migration 006 completed" if ok else "⚠️ Migration 006 verification failed")
            return ok
        except Exception:
            logger.exception("Migration 006 failed")
            db.rollback()
            raise


def downgrade() -> bool:
    logger.info("Starting migration 006 DOWNGRADE")
    with session_scope() as db:
        try:
            drop_mailboxes_table(db)
            db.commit()
            return True
        except Exception:
            logger.exception("Migration 006 downgrade failed")
            db.rollback()
            raise


if __name__ == "__main__":
    try:
        ok = downgrade() if "--downgrade" in sys.argv[1:] else run()
        sys.exit(0 if ok else 1)
    except Exception:
        logger.exception("Unhandled error during migration 006")
        sys.exit(1)
