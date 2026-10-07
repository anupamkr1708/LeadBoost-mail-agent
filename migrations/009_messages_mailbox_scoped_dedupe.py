"""
Migration 009: messages.mailbox_id + mailbox-scoped Message-ID dedupe (M3).

Schema change
-------------
messages:
  + mailbox_id  INTEGER, NULL, FK -> mailboxes.id ON DELETE RESTRICT
    NULL for every existing row (all outbound, webhook and legacy-global-IMAP
    inbound). Set only by mailbox-bound IMAP polling.
  - uq_messages_message_id_header          (global UNIQUE(message_id_header))
  + uq_messages_message_id_no_mailbox      UNIQUE(message_id_header)
                                           WHERE mailbox_id IS NULL
  + uq_messages_mailbox_message_id         UNIQUE(mailbox_id, message_id_header)
                                           WHERE mailbox_id IS NOT NULL

Behaviour is unchanged for every existing row: they all have mailbox_id NULL,
so the first partial index enforces exactly the old global rule. The second
index is what lets two organizations each store the same RFC Message-ID
received in their own mailbox. NULL Message-IDs stay unconstrained.

Requires migration 006 (mailboxes) first. Idempotent. Apply BEFORE deploying
M3 code: the ORM selects messages.mailbox_id on every Message query.

PostgreSQL: the old constraint is dropped and the partial indexes created in
one transaction. SQLite (local development only): the column and the new
indexes are added, but a UNIQUE constraint that was declared inline in
CREATE TABLE cannot be dropped, so a SQLite database created by pre-M3 code
keeps the global rule until it is recreated. Recreate dev databases from the
models (init_db / create_all) instead.

Downgrade restores the global constraint and drops the column. It refuses to
run if the same Message-ID now exists in more than one row (which the
mailbox-scoped rule permits and the global rule does not).

Run:        python migrations/009_messages_mailbox_scoped_dedupe.py
Downgrade:  python migrations/009_messages_mailbox_scoped_dedupe.py --downgrade
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import inspect, text

from mailer_agent.db import session_scope

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TABLE = "messages"
COLUMN = "mailbox_id"
OLD_CONSTRAINT = "uq_messages_message_id_header"
IDX_NO_MAILBOX = "uq_messages_message_id_no_mailbox"
IDX_MAILBOX = "uq_messages_mailbox_message_id"


def _dialect(db) -> str:
    return db.get_bind().dialect.name


def _column_exists(db) -> bool:
    return any(c["name"] == COLUMN for c in inspect(db.connection()).get_columns(TABLE))


def _index_exists(db, name: str) -> bool:
    return any(i["name"] == name for i in inspect(db.connection()).get_indexes(TABLE))


def _old_constraint_exists_postgres(db) -> bool:
    return db.execute(
        text(
            "SELECT 1 FROM information_schema.table_constraints "
            "WHERE table_name = :t AND constraint_name = :c"
        ),
        {"t": TABLE, "c": OLD_CONSTRAINT},
    ).fetchone() is not None


def upgrade_schema(db) -> None:
    if not _column_exists(db):
        db.execute(text(
            f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} INTEGER NULL "
            "REFERENCES mailboxes (id) ON DELETE RESTRICT"
        ))
        logger.info("✓ Added %s.%s", TABLE, COLUMN)
    else:
        logger.info("Column %s.%s already exists -- skipping", TABLE, COLUMN)

    # Create the replacement indexes BEFORE dropping the old constraint so the
    # no-mailbox rule is never unenforced, even briefly.
    if not _index_exists(db, IDX_NO_MAILBOX):
        db.execute(text(
            f"CREATE UNIQUE INDEX {IDX_NO_MAILBOX} ON {TABLE} (message_id_header) "
            f"WHERE {COLUMN} IS NULL"
        ))
        logger.info("✓ Created %s", IDX_NO_MAILBOX)
    if not _index_exists(db, IDX_MAILBOX):
        db.execute(text(
            f"CREATE UNIQUE INDEX {IDX_MAILBOX} ON {TABLE} ({COLUMN}, message_id_header) "
            f"WHERE {COLUMN} IS NOT NULL"
        ))
        logger.info("✓ Created %s", IDX_MAILBOX)

    if _dialect(db) == "postgresql":
        if _old_constraint_exists_postgres(db):
            db.execute(text(f"ALTER TABLE {TABLE} DROP CONSTRAINT {OLD_CONSTRAINT}"))
            logger.info("✓ Dropped global constraint %s", OLD_CONSTRAINT)
    else:
        logger.warning(
            "Non-PostgreSQL database: an inline global UNIQUE(message_id_header) from "
            "pre-M3 CREATE TABLE cannot be dropped. Recreate this development "
            "database from the models to get mailbox-scoped dedupe."
        )


def downgrade_schema(db) -> bool:
    if _column_exists(db):
        dupes = db.execute(text(
            f"SELECT message_id_header FROM {TABLE} WHERE message_id_header IS NOT NULL "
            "GROUP BY message_id_header HAVING COUNT(*) > 1 LIMIT 5"
        )).fetchall()
        if dupes:
            logger.error(
                "Refusing downgrade: Message-IDs now stored more than once (e.g. %s). "
                "Resolve them manually first.", [r[0] for r in dupes],
            )
            return False
    if _dialect(db) == "postgresql" and not _old_constraint_exists_postgres(db):
        db.execute(text(
            f"ALTER TABLE {TABLE} ADD CONSTRAINT {OLD_CONSTRAINT} UNIQUE (message_id_header)"
        ))
    for name in (IDX_MAILBOX, IDX_NO_MAILBOX):
        if _index_exists(db, name):
            db.execute(text(f"DROP INDEX {name}"))
    if _column_exists(db):
        db.execute(text(f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}"))
    return True


def verify(db) -> bool:
    try:
        db.execute(text(f"SELECT {COLUMN} FROM {TABLE} LIMIT 1"))
        return _index_exists(db, IDX_NO_MAILBOX) and _index_exists(db, IDX_MAILBOX)
    except Exception as e:  # noqa: BLE001
        logger.error("✗ Verification failed: %s", e)
        return False


def run() -> bool:
    with session_scope() as db:
        try:
            upgrade_schema(db)
            db.commit()
            return verify(db)
        except Exception:
            logger.exception("Migration 009 failed")
            db.rollback()
            raise


def downgrade() -> bool:
    with session_scope() as db:
        try:
            ok = downgrade_schema(db)
            if ok:
                db.commit()
            else:
                db.rollback()
            return ok
        except Exception:
            logger.exception("Migration 009 downgrade failed")
            db.rollback()
            raise


if __name__ == "__main__":
    try:
        ok = downgrade() if "--downgrade" in sys.argv[1:] else run()
        sys.exit(0 if ok else 1)
    except Exception:
        logger.exception("Unhandled error during migration 009")
        sys.exit(1)
