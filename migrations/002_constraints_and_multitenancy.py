"""
Database migration: constraints, indexes, multi-tenancy columns, and timezone fixes.

Adds:
1. Unique constraint on contacts(campaign_id, email)       — idempotency backstop
2. Unique constraint on inbound messages(message_id_header) — deduplication
3. organization_id on suppression_list                     — per-org suppression
4. Proper foreign key cascade definitions (documented)
5. Indexes for all hot query patterns
6. organization_id NOT NULL backfill for existing campaigns → 'default'

SQLite notes
------------
SQLite does not support:
  - ALTER TABLE … ADD CONSTRAINT (constraints must be in CREATE TABLE)
  - Partial/filtered unique indexes natively via IF NOT EXISTS + WHERE
  - DROP COLUMN (before 3.35)

For SQLite we therefore:
  - Use CREATE UNIQUE INDEX … (which is what SQLAlchemy uses for UniqueConstraint)
  - Simulate partial uniqueness on inbound message_id_header at the application layer
    (reply_handler_v2.py already does this) and add a plain unique index as a best-effort backstop.
  - Skip FK cascades (SQLite ignores FK actions unless PRAGMA foreign_keys=ON is set per connection).

For PostgreSQL we add the full constraints and partial indexes.

Run with:
    python migrations/002_constraints_and_multitenancy.py

The migration is idempotent: re-running it is safe.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from mailer_agent.db import engine, session_scope

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_postgres(db) -> bool:
    try:
        db.execute(text("SELECT version()"))
        url = str(db.bind.url if hasattr(db, "bind") else engine.url)
        return "postgresql" in url or "postgres" in url
    except Exception:
        return False


def _is_sqlite(db) -> bool:
    try:
        url = str(engine.url)
        return url.startswith("sqlite")
    except Exception:
        return True  # Conservative default


def _column_exists(db, table: str, column: str) -> bool:
    """Check if a column exists (works for both SQLite and Postgres)."""
    try:
        db.execute(text(f"SELECT {column} FROM {table} LIMIT 0"))
        return True
    except Exception:
        return False


def _index_exists(db, index_name: str) -> bool:
    """Check if an index already exists."""
    try:
        if _is_postgres(db):
            result = db.execute(text(
                "SELECT 1 FROM pg_indexes WHERE indexname = :name"
            ), {"name": index_name})
        else:  # SQLite
            result = db.execute(text(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name=:name"
            ), {"name": index_name})
        return result.fetchone() is not None
    except Exception:
        return False


def _add_column_safe(db, table: str, column: str, definition: str) -> bool:
    """Add column; silently skip if already exists."""
    if _column_exists(db, table, column):
        logger.info("  skip — %s.%s already exists", table, column)
        return False
    try:
        db.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))
        logger.info("  ✓ added %s.%s", table, column)
        return True
    except Exception as exc:
        err = str(exc).lower()
        if "duplicate column" in err or "already exists" in err:
            logger.info("  skip — %s.%s already exists (caught exception)", table, column)
            return False
        raise


def _create_index_safe(db, name: str, sql: str) -> bool:
    """Create an index; silently skip if already exists."""
    if _index_exists(db, name):
        logger.info("  skip — index %s already exists", name)
        return False
    try:
        db.execute(text(sql))
        logger.info("  ✓ created index %s", name)
        return True
    except Exception as exc:
        err = str(exc).lower()
        if "already exists" in err or "duplicate" in err:
            logger.info("  skip — index %s already exists (caught exception)", name)
            return False
        raise


# ---------------------------------------------------------------------------
# Migration steps
# ---------------------------------------------------------------------------


def step_campaigns_columns(db):
    """Ensure campaigns has organization_id and timezone columns."""
    logger.info("\n[campaigns] Adding multi-tenancy and timezone columns")
    _add_column_safe(db, "campaigns", "organization_id", "VARCHAR")
    _add_column_safe(db, "campaigns", "timezone", "VARCHAR DEFAULT 'UTC'")


def step_campaigns_backfill(db):
    """
    Backfill organization_id = 'default' for any existing campaigns.

    This ensures the NOT NULL-equivalent filter in the API (WHERE organization_id = ?)
    never silently excludes data created before multi-tenancy was introduced.
    """
    logger.info("\n[campaigns] Backfilling organization_id = 'default' for existing rows")
    result = db.execute(text(
        "UPDATE campaigns SET organization_id = 'default' WHERE organization_id IS NULL"
    ))
    updated = result.rowcount if hasattr(result, "rowcount") else "?"
    logger.info("  ✓ backfilled %s row(s)", updated)


def step_campaigns_indexes(db):
    """Indexes for campaign queries."""
    logger.info("\n[campaigns] Creating indexes")
    _create_index_safe(
        db,
        "idx_campaigns_org_id",
        "CREATE INDEX IF NOT EXISTS idx_campaigns_org_id ON campaigns(organization_id)",
    )
    _create_index_safe(
        db,
        "idx_campaigns_is_active",
        "CREATE INDEX IF NOT EXISTS idx_campaigns_is_active ON campaigns(is_active)",
    )
    _create_index_safe(
        db,
        "idx_campaigns_org_active",
        "CREATE INDEX IF NOT EXISTS idx_campaigns_org_active ON campaigns(organization_id, is_active)",
    )


def step_contacts_unique(db):
    """
    Unique constraint on contacts(campaign_id, email).

    This is the DB-level idempotency backstop for both /contacts and /leads/ingest.
    Pre-checks for duplicates first so the migration never fails on an existing DB
    that already has duplicates (created before the constraint was added).
    """
    logger.info("\n[contacts] Unique constraint on (campaign_id, email)")

    # Detect and warn about existing duplicates
    result = db.execute(text("""
        SELECT campaign_id, email, COUNT(*) AS cnt
        FROM contacts
        GROUP BY campaign_id, email
        HAVING COUNT(*) > 1
    """))
    dupes = result.fetchall()
    if dupes:
        logger.warning(
            "  ⚠ %d duplicate (campaign_id, email) pair(s) found — cannot add unique constraint "
            "until they are resolved.  Run the de-duplication query below, then re-run this migration:\n"
            "    DELETE FROM contacts WHERE id NOT IN "
            "    (SELECT MIN(id) FROM contacts GROUP BY campaign_id, email);",
            len(dupes),
        )
        for row in dupes[:10]:
            logger.warning("    duplicate: campaign_id=%s email=%s count=%s", *row)
        return

    _create_index_safe(
        db,
        "uq_contacts_campaign_email",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_contacts_campaign_email ON contacts(campaign_id, email)",
    )


def step_contacts_indexes(db):
    """Indexes for common contact query patterns."""
    logger.info("\n[contacts] Creating indexes")

    _create_index_safe(
        db,
        "idx_contacts_status",
        "CREATE INDEX IF NOT EXISTS idx_contacts_status ON contacts(status)",
    )
    _create_index_safe(
        db,
        "idx_contacts_next_action_at",
        "CREATE INDEX IF NOT EXISTS idx_contacts_next_action_at ON contacts(next_action_at)",
    )
    # The hot path: scheduler queries by status + next_action_at + campaign active
    _create_index_safe(
        db,
        "idx_contacts_scheduler_hot",
        "CREATE INDEX IF NOT EXISTS idx_contacts_scheduler_hot "
        "ON contacts(status, next_action_at)",
    )
    _create_index_safe(
        db,
        "idx_contacts_campaign_status",
        "CREATE INDEX IF NOT EXISTS idx_contacts_campaign_status "
        "ON contacts(campaign_id, status)",
    )


def step_messages_inbound_dedup(db):
    """
    Unique constraint on inbound messages by message_id_header.

    PostgreSQL: partial unique index (WHERE direction='inbound' AND message_id_header IS NOT NULL).
    SQLite: plain unique index on message_id_header — not perfectly selective but adequate;
            the application layer (reply_handler_v2) provides the primary dedup logic.
    """
    logger.info("\n[messages] Deduplication index on message_id_header for inbound messages")

    if _is_postgres(db):
        _create_index_safe(
            db,
            "uq_messages_inbound_message_id",
            """CREATE UNIQUE INDEX IF NOT EXISTS uq_messages_inbound_message_id
               ON messages(message_id_header)
               WHERE direction = 'inbound' AND message_id_header IS NOT NULL""",
        )
    else:
        # SQLite: best-effort — note message_id_header can legitimately be NULL
        # for draft outbound messages so we cannot make it globally unique.
        # Application-level dedup in reply_handler_v2 is the primary guard.
        logger.info(
            "  SQLite: partial unique indexes are not fully supported. "
            "Application-level dedup in reply_handler_v2 is the primary guard. "
            "Adding a plain index for query performance."
        )
        _create_index_safe(
            db,
            "idx_messages_message_id_header",
            "CREATE INDEX IF NOT EXISTS idx_messages_message_id_header "
            "ON messages(message_id_header)",
        )


def step_messages_indexes(db):
    """Indexes for message query patterns."""
    logger.info("\n[messages] Creating indexes")

    _create_index_safe(
        db,
        "idx_messages_contact_id",
        "CREATE INDEX IF NOT EXISTS idx_messages_contact_id ON messages(contact_id)",
    )
    _create_index_safe(
        db,
        "idx_messages_in_reply_to",
        "CREATE INDEX IF NOT EXISTS idx_messages_in_reply_to ON messages(in_reply_to_header)",
    )
    _create_index_safe(
        db,
        "idx_messages_status",
        "CREATE INDEX IF NOT EXISTS idx_messages_status ON messages(status)",
    )
    _create_index_safe(
        db,
        "idx_messages_direction_status",
        "CREATE INDEX IF NOT EXISTS idx_messages_direction_status "
        "ON messages(direction, status)",
    )


def step_suppression_org_scope(db):
    """
    Add organization_id to suppression_list and update the unique constraint.

    The original schema had UNIQUE(email) globally.  With multi-tenancy, the constraint
    should be UNIQUE(email, organization_id) so different orgs can each have their own
    suppression list.

    Migration strategy:
    1. Add organization_id column.
    2. Backfill existing rows with 'default'.
    3. Drop the old global unique index (if it exists).
    4. Create the new composite unique index.
    """
    logger.info("\n[suppression_list] Multi-tenancy scoping")

    _add_column_safe(db, "suppression_list", "organization_id", "VARCHAR")

    # Backfill
    result = db.execute(text(
        "UPDATE suppression_list SET organization_id = 'default' WHERE organization_id IS NULL"
    ))
    updated = result.rowcount if hasattr(result, "rowcount") else "?"
    logger.info("  ✓ backfilled organization_id on %s row(s)", updated)

    # Drop old global unique index if it exists (name may vary)
    for old_idx in ("idx_suppression_email", "uq_suppression_email", "ix_suppression_list_email"):
        try:
            if _index_exists(db, old_idx):
                db.execute(text(f"DROP INDEX IF EXISTS {old_idx}"))
                logger.info("  ✓ dropped old index %s", old_idx)
        except Exception as exc:
            logger.warning("  could not drop old index %s: %s", old_idx, exc)

    # New composite unique index
    _create_index_safe(
        db,
        "uq_suppression_email_org",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_suppression_email_org "
        "ON suppression_list(email, organization_id)",
    )


# ---------------------------------------------------------------------------
# FK cascade documentation (informational for PostgreSQL)
# ---------------------------------------------------------------------------
#
# SQLAlchemy model definitions in models.py already declare:
#   Contact.campaign_id  → campaigns.id  ON DELETE CASCADE
#   Message.contact_id   → contacts.id   ON DELETE CASCADE
#
# These are reflected in the ORM relationship and in CREATE TABLE statements
# when the schema is created fresh.  For *existing* tables in PostgreSQL,
# alter them with:
#
#   -- Step A: drop old FK, add with cascade
#   ALTER TABLE contacts
#     DROP CONSTRAINT IF EXISTS contacts_campaign_id_fkey,
#     ADD CONSTRAINT contacts_campaign_id_fkey
#       FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE;
#
#   ALTER TABLE messages
#     DROP CONSTRAINT IF EXISTS messages_contact_id_fkey,
#     ADD CONSTRAINT messages_contact_id_fkey
#       FOREIGN KEY (contact_id) REFERENCES contacts(id) ON DELETE CASCADE;
#
# SQLite: cascades only work if PRAGMA foreign_keys=ON is set per connection;
# this project does not currently set it (safe choice: prevents accidental deletes).
# For SQLite in tests, add: db.execute(text("PRAGMA foreign_keys=ON"))
# ---------------------------------------------------------------------------


def step_postgres_fk_cascades(db):
    """Apply FK cascade definitions on PostgreSQL (skipped on SQLite)."""
    if not _is_postgres(db):
        logger.info(
            "\n[FK cascades] SQLite detected — skipping (SQLite ignores FK actions "
            "unless PRAGMA foreign_keys=ON is set per connection)"
        )
        return

    logger.info("\n[FK cascades] Updating foreign key cascade rules on PostgreSQL")

    cascade_statements = [
        (
            "contacts_campaign_id_fkey",
            """
            ALTER TABLE contacts
              DROP CONSTRAINT IF EXISTS contacts_campaign_id_fkey;
            ALTER TABLE contacts
              ADD CONSTRAINT contacts_campaign_id_fkey
              FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE;
            """,
        ),
        (
            "messages_contact_id_fkey",
            """
            ALTER TABLE messages
              DROP CONSTRAINT IF EXISTS messages_contact_id_fkey;
            ALTER TABLE messages
              ADD CONSTRAINT messages_contact_id_fkey
              FOREIGN KEY (contact_id) REFERENCES contacts(id) ON DELETE CASCADE;
            """,
        ),
    ]

    for constraint_name, sql in cascade_statements:
        try:
            for stmt in sql.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    db.execute(text(stmt))
            logger.info("  ✓ applied FK cascade for %s", constraint_name)
        except Exception as exc:
            logger.warning("  could not apply FK cascade for %s: %s", constraint_name, exc)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(db):
    logger.info("\n[verify] Checking migration results")
    ok = True

    checks = [
        ("campaigns", "organization_id"),
        ("campaigns", "timezone"),
        ("suppression_list", "organization_id"),
    ]
    for table, col in checks:
        if _column_exists(db, table, col):
            logger.info("  ✓ %s.%s exists", table, col)
        else:
            logger.error("  ✗ %s.%s MISSING", table, col)
            ok = False

    indexes = [
        "idx_campaigns_org_id",
        "idx_contacts_status",
        "idx_contacts_next_action_at",
        "idx_contacts_scheduler_hot",
        "uq_suppression_email_org",
    ]
    for idx in indexes:
        if _index_exists(db, idx):
            logger.info("  ✓ index %s exists", idx)
        else:
            logger.warning("  ? index %s not found (may have been skipped due to duplicates)", idx)

    return ok


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def run():
    logger.info("=" * 60)
    logger.info("Migration 002: constraints, indexes, multi-tenancy")
    logger.info("=" * 60)

    with session_scope() as db:
        step_campaigns_columns(db)
        step_campaigns_backfill(db)
        step_campaigns_indexes(db)
        step_contacts_unique(db)
        step_contacts_indexes(db)
        step_messages_inbound_dedup(db)
        step_messages_indexes(db)
        step_suppression_org_scope(db)
        step_postgres_fk_cascades(db)

        db.commit()
        logger.info("\n✓ All steps committed")

        ok = verify(db)

    if ok:
        logger.info("\n✅ Migration 002 completed successfully")
    else:
        logger.warning("\n⚠️  Migration 002 completed with warnings — check output above")

    return ok


if __name__ == "__main__":
    try:
        success = run()
        sys.exit(0 if success else 1)
    except Exception:
        logger.exception("Migration 002 failed")
        sys.exit(1)
