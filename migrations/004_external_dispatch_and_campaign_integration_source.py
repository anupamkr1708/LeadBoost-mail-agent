"""
Migration 004: LeadBoost integration schema -- campaigns.integration_source
and the external_dispatches table.

Phase C (see mailer_agent/api/integrations.py and models.py::ExternalDispatch
for the full design). Adds the durable schema the new
POST /integrations/leadboost/outreach-actions endpoint depends on. Adds no
behavior on its own -- these are additive, unused-until-referenced schema
objects; nothing existing reads or writes them before the endpoint from this
same phase is deployed alongside this migration.

Schema changes
--------------
campaigns:
  + integration_source  VARCHAR, NULL   -- which external integration this
    campaign is the fixed home for (e.g. "leadboost"). NULL for every
    existing/ordinary campaign; never touched by any pre-existing code path.

  + UNIQUE INDEX uq_campaigns_org_integration_source ON
    campaigns(organization_id, integration_source)
    -- Both PostgreSQL and SQLite treat NULL as distinct-from-itself in a
       unique index (verified against this repo's actual engines, see
       mailer_agent/db.py:23 for the Postgres-prod/SQLite-dev-test split),
       so every existing campaign (integration_source IS NULL) is
       completely unaffected by this index; only rows that explicitly set
       integration_source='leadboost' are constrained to one per
       organization_id. A plain index, not a named UNIQUE CONSTRAINT,
       because SQLite has no ALTER TABLE ... ADD CONSTRAINT -- CREATE
       UNIQUE INDEX is the one syntax that works unchanged on both engines
       against an existing table.

New table
---------
external_dispatches -- see models.py::ExternalDispatch for the full,
  column-by-column design rationale (idempotency backstop, tenancy,
  correlation-vs-authorization distinction, public reference, claim
  lease fields). Created via
  ExternalDispatch.__table__.create(bind=..., checkfirst=True) rather
  than hand-written CREATE TABLE SQL: unlike campaigns.integration_source
  above, this is a brand-new table with zero existing rows to reason
  about, so there is no data-safety reason to hand-rewrite the DDL that
  models.py already defines authoritatively -- and letting SQLAlchemy's
  own metadata generate it (the same metadata Base.metadata.create_all()
  already uses on every app startup via db.py::init_db()) means this
  migration can never drift out of sync with the ORM model by construction.
  Still given an explicit, numbered migration step (rather than relying
  solely on init_db()'s implicit create_all() on next deploy) so the
  schema change is reviewable and applied by the same deploy step as the
  campaigns column/index change above, not silently by whichever process
  happens to start first.

This migration does not touch any existing row's data, and does not
change the type or nullability of any pre-existing column.

Run with::

    python migrations/004_external_dispatch_and_campaign_integration_source.py

DEPLOYMENT NOTE (see render.yaml / migrations/README.md): this repository's
render.yaml buildCommand does not currently run any migration script
automatically -- this one, like 001-003 before it, must be applied via an
explicit deploy step (a Render pre-deploy command, once configured, or the
existing manual runbook) before the Phase C integration endpoint is enabled
in an environment. Wiring that deploy step is out of scope for this batch
(tracked as a later-phase item); this migration is written to be safely
re-run either way, since every step here is idempotent.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from mailer_agent.db import session_scope
from mailer_agent.models import ExternalDispatch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (same pattern as 001_production_hardening.py / 002_work_claiming.py)
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

def add_campaign_integration_source_column(db) -> None:
    logger.info("\n=== Adding campaigns.integration_source ===")
    _add_column_if_missing(db, "campaigns", "integration_source", "VARCHAR")


def add_campaign_integration_source_index(db) -> None:
    logger.info("\n=== Adding uq_campaigns_org_integration_source ===")
    _create_index_if_missing(
        db,
        "uq_campaigns_org_integration_source",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_campaigns_org_integration_source "
        "ON campaigns(organization_id, integration_source)",
    )


def create_external_dispatches_table(db) -> bool:
    logger.info("\n=== Creating external_dispatches table (if not exists) ===")
    try:
        # bind=db.connection() (not a fresh engine connection) so this
        # DDL runs inside the same transaction as the ALTER/CREATE INDEX
        # statements above, via the same session -- see session_scope()
        # in mailer_agent/db.py, which commits/rolls back this whole
        # function's work as one unit.
        ExternalDispatch.__table__.create(bind=db.connection(), checkfirst=True)
        logger.info("✓ external_dispatches table present")
        return True
    except Exception as e:
        logger.error("✗ Failed to create external_dispatches: %s", e)
        raise


def verify(db) -> bool:
    logger.info("\n=== Verifying migration 004 ===")
    checks = []
    try:
        db.execute(text("SELECT integration_source FROM campaigns LIMIT 1"))
        checks.append("✓ campaigns.integration_source exists")
    except Exception as e:
        checks.append(f"✗ campaigns.integration_source missing: {e}")

    try:
        db.execute(
            text(
                "SELECT organization_id, idempotency_key, campaign_id, contact_id, "
                "message_id, request_fingerprint, public_reference, state "
                "FROM external_dispatches LIMIT 1"
            )
        )
        checks.append("✓ external_dispatches is queryable with expected columns")
    except Exception as e:
        checks.append(f"✗ external_dispatches missing/broken: {e}")

    for check in checks:
        logger.info(check)
    return all("✓" in c for c in checks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> bool:
    logger.info("Starting migration 004: LeadBoost integration schema")
    with session_scope() as db:
        try:
            add_campaign_integration_source_column(db)
            add_campaign_integration_source_index(db)
            create_external_dispatches_table(db)
            db.commit()
            logger.info("\n✓ Migration 004 committed")

            if verify(db):
                logger.info("\n✅ Migration 004 completed and verified!")
                return True
            else:
                logger.warning("\n⚠️ Migration 004 committed but verification failed")
                return False
        except Exception:
            logger.exception("Migration 004 failed")
            db.rollback()
            raise


if __name__ == "__main__":
    try:
        ok = run()
        sys.exit(0 if ok else 1)
    except Exception:
        logger.exception("Unhandled error during migration 004")
        sys.exit(1)
