"""
Database migration script for production hardening.

Adds:
1. organization_id to campaigns (multi-tenancy)
2. timezone to campaigns
3. Unique constraints for idempotency
4. Indexes for performance
5. Send attempt tracking fields

Run with: python migrations/001_production_hardening.py
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from mailer_agent.db import engine, session_scope

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def add_column_if_not_exists(db, table: str, column: str, definition: str) -> bool:
    """Add column if it doesn't already exist."""
    try:
        # Check if column exists (PostgreSQL)
        result = db.execute(text(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='{table}' AND column_name='{column}'
        """))
        if result.fetchone():
            logger.info(f"Column {table}.{column} already exists, skipping")
            return False
        
        # Add column
        db.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))
        logger.info(f"✓ Added column {table}.{column}")
        return True
    except Exception as e:
        # Fallback for SQLite (doesn't support information_schema)
        try:
            db.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))
            logger.info(f"✓ Added column {table}.{column}")
            return True
        except Exception as e2:
            if "duplicate column" in str(e2).lower() or "already exists" in str(e2).lower():
                logger.info(f"Column {table}.{column} already exists, skipping")
                return False
            logger.error(f"Failed to add column {table}.{column}: {e2}")
            raise


def create_index_if_not_exists(db, index_name: str, sql: str) -> bool:
    """Create index if it doesn't already exist."""
    try:
        db.execute(text(sql))
        logger.info(f"✓ Created index {index_name}")
        return True
    except Exception as e:
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            logger.info(f"Index {index_name} already exists, skipping")
            return False
        logger.error(f"Failed to create index {index_name}: {e}")
        raise


def add_constraint_if_not_exists(db, constraint_name: str, sql: str) -> bool:
    """Add constraint if it doesn't already exist."""
    try:
        db.execute(text(sql))
        logger.info(f"✓ Added constraint {constraint_name}")
        return True
    except Exception as e:
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            logger.info(f"Constraint {constraint_name} already exists, skipping")
            return False
        logger.error(f"Failed to add constraint {constraint_name}: {e}")
        raise


def migrate_campaigns(db):
    """Add multi-tenancy and timezone fields to campaigns."""
    logger.info("\n=== Migrating campaigns table ===")
    
    # Add organization_id for multi-tenancy
    add_column_if_not_exists(
        db, 
        "campaigns", 
        "organization_id", 
        "VARCHAR"
    )
    
    # Add timezone for campaign scheduling
    add_column_if_not_exists(
        db,
        "campaigns",
        "timezone",
        "VARCHAR DEFAULT 'UTC'"
    )
    
    # Create index on organization_id
    create_index_if_not_exists(
        db,
        "idx_campaigns_org_id",
        "CREATE INDEX IF NOT EXISTS idx_campaigns_org_id ON campaigns(organization_id)"
    )


def migrate_contacts(db):
    """Add unique constraint for campaign + email."""
    logger.info("\n=== Migrating contacts table ===")
    
    # Check if index already exists
    try:
        create_index_if_not_exists(
            db,
            "idx_contacts_campaign_email",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_contacts_campaign_email ON contacts(campaign_id, email)"
        )
    except Exception as e:
        if "unique constraint" in str(e).lower() or "duplicate" in str(e).lower():
            logger.warning(
                "Cannot create unique index - duplicate contacts exist. "
                "Clean up duplicates first with: "
                "DELETE FROM contacts WHERE id NOT IN "
                "(SELECT MIN(id) FROM contacts GROUP BY campaign_id, email)"
            )
        else:
            raise


def migrate_messages(db):
    """Add unique constraint for inbound Message-ID and send attempt tracking."""
    logger.info("\n=== Migrating messages table ===")
    
    # Add send attempt tracking fields
    add_column_if_not_exists(
        db,
        "messages",
        "send_attempt_count",
        "INTEGER DEFAULT 0"
    )
    
    add_column_if_not_exists(
        db,
        "messages",
        "last_send_attempt_at",
        "TIMESTAMP"
    )
    
    # For PostgreSQL: Create partial unique index on inbound Message-ID
    try:
        create_index_if_not_exists(
            db,
            "idx_messages_inbound_message_id",
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_inbound_message_id 
            ON messages(message_id_header) 
            WHERE direction = 'inbound' AND message_id_header IS NOT NULL
            """
        )
    except Exception as e:
        # SQLite doesn't support partial indexes in the same way
        if "syntax" in str(e).lower():
            logger.warning(
                "Partial unique index not supported (SQLite). "
                "Inbound Message-ID deduplication will be application-level only."
            )
        else:
            raise
    
    # Create index on message_id_header for thread correlation performance
    create_index_if_not_exists(
        db,
        "idx_messages_message_id",
        "CREATE INDEX IF NOT EXISTS idx_messages_message_id ON messages(message_id_header)"
    )


def migrate_suppression(db):
    """Ensure suppression list has unique constraint."""
    logger.info("\n=== Migrating suppression_list table ===")
    
    # The email field should already be unique (defined in model)
    # But ensure index exists
    create_index_if_not_exists(
        db,
        "idx_suppression_email",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_suppression_email ON suppression_list(email)"
    )


def verify_migration(db):
    """Verify migration completed successfully."""
    logger.info("\n=== Verifying migration ===")
    
    checks = []
    
    # Check campaigns has new fields
    try:
        result = db.execute(text("SELECT organization_id, timezone FROM campaigns LIMIT 1"))
        checks.append("✓ campaigns.organization_id exists")
        checks.append("✓ campaigns.timezone exists")
    except Exception as e:
        checks.append(f"✗ campaigns fields missing: {e}")
    
    # Check messages has send tracking
    try:
        result = db.execute(text("SELECT send_attempt_count FROM messages LIMIT 1"))
        checks.append("✓ messages.send_attempt_count exists")
    except Exception as e:
        checks.append(f"✗ messages fields missing: {e}")
    
    for check in checks:
        logger.info(check)
    
    return all("✓" in check for check in checks)


def run_migration():
    """Execute all migrations."""
    logger.info("Starting database migration for production hardening...")
    
    with session_scope() as db:
        try:
            # Run migrations in order
            migrate_campaigns(db)
            migrate_contacts(db)
            migrate_messages(db)
            migrate_suppression(db)
            
            # Commit all changes
            db.commit()
            logger.info("\n✓ All migrations committed successfully")
            
            # Verify
            if verify_migration(db):
                logger.info("\n✅ Migration completed and verified successfully!")
                return True
            else:
                logger.warning("\n⚠️ Migration completed but verification failed")
                return False
                
        except Exception as e:
            logger.error(f"\n❌ Migration failed: {e}")
            db.rollback()
            raise


if __name__ == "__main__":
    try:
        success = run_migration()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.exception("Migration failed with exception")
        sys.exit(1)
