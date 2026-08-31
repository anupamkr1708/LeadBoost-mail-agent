"""
Database migration: Add work claiming fields to contacts table.

Adds:
- claimed_by (worker identity)
- claimed_at (lease timestamp)

Run with: python migrations/002_add_work_claiming.py
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from mailer_agent.db import session_scope

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
        # Fallback for SQLite
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


def run_migration():
    """Execute migration."""
    logger.info("Adding work claiming fields to contacts table...")
    
    with session_scope() as db:
        try:
            # Add work claiming fields
            add_column_if_not_exists(
                db,
                "contacts",
                "claimed_by",
                "VARCHAR"
            )
            
            add_column_if_not_exists(
                db,
                "contacts",
                "claimed_at",
                "TIMESTAMP"
            )
            
            # Create index on claimed_by for fast worker queries
            create_index_if_not_exists(
                db,
                "idx_contacts_claimed_by",
                "CREATE INDEX IF NOT EXISTS idx_contacts_claimed_by ON contacts(claimed_by)"
            )
            
            # Commit all changes
            db.commit()
            logger.info("\n✅ Migration 002 completed successfully!")
            return True
                
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
