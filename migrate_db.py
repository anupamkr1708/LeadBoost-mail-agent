"""
Database migration script for production upgrade.

Adds new fields to support semantic intelligence and state machine.
"""

import logging
from sqlalchemy import text

from mailer_agent.db import engine, init_db

logger = logging.getLogger("mailer_agent.migration")


def migrate_database():
    """
    Apply schema migrations to existing database.
    
    Safe to run multiple times (idempotent).
    """
    logger.info("Starting database migration...")
    
    # Ensure tables exist
    init_db()
    
    with engine.begin() as conn:
        # Check and add columns to contacts table
        _add_column_if_not_exists(
            conn, "contacts", "last_reply_at",
            "TIMESTAMP NULL"
        )
        _add_column_if_not_exists(
            conn, "contacts", "last_outbound_at",
            "TIMESTAMP NULL"
        )
        _add_column_if_not_exists(
            conn, "contacts", "buying_stage",
            "VARCHAR NULL"
        )
        _add_column_if_not_exists(
            conn, "contacts", "engagement_score",
            "FLOAT DEFAULT 0.0"
        )
        
        # Check and add columns to messages table
        _add_column_if_not_exists(
            conn, "messages", "semantic_analysis",
            "JSON NULL" if "postgresql" in str(conn.engine.url) else "TEXT NULL"
        )
        _add_column_if_not_exists(
            conn, "messages", "classification_success",
            "BOOLEAN NULL"
        )
        _add_column_if_not_exists(
            conn, "messages", "classification_failure_reason",
            "VARCHAR NULL"
        )
    
    logger.info("Database migration completed successfully")


def _add_column_if_not_exists(conn, table_name: str, column_name: str, column_def: str):
    """Add column if it doesn't exist (idempotent)."""
    
    # Check if column exists
    if "postgresql" in str(conn.engine.url):
        check_sql = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name=:table AND column_name=:column
        """)
        result = conn.execute(check_sql, {"table": table_name, "column": column_name})
    else:
        # SQLite
        check_sql = text(f"PRAGMA table_info({table_name})")
        result = conn.execute(check_sql)
        columns = [row[1] for row in result]
        exists = column_name in columns
        
        if exists:
            logger.debug(f"Column {table_name}.{column_name} already exists, skipping")
            return
        
        # Add column
        add_sql = text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")
        logger.info(f"Adding column {table_name}.{column_name}")
        conn.execute(add_sql)
        return
    
    # PostgreSQL path
    if result.fetchone():
        logger.debug(f"Column {table_name}.{column_name} already exists, skipping")
    else:
        add_sql = text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")
        logger.info(f"Adding column {table_name}.{column_name}")
        conn.execute(add_sql)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate_database()
    print("Migration complete!")
