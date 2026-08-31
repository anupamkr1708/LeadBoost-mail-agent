"""
Unified inbound email processing with semantic intelligence.

This module now delegates to reply_handler_v2.py as the single source of truth.
All reply processing uses multi-dimensional semantic classification, state machine
transitions, and proper failure handling.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from mailer_agent.mail.imap_reader import InboundEmail
# Import from v2 as authoritative implementation
from mailer_agent.mail.reply_handler_v2 import process_inbound_email_v2

logger = logging.getLogger("mailer_agent.mail.reply_handler")


def process_inbound_email(db: Session, email_in: InboundEmail) -> dict:
    """
    Process inbound email with full semantic intelligence.
    
    This is now a thin wrapper around the v2 implementation which provides:
    - Multi-dimensional semantic classification (not single intent)
    - State machine-driven transitions
    - Message deduplication by Message-ID
    - Explicit failure handling (provider failures ≠ neutral)
    - Proper timestamp tracking
    
    Returns dict with processing result.
    """
    return process_inbound_email_v2(db, email_in)


# Backward compatibility exports
__all__ = ["process_inbound_email"]
