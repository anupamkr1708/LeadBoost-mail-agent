"""
Unified follow-up engine with conversation awareness.

This module now delegates to engine_v2.py (IntegratedFollowUpEngine) as the single source of truth.
All follow-up scheduling uses conversation-aware logic instead of blind timing.
"""

from __future__ import annotations

import logging
import random
import time

from sqlalchemy.orm import Session

from mailer_agent.config import get_settings
from mailer_agent.models import Contact, SuppressionEntry
# Import from v2 as authoritative implementation
from mailer_agent.followup.engine_v2 import integrated_engine

logger = logging.getLogger("mailer_agent.followup.engine")
settings = get_settings()


def is_suppressed(db: Session, email_addr: str, org_id: str | None = None) -> bool:
    """
    Check if email is on suppression list.

    If org_id is provided (API request context), checks only that org's list.
    If org_id is None (scheduler context), checks across all orgs — this is
    the safe default: if an address unsubscribed from *any* org's campaign
    we don't want the scheduler silently sending to it from another org.
    Callers with a known org should pass it for correct per-org semantics.
    """
    query = db.query(SuppressionEntry).filter_by(email=email_addr)
    if org_id is not None:
        query = query.filter_by(organization_id=org_id)
    return query.first() is not None


def stagger_sleep() -> None:
    """
    Real prospects landing seconds apart from the same sender reads
    as automated bulk activity. Called between consecutive sends in a
    batch -- never before the first one, since a single send shouldn't
    be delayed for no reason.
    """
    delay = settings.send_delay_seconds + random.uniform(0, settings.send_jitter_seconds)
    time.sleep(delay)


def send_initial_outreach(db: Session, contact: Contact) -> dict:
    """
    Send initial outreach (delegates to integrated engine).
    
    Now uses conversation-aware scheduling and state machine transitions.
    """
    return integrated_engine.send_initial_outreach(db, contact)


def send_followup_if_due(db: Session, contact: Contact) -> dict | None:
    """
    Send follow-up if due (delegates to integrated engine).
    
    Now checks conversation state before sending (not just calendar timing).
    """
    return integrated_engine.send_followup_if_due(db, contact)


def run_followup_cycle(
    db: Session, limit: int | None = None, worker_id: str | None = None
) -> list[dict]:
    """
    Run follow-up cycle for all due contacts (delegates to integrated engine).

    Now uses conversation-aware checks, state machine validation, and
    database-level work claiming (SELECT FOR UPDATE SKIP LOCKED on PostgreSQL).
    Pass ``worker_id`` to attribute claimed rows to this worker; defaults to
    the value of WORKER_ID env var (or "worker-<hostname>") if not supplied.
    """
    return integrated_engine.run_followup_cycle(db, limit=limit, worker_id=worker_id)


def run_new_contact_cycle(db: Session, limit: int | None = None) -> list[dict]:
    """
    Send initial outreach to all NEW contacts (legacy function).
    
    This was previously used by scheduler but is now superseded by
    durable campaign start logic. Kept for backward compatibility.
    """
    from mailer_agent.models import Campaign, ContactStatus
    
    limit = limit or settings.max_sends_per_cycle
    contacts = (
        db.query(Contact)
        .join(Campaign)
        .filter(Contact.status == ContactStatus.NEW.value, Campaign.is_active.is_(True))
        .limit(limit)
        .all()
    )
    results = []
    for i, c in enumerate(contacts):
        if i > 0:
            stagger_sleep()
        results.append(send_initial_outreach(db, c))
    db.commit()
    return results


# Backward compatibility exports
__all__ = [
    "is_suppressed",
    "send_initial_outreach",
    "send_followup_if_due",
    "run_followup_cycle",
    "run_new_contact_cycle",
    "stagger_sleep",
]
