"""
Updated scheduler using integrated components.

Uses the integrated engine with semantic intelligence and conversation-aware scheduling.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from mailer_agent.config import get_settings
from mailer_agent.db import session_scope
from mailer_agent.integration import mailer_agent
from mailer_agent.mail.imap_reader import fetch_unseen_replies

logger = logging.getLogger("mailer_agent.scheduler_v2")
settings = get_settings()

_scheduler: BackgroundScheduler | None = None


def poll_replies_job() -> None:
    """
    Poll for replies and process with full semantic intelligence.
    """
    emails = fetch_unseen_replies()
    if not emails:
        return
    
    logger.info(f"Fetched {len(emails)} unseen email(s)")
    
    with session_scope() as db:
        for email_in in emails:
            try:
                result = mailer_agent.process_reply(db, email_in)
                logger.info(f"Processed inbound from {email_in.from_email}: {result.get('action', 'unknown')}")
            except Exception as e:
                logger.exception(f"Failed to process inbound email from {email_in.from_email}: {e}")


def dispatch_followups_job() -> None:
    """
    Dispatch follow-ups using conversation-aware logic.
    """
    with session_scope() as db:
        results = mailer_agent.run_followup_cycle(db)
        if results:
            logger.info(f"Follow-up cycle: {len(results)} contacts processed")
            
            # Log breakdown
            sent = sum(1 for r in results if r.get('action') == 'sent')
            failed = sum(1 for r in results if r.get('action') == 'failed')
            skipped = len(results) - sent - failed
            
            logger.info(f"  Sent: {sent}, Failed: {failed}, Skipped: {skipped}")


def health_check_job() -> None:
    """
    Periodic health check and metrics logging.
    """
    try:
        metrics = mailer_agent.get_llm_metrics()
        logger.info(f"System health check - LLM metrics: {metrics}")
    except Exception as e:
        logger.error(f"Health check failed: {e}")


def start_scheduler() -> BackgroundScheduler:
    """
    Start the integrated scheduler.
    """
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    
    scheduler = BackgroundScheduler(timezone="UTC")
    
    # Reply polling
    scheduler.add_job(
        poll_replies_job,
        "interval",
        seconds=settings.imap_poll_seconds,
        id="poll_replies",
        max_instances=1  # Prevent overlapping runs
    )
    
    # Follow-up dispatch
    scheduler.add_job(
        dispatch_followups_job,
        "interval",
        seconds=settings.followup_poll_seconds,
        id="dispatch_followups",
        max_instances=1
    )
    
    # Health check (every 5 minutes)
    scheduler.add_job(
        health_check_job,
        "interval",
        seconds=300,
        id="health_check",
        max_instances=1
    )
    
    scheduler.start()
    _scheduler = scheduler
    
    logger.info(
        f"Integrated scheduler started:\n"
        f"  Reply poll: every {settings.imap_poll_seconds}s\n"
        f"  Follow-up dispatch: every {settings.followup_poll_seconds}s\n"
        f"  Health check: every 300s"
    )
    
    return scheduler


def stop_scheduler() -> None:
    """Stop the scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler stopped")
