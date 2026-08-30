"""
Background job scheduling.

Single-process APScheduler is intentional for this standalone service:
it's the right amount of infrastructure for "a script with an API in
front of it" per the brief. If this needs to run across multiple
worker processes/machines later, swap this module for Celery beat +
workers -- nothing in followup/engine.py or mail/reply_handler.py
would need to change, since they're plain functions that take a
Session and don't know they're being called from a scheduler.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from mailer_agent.config import get_settings
from mailer_agent.db import session_scope
from mailer_agent.followup.engine import run_followup_cycle, run_new_contact_cycle
from mailer_agent.mail.imap_reader import fetch_unseen_replies
from mailer_agent.mail.reply_handler import process_inbound_email

logger = logging.getLogger("mailer_agent.scheduler")
settings = get_settings()

_scheduler: BackgroundScheduler | None = None


def poll_replies_job() -> None:
    emails = fetch_unseen_replies()
    if not emails:
        return
    logger.info("Fetched %d unseen email(s)", len(emails))
    with session_scope() as db:
        for email_in in emails:
            try:
                result = process_inbound_email(db, email_in)
                logger.info("Processed inbound from %s: %s", email_in.from_email, result)
            except Exception as e:
                logger.exception("Failed to process inbound email from %s: %s", email_in.from_email, e)


def dispatch_new_contacts_job() -> None:
    with session_scope() as db:
        results = run_new_contact_cycle(db)
        if results:
            logger.info("Initial outreach cycle: %s", results)


def dispatch_followups_job() -> None:
    with session_scope() as db:
        results = run_followup_cycle(db)
        if results:
            logger.info("Follow-up cycle: %s", results)


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(poll_replies_job, "interval", seconds=settings.imap_poll_seconds, id="poll_replies")
    scheduler.add_job(dispatch_new_contacts_job, "interval", seconds=settings.followup_poll_seconds, id="dispatch_new")
    scheduler.add_job(dispatch_followups_job, "interval", seconds=settings.followup_poll_seconds, id="dispatch_followups")
    scheduler.start()
    _scheduler = scheduler
    logger.info(
        "Scheduler started (reply poll every %ss, dispatch every %ss)",
        settings.imap_poll_seconds, settings.followup_poll_seconds,
    )
    return scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
