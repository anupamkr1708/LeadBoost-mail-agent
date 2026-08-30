"""
The actual "who gets messaged, and when" logic.

Cadence is entirely data-driven: each Campaign carries its own
`follow_up_days` list (e.g. [3, 7, 14], user-supplied at campaign
creation, editable per campaign) -- this module never hardcodes a wait
period. It only ever reads that list and `contact.follow_up_index` to
work out "how many days from now until the next touch".
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from mailer_agent.config import get_settings
from mailer_agent.llm.agent import draft_message
from mailer_agent.mail.imap_reader import as_reply_subject
from mailer_agent.mail.sender import send_email
from mailer_agent.memory.store import build_conversation_context, maybe_summarize_older_messages
from mailer_agent.models import (
    Campaign,
    Contact,
    ContactStatus,
    Message,
    MessageDirection,
    MessageStatus,
    MessageType,
    SuppressionEntry,
)

logger = logging.getLogger("mailer_agent.followup.engine")
settings = get_settings()


def is_suppressed(db: Session, email_addr: str) -> bool:
    return db.query(SuppressionEntry).filter_by(email=email_addr).first() is not None


def _most_recent_message_id(contact: Contact) -> str | None:
    for m in reversed(contact.messages):
        if m.message_id_header:
            return m.message_id_header
    return None


def _compute_next_action_at(campaign: Campaign, follow_up_index: int) -> tuple[datetime | None, bool]:
    """
    Returns (next_action_at, sequence_exhausted). follow_up_index is the
    number of follow-ups already sent (0 right after the initial email).
    """
    days_list: list[int] = campaign.follow_up_days or []
    if follow_up_index >= len(days_list):
        return None, True
    return datetime.utcnow() + timedelta(days=days_list[follow_up_index]), False


def send_initial_outreach(db: Session, contact: Contact) -> dict:
    campaign = contact.campaign

    if is_suppressed(db, contact.email):
        contact.status = ContactStatus.SUPPRESSED.value
        db.add(contact)
        db.flush()
        return {"contact_id": contact.id, "action": "skipped_suppressed"}

    context_transcript = build_conversation_context(db, contact)
    draft = draft_message(
        campaign=campaign, contact=contact, action_type="initial_outreach", context_transcript=context_transcript
    )

    send_result = send_email(
        to_email=contact.email,
        from_email=campaign.sender_email,
        from_name=campaign.sender_name,
        subject=draft.subject or f"{campaign.sender_org} <> {contact.company or contact.name}",
        body_text=draft.body,
        reply_to=campaign.reply_to_email,
    )

    msg = Message(
        contact_id=contact.id,
        direction=MessageDirection.OUTBOUND.value,
        message_type=MessageType.INITIAL.value,
        subject=draft.subject,
        body=draft.body,
        status=MessageStatus.SENT.value if send_result.success else MessageStatus.FAILED.value,
        message_id_header=send_result.message_id,
        error_message=send_result.error,
    )
    db.add(msg)

    if send_result.success:
        contact.status = ContactStatus.ACTIVE.value
        next_at, exhausted = _compute_next_action_at(campaign, follow_up_index=0)
        contact.next_action_at = None if exhausted else next_at
    # on failure, leave status as-is so the next cycle retries

    db.add(contact)
    db.flush()
    maybe_summarize_older_messages(db, contact)

    return {"contact_id": contact.id, "action": "sent" if send_result.success else "failed", "source": draft.source}


def send_followup_if_due(db: Session, contact: Contact) -> dict | None:
    if contact.status != ContactStatus.ACTIVE.value:
        return None
    if not contact.next_action_at or contact.next_action_at > datetime.utcnow():
        return None
    if is_suppressed(db, contact.email):
        contact.status = ContactStatus.SUPPRESSED.value
        contact.next_action_at = None
        db.add(contact)
        db.flush()
        return {"contact_id": contact.id, "action": "skipped_suppressed"}

    campaign = contact.campaign
    days_waited = None
    last_outbound = next((m for m in reversed(contact.messages) if m.direction == MessageDirection.OUTBOUND.value), None)
    if last_outbound and last_outbound.created_at:
        days_waited = (datetime.utcnow() - last_outbound.created_at).days

    context_transcript = build_conversation_context(db, contact)
    draft = draft_message(
        campaign=campaign,
        contact=contact,
        action_type="follow_up",
        context_transcript=context_transcript,
        days_waited=days_waited,
    )

    prior_msg_id = _most_recent_message_id(contact)
    subject = draft.subject or (
        as_reply_subject(contact.messages[0].subject) if contact.messages and contact.messages[0].subject else None
    )

    send_result = send_email(
        to_email=contact.email,
        from_email=campaign.sender_email,
        from_name=campaign.sender_name,
        subject=subject or f"Following up -- {campaign.sender_org}",
        body_text=draft.body,
        reply_to=campaign.reply_to_email,
        in_reply_to_header=prior_msg_id,
    )

    msg = Message(
        contact_id=contact.id,
        direction=MessageDirection.OUTBOUND.value,
        message_type=MessageType.FOLLOW_UP.value,
        subject=subject,
        body=draft.body,
        status=MessageStatus.SENT.value if send_result.success else MessageStatus.FAILED.value,
        message_id_header=send_result.message_id,
        in_reply_to_header=prior_msg_id,
        error_message=send_result.error,
    )
    db.add(msg)

    if send_result.success:
        contact.follow_up_index += 1
        next_at, exhausted = _compute_next_action_at(campaign, contact.follow_up_index)
        if exhausted:
            contact.status = ContactStatus.SEQUENCE_COMPLETE.value
            contact.next_action_at = None
        else:
            contact.next_action_at = next_at
    db.add(contact)
    db.flush()
    maybe_summarize_older_messages(db, contact)

    return {"contact_id": contact.id, "action": "sent" if send_result.success else "failed", "source": draft.source}


def run_new_contact_cycle(db: Session, limit: int | None = None) -> list[dict]:
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


def run_followup_cycle(db: Session, limit: int | None = None) -> list[dict]:
    limit = limit or settings.max_sends_per_cycle
    due_contacts = (
        db.query(Contact)
        .join(Campaign)
        .filter(
            Contact.status == ContactStatus.ACTIVE.value,
            Contact.next_action_at.isnot(None),
            Contact.next_action_at <= datetime.utcnow(),
            Campaign.is_active.is_(True),
        )
        .limit(limit)
        .all()
    )
    results = []
    for i, c in enumerate(due_contacts):
        if i > 0:
            stagger_sleep()
        r = send_followup_if_due(db, c)
        if r:
            results.append(r)
    db.commit()
    return results


def stagger_sleep() -> None:
    """Real prospects landing seconds apart from the same sender reads
    as automated bulk activity. Called between consecutive sends in a
    batch -- never before the first one, since a single send shouldn't
    be delayed for no reason."""
    delay = settings.send_delay_seconds + random.uniform(0, settings.send_jitter_seconds)
    time.sleep(delay)
