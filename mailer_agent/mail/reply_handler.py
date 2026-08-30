"""
Turns a parsed InboundEmail into: a recorded Message row, an intent
classification, a contact-status update, and (if configured) an
automatic reply.

Correlation order:
  1. Match In-Reply-To / References against a stored Message.message_id_header
     (the reliable path -- works regardless of subject-line edits).
  2. Fall back to matching the sender's address against an active Contact
     in the system (handles clients that strip threading headers).
  3. No match -> logged and skipped; nothing silently guessed.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import or_
from sqlalchemy.orm import Session

from mailer_agent.config import get_settings
from mailer_agent.llm.agent import classify_reply, draft_message
from mailer_agent.mail.imap_reader import InboundEmail
from mailer_agent.mail.sender import send_email
from mailer_agent.memory.store import build_conversation_context, maybe_summarize_older_messages
from mailer_agent.models import Contact, ContactStatus, Message, MessageDirection, MessageStatus, MessageType, SuppressionEntry

logger = logging.getLogger("mailer_agent.mail.reply_handler")
settings = get_settings()

# Intents that should stop the sequence outright.
TERMINAL_NEGATIVE_INTENTS = {"unsubscribe", "not_interested"}
# Intents that warrant a reply but should never be auto-sent, even if
# auto_reply_enabled=True, because getting it wrong is costly.
ALWAYS_REQUIRE_APPROVAL_INTENTS = {"objection"}


def _find_contact_by_message_id(db: Session, email_in: InboundEmail) -> Contact | None:
    candidate_ids = set()
    if email_in.in_reply_to:
        candidate_ids.add(email_in.in_reply_to)
    candidate_ids.update(email_in.references)
    if not candidate_ids:
        return None

    msg = (
        db.query(Message)
        .filter(Message.message_id_header.in_(candidate_ids))
        .order_by(Message.created_at.desc())
        .first()
    )
    return msg.contact if msg else None


def _find_contact_by_email(db: Session, email_in: InboundEmail) -> Contact | None:
    return (
        db.query(Contact)
        .filter(
            Contact.email == email_in.from_email,
            Contact.status != ContactStatus.SUPPRESSED.value,
        )
        .order_by(Contact.updated_at.desc())
        .first()
    )


def process_inbound_email(db: Session, email_in: InboundEmail) -> dict:
    contact = _find_contact_by_message_id(db, email_in) or _find_contact_by_email(db, email_in)
    if not contact:
        logger.info("No matching contact for inbound email from %s -- skipped", email_in.from_email)
        return {"matched": False}

    campaign = contact.campaign

    inbound_msg = Message(
        contact_id=contact.id,
        direction=MessageDirection.INBOUND.value,
        message_type=None,
        subject=email_in.subject,
        body=email_in.body_text,
        status=MessageStatus.RECEIVED.value,
        message_id_header=email_in.message_id,
        in_reply_to_header=email_in.in_reply_to,
    )
    db.add(inbound_msg)
    db.flush()

    context_transcript = build_conversation_context(db, contact)
    classification = classify_reply(
        campaign=campaign, contact=contact, inbound_body=email_in.body_text, context_transcript=context_transcript
    )
    inbound_msg.detected_intent = classification.intent
    inbound_msg.intent_confidence = classification.confidence

    result = {"matched": True, "contact_id": contact.id, "intent": classification.intent}

    if classification.intent == "unsubscribe":
        if not db.query(SuppressionEntry).filter_by(email=contact.email).first():
            db.add(SuppressionEntry(email=contact.email, reason="unsubscribed"))
        contact.status = ContactStatus.SUPPRESSED.value
        contact.next_action_at = None
        db.add(contact)
        result["action"] = "suppressed"

    elif classification.intent == "not_interested":
        contact.status = ContactStatus.CLOSED_LOST.value
        contact.next_action_at = None
        db.add(contact)
        result["action"] = "closed_lost"

    elif classification.intent == "out_of_office":
        # No status change, no auto-reply -- just wait for the real reply.
        result["action"] = "ignored_oos"

    else:
        # interested / question / objection / neutral -> needs a reply.
        contact.status = ContactStatus.REPLIED.value
        contact.next_action_at = None  # scheduler shouldn't follow-up while awaiting our own reply
        db.add(contact)

        action_type = "closing" if classification.intent == "interested" else "reply"
        draft = draft_message(
            campaign=campaign,
            contact=contact,
            action_type=action_type,
            context_transcript=build_conversation_context(db, contact),  # refresh with the new inbound msg
        )

        can_auto_send = (
            settings.auto_reply_enabled
            and classification.intent not in ALWAYS_REQUIRE_APPROVAL_INTENTS
            and classification.confidence >= 0.6
        )

        reply_msg = Message(
            contact_id=contact.id,
            direction=MessageDirection.OUTBOUND.value,
            message_type=MessageType.REPLY.value if action_type == "reply" else MessageType.CLOSING.value,
            subject=draft.subject or (f"Re: {email_in.subject}" if email_in.subject else None),
            body=draft.body,
            status=MessageStatus.DRAFT.value,
            in_reply_to_header=email_in.message_id,
        )

        if can_auto_send:
            reply_subject = draft.subject or (f"Re: {email_in.subject}" if email_in.subject else "Re: our conversation")
            send_result = send_email(
                to_email=contact.email,
                from_email=campaign.sender_email,
                from_name=campaign.sender_name,
                subject=reply_subject,
                body_text=draft.body,
                reply_to=campaign.reply_to_email,
                in_reply_to_header=email_in.message_id,
            )
            reply_msg.status = MessageStatus.SENT.value if send_result.success else MessageStatus.FAILED.value
            reply_msg.message_id_header = send_result.message_id
            reply_msg.error_message = send_result.error
            reply_msg.subject = reply_subject
            result["action"] = "auto_replied" if send_result.success else "auto_reply_failed"
        else:
            result["action"] = "reply_drafted_awaiting_approval"

        db.add(reply_msg)

    db.flush()
    maybe_summarize_older_messages(db, contact)
    return result
