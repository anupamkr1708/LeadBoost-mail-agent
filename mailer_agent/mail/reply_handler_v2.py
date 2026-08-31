"""
Enhanced reply handler with semantic intelligence.

Improvements over original:
1. Uses multi-dimensional semantic classification
2. Explicit failure handling (provider failures ≠ semantic neutrality)
3. State machine-driven transitions
4. Message deduplication
5. Proper timestamp tracking
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import or_
from sqlalchemy.orm import Session

from mailer_agent.config import get_settings
from mailer_agent.llm.agent import draft_message
from mailer_agent.mail.imap_reader import InboundEmail, as_reply_subject
from mailer_agent.mail.sender import send_email
from mailer_agent.memory.store import build_conversation_context, maybe_summarize_older_messages
from mailer_agent.models import (
    Contact,
    ContactStatus,
    Message,
    MessageDirection,
    MessageStatus,
    MessageType,
    SuppressionEntry,
)
from mailer_agent.semantic.classifier import classify_prospect_reply
from mailer_agent.semantic_models import IntentType
from mailer_agent.state_machine import (
    StateTransitionEvent,
    infer_event_from_semantic_intent,
    transition_contact_state,
)
from mailer_agent.utils.datetime_utils import utcnow

logger = logging.getLogger("mailer_agent.mail.reply_handler_v2")
settings = get_settings()


# Intents that require human approval (never auto-send)
ALWAYS_REQUIRE_APPROVAL_INTENTS = {
    IntentType.OBJECTION,
    IntentType.PRICING_REQUEST,  # Don't auto-respond to pricing without verified data
    IntentType.NOT_INTERESTED,  # Let human confirm before writing off
}


def process_inbound_email_v2(db: Session, email_in: InboundEmail) -> dict:
    """
    Enhanced inbound email processing with semantic intelligence.
    
    Returns dict with processing result.
    """
    # Step 1: Deduplicate by Message-ID
    if email_in.message_id:
        existing = db.query(Message).filter(
            Message.message_id_header == email_in.message_id,
            Message.direction == MessageDirection.INBOUND.value
        ).first()
        
        if existing:
            logger.info(
                f"Duplicate inbound email detected (Message-ID: {email_in.message_id}), skipping"
            )
            return {
                "matched": True,
                "contact_id": existing.contact_id,
                "action": "skipped_duplicate",
                "message_id": existing.id
            }
    
    # Step 2: Correlate to contact
    contact = _find_contact_by_message_id(db, email_in) or _find_contact_by_email(db, email_in)
    
    if not contact:
        logger.info(f"No matching contact for inbound email from {email_in.from_email} -- skipped")
        return {"matched": False, "from_email": email_in.from_email}
    
    campaign = contact.campaign
    
    # Step 3: Persist inbound message
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
    
    # Update contact timestamps
    contact.last_reply_at = utcnow()
    
    # Step 4: Semantic classification
    context_transcript = build_conversation_context(db, contact)
    
    classification_result = classify_prospect_reply(
        campaign=campaign,
        contact=contact,
        inbound_body=email_in.body_text,
        conversation_context=context_transcript
    )
    
    # Store classification result in message
    inbound_msg.classification_success = classification_result.success
    
    if classification_result.success and classification_result.semantic_intent:
        intent = classification_result.semantic_intent
        
        # Serialize semantic intent to JSON
        inbound_msg.semantic_analysis = json.dumps({
            "intents": [i.value for i in intent.intents],
            "sentiment": intent.sentiment.value,
            "buying_stage": intent.buying_stage.value,
            "urgency": intent.urgency.value,
            "requested_timing": intent.requested_timing,
            "has_pricing_question": intent.has_pricing_question,
            "has_budget_signal": intent.has_budget_signal,
            "requested_information": intent.requested_information,
            "objections_raised": intent.objections_raised,
            "questions_asked": intent.questions_asked,
            "confidence": intent.confidence,
            "reasoning": intent.reasoning,
            "requires_human_review": intent.requires_human_review,
        })
        
        # Backward compatibility: set detected_intent to primary intent
        inbound_msg.detected_intent = intent.intents[0].value if intent.intents else "neutral"
        inbound_msg.intent_confidence = intent.confidence
        
        # Update contact buying stage
        contact.buying_stage = intent.buying_stage.value
        
    else:
        # Classification failed
        inbound_msg.classification_failure_reason = (
            classification_result.failure_reason.value if classification_result.failure_reason else "unknown"
        )
        inbound_msg.detected_intent = "unknown"
        inbound_msg.intent_confidence = 0.0
    
    result = {
        "matched": True,
        "contact_id": contact.id,
        "classification_success": classification_result.success,
    }
    
    # Step 5: Handle based on classification
    if not classification_result.success:
        # Classification failed - require human review
        logger.warning(
            f"Classification failed for contact {contact.id}: "
            f"{classification_result.failure_reason} - routing to human review"
        )
        
        transition_contact_state(
            contact,
            StateTransitionEvent.NEEDS_HUMAN,
            reason=f"Classification failed: {classification_result.failure_reason}"
        )
        contact.next_action_at = None
        result["action"] = "needs_review_classification_failed"
        
    elif not classification_result.semantic_intent:
        # Should not happen if success=True, but handle gracefully
        logger.error(f"Classification success but no semantic_intent for contact {contact.id}")
        transition_contact_state(contact, StateTransitionEvent.NEEDS_HUMAN, reason="Missing semantic intent")
        result["action"] = "needs_review_missing_intent"
        
    else:
        # Successful classification
        intent = classification_result.semantic_intent
        result["primary_intent"] = intent.intents[0].value if intent.intents else "neutral"
        result["confidence"] = intent.confidence
        
        # Infer state transition event
        event = infer_event_from_semantic_intent(intent)
        
        # Handle terminal intents
        if IntentType.UNSUBSCRIBE in intent.intents:
            _handle_unsubscribe(db, contact, result)
            
        elif IntentType.NOT_INTERESTED in intent.intents:
            transition_contact_state(contact, StateTransitionEvent.NOT_INTERESTED)
            contact.next_action_at = None
            result["action"] = "closed_lost"
            
        elif IntentType.OUT_OF_OFFICE in intent.intents:
            # No state change, no action - just wait
            result["action"] = "ignored_oos"
            
        else:
            # Regular reply - transition state
            transition_contact_state(contact, event, reason=f"Semantic: {intent.reasoning[:100]}")
            
            # Reschedule follow-up based on conversation context
            from mailer_agent.followup.conversation_aware import FollowUpScheduler
            scheduler = FollowUpScheduler()
            contact.next_action_at = scheduler.reschedule_after_reply(
                contact, campaign, intent
            )
            
            # Draft reply
            _draft_and_maybe_send_reply(db, contact, campaign, email_in, intent, result)
    
    db.add(contact)
    db.add(inbound_msg)
    db.flush()
    
    # Step 6: Update conversation memory
    maybe_summarize_older_messages(db, contact)
    
    return result


def _find_contact_by_message_id(db: Session, email_in: InboundEmail) -> Contact | None:
    """Find contact by threading headers (reliable method)."""
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
    """Find contact by email address (fallback method)."""
    return (
        db.query(Contact)
        .filter(
            Contact.email == email_in.from_email,
            Contact.status != ContactStatus.SUPPRESSED.value,
        )
        .order_by(Contact.updated_at.desc())
        .first()
    )


def _handle_unsubscribe(db: Session, contact: Contact, result: dict):
    """Handle unsubscribe request."""
    # Add to suppression list if not already there
    if not db.query(SuppressionEntry).filter_by(email=contact.email).first():
        db.add(SuppressionEntry(email=contact.email, reason="unsubscribed"))
    
    # Transition to suppressed
    transition_contact_state(contact, StateTransitionEvent.UNSUBSCRIBED)
    contact.next_action_at = None
    
    result["action"] = "suppressed"


def _draft_and_maybe_send_reply(
    db: Session,
    contact: Contact,
    campaign,
    email_in: InboundEmail,
    intent,
    result: dict
):
    """Draft reply and auto-send if conditions are met."""
    
    # Determine action type based on semantic analysis
    if IntentType.POSITIVE_INTEREST in intent.intents and intent.confidence >= 0.7:
        action_type = "closing"
    else:
        action_type = "reply"
    
    # Build context for drafting
    context_transcript = build_conversation_context(db, contact)
    
    # Draft message
    draft = draft_message(
        campaign=campaign,
        contact=contact,
        action_type=action_type,
        context_transcript=context_transcript
    )

    # Grounding gate: if unsupported claims found, force draft status
    # regardless of auto-reply settings.
    grounding_blocked = draft.grounding and not draft.grounding.is_safe_to_send
    if grounding_blocked:
        logger.info(
            "Contact %s reply held for grounding review: %s",
            contact.id, draft.grounding.validation_notes,
        )
    
    # Determine if we can auto-send
    can_auto_send = _can_auto_send_reply(intent) and not grounding_blocked
    
    # Create reply message
    reply_msg = Message(
        contact_id=contact.id,
        direction=MessageDirection.OUTBOUND.value,
        message_type=MessageType.REPLY.value if action_type == "reply" else MessageType.CLOSING.value,
        subject=draft.subject or as_reply_subject(email_in.subject),
        body=draft.body,
        status=MessageStatus.DRAFT.value,
        in_reply_to_header=email_in.message_id,
    )
    
    if can_auto_send:
        # Auto-send
        reply_subject = draft.subject or as_reply_subject(email_in.subject)
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
        
        if send_result.success:
            contact.last_outbound_at = utcnow()
        
        result["action"] = "auto_replied" if send_result.success else "auto_reply_failed"
        result["message_id"] = reply_msg.id
        
    else:
        # Requires human approval
        result["action"] = "reply_drafted_awaiting_approval"
        result["message_id"] = reply_msg.id
        if grounding_blocked and draft.grounding:
            result["approval_reason"] = (
                f"Grounding: {draft.grounding.validation_notes}"
            )
        else:
            result["approval_reason"] = _get_approval_reason(intent)
    
    db.add(reply_msg)


def _can_auto_send_reply(intent) -> bool:
    """Determine if reply can be auto-sent."""
    
    if not settings.auto_reply_enabled:
        return False
    
    # Check confidence threshold
    if intent.confidence < 0.6:
        return False
    
    # Check if requires human review
    if intent.requires_human_review:
        return False
    
    # Check intent types
    for intent_type in intent.intents:
        if intent_type in ALWAYS_REQUIRE_APPROVAL_INTENTS:
            return False
    
    return True


def _get_approval_reason(intent) -> str:
    """Get human-readable reason why approval is required."""
    
    if not settings.auto_reply_enabled:
        return "AUTO_REPLY_ENABLED is false"
    
    if intent.confidence < 0.6:
        return f"Low confidence ({intent.confidence:.2f})"
    
    if intent.requires_human_review:
        return intent.human_review_reason or "Semantic analysis requires human review"
    
    for intent_type in intent.intents:
        if intent_type in ALWAYS_REQUIRE_APPROVAL_INTENTS:
            return f"Intent type requires approval: {intent_type.value}"
    
    return "Unknown reason"
