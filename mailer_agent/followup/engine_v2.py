"""
Integrated follow-up engine with conversation awareness.

Combines semantic intelligence, state machine, and conversation-aware scheduling.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from mailer_agent.config import get_settings
from mailer_agent.followup.conversation_aware import FollowUpScheduler
from mailer_agent.llm.agent import draft_message
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
)
from mailer_agent.state_machine import (
    StateTransitionEvent,
    can_send_followup,
    transition_contact_state,
)

logger = logging.getLogger("mailer_agent.followup.engine_v2")
settings = get_settings()


class IntegratedFollowUpEngine:
    """
    Integrated follow-up engine with semantic intelligence.
    
    Uses conversation-aware scheduling instead of blind timing.
    """
    
    def __init__(self):
        self.scheduler = FollowUpScheduler()
    
    def send_initial_outreach(self, db: Session, contact: Contact) -> dict:
        """
        Send initial outreach with integrated tracking.
        """
        from mailer_agent.followup.engine import is_suppressed
        
        campaign = contact.campaign
        
        # Suppression check
        if is_suppressed(db, contact.email):
            contact.status = ContactStatus.SUPPRESSED.value
            db.add(contact)
            db.flush()
            return {"contact_id": contact.id, "action": "skipped_suppressed"}
        
        # Build context and draft
        context_transcript = build_conversation_context(db, contact)
        draft = draft_message(
            campaign=campaign,
            contact=contact,
            action_type="initial_outreach",
            context_transcript=context_transcript
        )
        
        # Send email
        send_result = send_email(
            to_email=contact.email,
            from_email=campaign.sender_email,
            from_name=campaign.sender_name,
            subject=draft.subject or f"{campaign.sender_org} <> {contact.company or contact.name}",
            body_text=draft.body,
            reply_to=campaign.reply_to_email,
        )
        
        # Persist message
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
            # Transition state
            transition_contact_state(
                contact,
                StateTransitionEvent.INITIAL_SENT,
                reason="Initial outreach sent successfully"
            )
            
            # Update timestamps
            contact.last_outbound_at = datetime.utcnow()
            
            # Schedule next follow-up using conversation-aware logic
            contact.next_action_at = self.scheduler.compute_next_followup(
                contact, campaign, semantic_intent=None
            )
        
        db.add(contact)
        db.flush()
        maybe_summarize_older_messages(db, contact)
        
        return {
            "contact_id": contact.id,
            "action": "sent" if send_result.success else "failed",
            "source": draft.source,
            "next_action_at": contact.next_action_at.isoformat() if contact.next_action_at else None
        }
    
    def send_followup_if_due(self, db: Session, contact: Contact) -> Optional[dict]:
        """
        Send follow-up if due, using conversation-aware checks.
        """
        from mailer_agent.followup.engine import is_suppressed
        from mailer_agent.mail.imap_reader import as_reply_subject
        
        # Check if should send
        should_send, reason = self.scheduler.should_send_followup_now(contact)
        if not should_send:
            logger.debug(f"Contact {contact.id} follow-up not sending: {reason}")
            return None
        
        # Double-check state machine
        if not can_send_followup(contact):
            logger.warning(
                f"Contact {contact.id} passed scheduler check but failed state machine check"
            )
            return None
        
        # Suppression check
        if is_suppressed(db, contact.email):
            transition_contact_state(
                contact,
                StateTransitionEvent.UNSUBSCRIBED,
                reason="Found in suppression list during follow-up"
            )
            contact.next_action_at = None
            db.add(contact)
            db.flush()
            return {"contact_id": contact.id, "action": "skipped_suppressed"}
        
        campaign = contact.campaign
        
        # Calculate days waited
        days_waited = None
        if contact.last_outbound_at:
            days_waited = (datetime.utcnow() - contact.last_outbound_at).days
        
        # Get conversation-aware hints
        message_hints = self.scheduler.get_followup_message_hint(contact)
        
        # Build context and draft
        context_transcript = build_conversation_context(db, contact)
        draft = draft_message(
            campaign=campaign,
            contact=contact,
            action_type="follow_up",
            context_transcript=context_transcript,
            days_waited=days_waited
        )
        
        # Prepare subject
        prior_msg_id = self._most_recent_message_id(contact)
        subject = draft.subject or (
            as_reply_subject(contact.messages[0].subject)
            if contact.messages and contact.messages[0].subject
            else None
        )
        
        # Send email
        send_result = send_email(
            to_email=contact.email,
            from_email=campaign.sender_email,
            from_name=campaign.sender_name,
            subject=subject or f"Following up -- {campaign.sender_org}",
            body_text=draft.body,
            reply_to=campaign.reply_to_email,
            in_reply_to_header=prior_msg_id,
        )
        
        # Persist message
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
            # Transition state
            transition_contact_state(
                contact,
                StateTransitionEvent.FOLLOWUP_SENT,
                reason=f"Follow-up #{contact.follow_up_index + 1} sent"
            )
            
            # Update timestamps
            contact.last_outbound_at = datetime.utcnow()
            contact.follow_up_index += 1
            
            # Schedule next follow-up using conversation-aware logic
            next_at = self.scheduler.compute_next_followup(contact, campaign)
            
            if not next_at:
                # Sequence exhausted
                transition_contact_state(
                    contact,
                    StateTransitionEvent.SEQUENCE_EXHAUSTED,
                    reason="Follow-up cadence exhausted"
                )
            
            contact.next_action_at = next_at
        
        db.add(contact)
        db.flush()
        maybe_summarize_older_messages(db, contact)
        
        return {
            "contact_id": contact.id,
            "action": "sent" if send_result.success else "failed",
            "source": draft.source,
            "follow_up_index": contact.follow_up_index,
            "next_action_at": contact.next_action_at.isoformat() if contact.next_action_at else None
        }
    
    def _most_recent_message_id(self, contact: Contact) -> Optional[str]:
        """Get most recent outbound Message-ID for threading."""
        for m in reversed(contact.messages):
            if m.message_id_header:
                return m.message_id_header
        return None
    
    def run_followup_cycle(self, db: Session, limit: Optional[int] = None) -> list[dict]:
        """
        Run follow-up cycle with conversation-aware checks.
        """
        limit = limit or settings.max_sends_per_cycle
        
        # Find due contacts
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
        for i, contact in enumerate(due_contacts):
            if i > 0:
                from mailer_agent.followup.engine import stagger_sleep
                stagger_sleep()
            
            result = self.send_followup_if_due(db, contact)
            if result:
                results.append(result)
        
        db.commit()
        
        logger.info(f"Follow-up cycle completed: {len(results)} sent")
        return results


# Global instance
integrated_engine = IntegratedFollowUpEngine()
