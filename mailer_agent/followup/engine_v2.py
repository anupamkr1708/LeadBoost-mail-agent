"""
Integrated follow-up engine with conversation awareness.

Combines semantic intelligence, state machine, and conversation-aware scheduling.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from mailer_agent.config import get_settings
from mailer_agent.followup.conversation_aware import FollowUpScheduler
from mailer_agent.llm.agent import draft_message
from mailer_agent.mail.sender import SendOutcome, send_email
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
from mailer_agent.utils.datetime_utils import utcnow

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
            # Use state machine for transition instead of direct assignment
            transition_contact_state(contact, StateTransitionEvent.UNSUBSCRIBED)
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

        # Grounding gate: hold for human review if unsupported claims found
        if draft.grounding and not draft.grounding.is_safe_to_send:
            logger.info(
                "Contact %s initial outreach held for grounding review: %s",
                contact.id, draft.grounding.validation_notes,
            )
            msg = Message(
                contact_id=contact.id,
                direction=MessageDirection.OUTBOUND.value,
                message_type=MessageType.INITIAL.value,
                subject=draft.subject or f"{campaign.sender_org} <> {contact.company or contact.name}",
                body=draft.body,
                status=MessageStatus.DRAFT.value,
            )
            db.add(msg)
            db.add(contact)
            db.flush()
            return {
                "contact_id": contact.id,
                "action": "held_grounding_review",
                "grounding_notes": draft.grounding.validation_notes,
                "source": draft.source,
            }
        
        # Send email
        send_result = send_email(
            to_email=contact.email,
            from_email=campaign.sender_email,
            from_name=campaign.sender_name,
            subject=draft.subject or f"{campaign.sender_org} <> {contact.company or contact.name}",
            body_text=draft.body,
            reply_to=campaign.reply_to_email,
        )
        
        # Persist message. send_result.outcome is authoritative (SENT /
        # FAILED / UNKNOWN) -- UNKNOWN means the SMTP call raised during
        # or after transmission and delivery status is genuinely
        # ambiguous, so it must NOT collapse into "failed" (which would
        # imply it's safe to blindly retry and risk a duplicate send).
        msg = Message(
            contact_id=contact.id,
            direction=MessageDirection.OUTBOUND.value,
            message_type=MessageType.INITIAL.value,
            subject=draft.subject,
            body=draft.body,
            status=send_result.outcome.value,
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
            contact.last_outbound_at = utcnow()
            
            # Schedule next follow-up using conversation-aware logic
            contact.next_action_at = self.scheduler.compute_next_followup(
                contact, campaign, semantic_intent=None
            )
        elif send_result.outcome == SendOutcome.UNKNOWN:
            # Delivery status is genuinely ambiguous -- do NOT leave
            # next_action_at in the past (which would make the next
            # scheduler pass blindly retry and risk a duplicate send).
            # Route to human review instead.
            transition_contact_state(
                contact,
                StateTransitionEvent.NEEDS_HUMAN,
                reason=f"Initial outreach send outcome unknown: {send_result.error}",
            )
            contact.next_action_at = None
        # else: outcome == FAILED -- contact stays NEW with next_action_at
        # already due, so the next dispatch cycle retries safely (nothing
        # was ever transmitted).
        
        db.add(contact)
        db.flush()
        maybe_summarize_older_messages(db, contact)
        
        return {
            "contact_id": contact.id,
            "action": send_result.outcome.value,
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
            days_waited = (utcnow() - contact.last_outbound_at).days
        
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

        # Grounding gate: hold for human review if unsupported claims found
        if draft.grounding and not draft.grounding.is_safe_to_send:
            logger.info(
                "Contact %s follow-up held for grounding review: %s",
                contact.id, draft.grounding.validation_notes,
            )
            prior_msg_id_g = self._most_recent_message_id(contact)
            subject_g = draft.subject or f"Following up -- {campaign.sender_org}"
            msg = Message(
                contact_id=contact.id,
                direction=MessageDirection.OUTBOUND.value,
                message_type=MessageType.FOLLOW_UP.value,
                subject=subject_g,
                body=draft.body,
                status=MessageStatus.DRAFT.value,
                in_reply_to_header=prior_msg_id_g,
            )
            db.add(msg)
            db.add(contact)
            db.flush()
            return {
                "contact_id": contact.id,
                "action": "held_grounding_review",
                "grounding_notes": draft.grounding.validation_notes,
                "source": draft.source,
            }
        
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
        
        # Persist message. Status mirrors send_result.outcome (sent /
        # failed / unknown) directly -- see mail/sender.py for why
        # "unknown" must never collapse into "failed".
        msg = Message(
            contact_id=contact.id,
            direction=MessageDirection.OUTBOUND.value,
            message_type=MessageType.FOLLOW_UP.value,
            subject=subject,
            body=draft.body,
            status=send_result.outcome.value,
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
            contact.last_outbound_at = utcnow()
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
        elif send_result.outcome == SendOutcome.UNKNOWN:
            # Ambiguous delivery -- escalate to a human instead of
            # leaving this contact eligible for an automatic retry that
            # could duplicate a message that actually went out.
            transition_contact_state(
                contact,
                StateTransitionEvent.NEEDS_HUMAN,
                reason=f"Follow-up send outcome unknown: {send_result.error}",
            )
            contact.next_action_at = None
        # else: FAILED -- next_action_at is left as-is (already due), so
        # the next cycle retries; nothing was ever transmitted.
        
        db.add(contact)
        db.flush()
        maybe_summarize_older_messages(db, contact)
        
        return {
            "contact_id": contact.id,
            "action": send_result.outcome.value,
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
    
    def run_followup_cycle(
        self,
        db: Session,
        limit: Optional[int] = None,
        worker_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Run follow-up cycle with conversation-aware checks and safe work claiming.

        Work claiming (Phase 8)
        -----------------------
        Rather than a plain SELECT that every concurrent worker would see
        simultaneously, this method atomically claims each contact row using
        a database-level lock before processing it.

        PostgreSQL (production): uses ``SELECT … FOR UPDATE SKIP LOCKED``
          -- competing workers skip already-locked rows instead of blocking,
          so two workers never process the same contact simultaneously.

        SQLite (dev/tests): uses an optimistic UPDATE with a WHERE guard;
          SQLite serialises writes anyway so races don't occur in practice.

        Lease expiry / recovery
        -----------------------
        Each claim records ``(claimed_by, claimed_at)`` on the Contact row.
        If the worker crashes before releasing a claim, any subsequent worker
        cycle will reclaim the row once ``claimed_at`` is older than
        ``CLAIM_LEASE_SECONDS`` (default 300 s).  Stale leases from a crashed
        previous process are also swept on each call via
        ``recover_expired_claims()``.
        """
        from mailer_agent.followup.work_claiming import (
            claim_due_contacts,
            make_worker_id,
            recover_expired_claims,
            release_claim,
        )
        from mailer_agent.followup.engine import stagger_sleep

        effective_limit = limit or settings.max_sends_per_cycle
        effective_worker_id = worker_id or make_worker_id()

        # --- Startup sweep: recover any leases left by a crashed worker ----
        recovered = recover_expired_claims(db)
        if recovered:
            logger.warning(
                "Recovered %d expired claim(s) at start of follow-up cycle", recovered
            )

        # --- Atomically claim due contacts ---------------------------------
        due_contacts = claim_due_contacts(
            db,
            worker_id=effective_worker_id,
            status=ContactStatus.ACTIVE.value,
            limit=effective_limit,
        )

        # Reload full objects with campaign relationship so joins are present.
        # claim_due_contacts already returns ORM objects but we refetch to
        # ensure the Campaign join is eagerly available.
        contact_ids = [c.id for c in due_contacts]
        if not contact_ids:
            return []

        due_contacts = (
            db.query(Contact)
            .join(Campaign)
            .filter(
                Contact.id.in_(contact_ids),
                Campaign.is_active.is_(True),
            )
            .all()
        )

        # --- Process each claimed contact ----------------------------------
        results = []
        for i, contact in enumerate(due_contacts):
            if i > 0:
                stagger_sleep()

            try:
                result = self.send_followup_if_due(db, contact)
                if result:
                    results.append(result)
            except Exception:
                logger.exception("Error processing contact %d", contact.id)
            finally:
                # Always release — even on error — so the lease doesn't expire
                # unnecessarily and block this contact from being retried.
                release_claim(db, contact)

        db.commit()

        logger.info(
            "Follow-up cycle completed (worker=%s): %d claimed, %d processed",
            effective_worker_id,
            len(contact_ids),
            len(results),
        )
        return results


# Global instance
integrated_engine = IntegratedFollowUpEngine()
