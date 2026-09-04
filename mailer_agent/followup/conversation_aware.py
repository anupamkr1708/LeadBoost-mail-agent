"""
Conversation-aware follow-up scheduling engine.

Integrates semantic understanding with follow-up timing decisions.
Respects prospect-requested timing and relationship state.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from mailer_agent.models import Campaign, Contact, ContactStatus
from mailer_agent.semantic_models import BuyingStage, IntentType, SemanticIntent, UrgencyLevel
from mailer_agent.state_machine import can_send_followup, should_cancel_scheduled_followup
from mailer_agent.utils.datetime_utils import utcnow

logger = logging.getLogger("mailer_agent.followup.conversation_aware")


class FollowUpScheduler:
    """
    Conversation-aware follow-up scheduler.
    
    Determines when to send follow-ups based on:
    - Campaign cadence (follow_up_days)
    - Relationship state
    - Semantic analysis (timing requests, urgency)
    - Last interaction timestamps
    """
    
    def compute_next_followup(
        self,
        contact: Contact,
        campaign: Campaign,
        semantic_intent: Optional[SemanticIntent] = None
    ) -> Optional[datetime]:
        """
        Compute next follow-up time based on conversation state.
        
        Returns None if no follow-up should be scheduled.
        """
        
        # Check if follow-up should be cancelled
        if should_cancel_scheduled_followup(contact):
            logger.info(
                f"Contact {contact.id} in state {contact.status} - follow-up cancelled"
            )
            return None
        
        # Check if we've exhausted the cadence
        if contact.follow_up_index >= len(campaign.follow_up_days or []):
            logger.info(
                f"Contact {contact.id} exhausted follow-up cadence "
                f"(index {contact.follow_up_index} >= {len(campaign.follow_up_days or [])})"
            )
            return None
        
        # If semantic intent provided, check for timing constraints
        if semantic_intent:
            requested_timing = self._extract_requested_timing(semantic_intent)
            if requested_timing:
                logger.info(
                    f"Contact {contact.id} requested timing: {requested_timing}"
                )
                return requested_timing
        
        # Use campaign cadence
        days_to_wait = campaign.follow_up_days[contact.follow_up_index]
        next_time = utcnow() + timedelta(days=days_to_wait)
        
        logger.debug(
            f"Contact {contact.id} next follow-up in {days_to_wait} days "
            f"(index {contact.follow_up_index})"
        )
        
        return next_time
    
    def should_send_followup_now(self, contact: Contact) -> tuple[bool, str]:
        """
        Determine if follow-up should send right now.
        
        Returns (should_send, reason).
        """
        
        # Must be in ACTIVE state
        if contact.status != ContactStatus.ACTIVE.value:
            return False, f"Status is {contact.status}, not ACTIVE"
        
        # Must have next_action_at scheduled
        if not contact.next_action_at:
            return False, "No next_action_at scheduled"
        
        # Must be due
        if contact.next_action_at > utcnow():
            wait_seconds = (contact.next_action_at - utcnow()).total_seconds()
            return False, f"Not due yet (wait {wait_seconds:.0f}s)"
        
        # Check if recently replied
        if contact.last_reply_at:
            time_since_reply = utcnow() - contact.last_reply_at
            if time_since_reply.days < 1:
                return False, f"Prospect replied {time_since_reply.seconds // 3600}h ago - conversation active"
        
        # All checks passed
        return True, "Due and eligible"
    
    def reschedule_after_reply(
        self,
        contact: Contact,
        campaign: Campaign,
        semantic_intent: SemanticIntent
    ) -> Optional[datetime]:
        """
        Reschedule follow-up after prospect reply.
        
        Based on semantic analysis, may:
        - Cancel follow-up (active conversation)
        - Schedule for requested time (timing constraint)
        - Schedule for long-term nurture (using campaign cadence)
        - Keep existing schedule
        """
        
        # Check for timing constraints
        if IntentType.TIMING_CONSTRAINT in semantic_intent.intents:
            requested_time = self._extract_requested_timing(semantic_intent)
            if requested_time:
                logger.info(
                    f"Contact {contact.id} requested future contact: {semantic_intent.requested_timing}"
                )
                return requested_time
        
        # Check urgency
        if semantic_intent.urgency == UrgencyLevel.LONG_TERM:
            # Long-term opportunity - use campaign's max follow-up delay or default
            follow_up_days = campaign.follow_up_days or []
            if follow_up_days:
                # Use the longest delay from campaign cadence
                nurture_days = max(follow_up_days) * 2  # Double the longest follow-up
            else:
                # No campaign cadence defined, use reasonable default
                nurture_days = 30  # 1 month as fallback
            
            next_time = utcnow() + timedelta(days=nurture_days)
            logger.info(
                f"Contact {contact.id} long-term opportunity - scheduling {nurture_days} days out "
                f"(based on campaign cadence)"
            )
            return next_time
        
        # Active conversation (no timing constraint, not long-term)
        # Don't schedule automatic follow-up - human should handle
        if IntentType.POSITIVE_INTEREST in semantic_intent.intents or \
           IntentType.MEETING_REQUEST in semantic_intent.intents or \
           IntentType.PRICING_REQUEST in semantic_intent.intents:
            logger.info(
                f"Contact {contact.id} active conversation - no automatic follow-up"
            )
            return None
        
        # Default: resume normal cadence using campaign's first follow-up delay
        follow_up_days = campaign.follow_up_days or []
        if follow_up_days:
            # Use campaign's first follow-up delay
            delay_days = follow_up_days[0]
        else:
            # No campaign cadence, must have a fallback
            delay_days = 7  # One week as absolute minimum default
        
        next_time = utcnow() + timedelta(days=delay_days)
        logger.debug(
            f"Contact {contact.id} neutral reply - resuming cadence in {delay_days} days "
            f"(from campaign policy)"
        )
        return next_time
    
    def _extract_requested_timing(
        self,
        semantic_intent: SemanticIntent
    ) -> Optional[datetime]:
        """
        Read the LLM's structured timing interpretation
        (SemanticIntent.timing), rather than regex-parsing a free-text
        string here.

        This used to receive a bare string ("next month", "18 months",
        "sometime next quarter maybe") and regex-match it against a
        hardcoded phrase list, falling back to a hardcoded 30-day
        default whenever nothing matched -- exactly the kind of semantic
        heuristic (and silently-invented business date) this
        architecture is supposed to avoid. The LLM now does the
        temporal interpretation directly (see semantic/classifier.py's
        prompt) and reports a normalized_target ONLY when it can
        genuinely determine one; deterministic Python here only
        ever *consumes* that decision.

        Returns None whenever there's no specific date to act on --
        including when the LLM explicitly flagged the timing as vague/
        conditional/ambiguous (requires_clarification=True). Callers
        treat None as "no override, use the campaign's normal cadence" --
        this deliberately does NOT invent a fallback date for the
        unclear case; it just declines to override, which is the safe
        behavior for "we don't actually know when they want to be
        contacted".
        """
        timing = semantic_intent.timing
        if not timing:
            return None

        if timing.requires_clarification or not timing.normalized_target:
            logger.info(
                "Timing signal present but not resolvable to a specific "
                "date (expression=%r, requires_clarification=%s) -- no "
                "override; falling back to campaign cadence.",
                timing.expression, timing.requires_clarification,
            )
            return None

        try:
            parsed = datetime.fromisoformat(timing.normalized_target)
        except (ValueError, TypeError):
            logger.warning(
                "LLM provided an unparseable normalized_target %r for "
                "expression %r -- treating as no override.",
                timing.normalized_target, timing.expression,
            )
            return None

        if parsed.tzinfo is None:
            from mailer_agent.utils.datetime_utils import UTC
            parsed = parsed.replace(tzinfo=UTC)

        return parsed
    
    def get_followup_message_hint(self, contact: Contact) -> dict:
        """
        Get context hints for follow-up message generation.
        
        Helps message generator understand conversation context.
        """
        hints = {
            "follow_up_number": contact.follow_up_index + 1,
            "has_previous_reply": contact.last_reply_at is not None,
            "buying_stage": contact.buying_stage,
            "engagement_score": contact.engagement_score,
        }
        
        # Add timing context
        if contact.last_outbound_at:
            days_since = (utcnow() - contact.last_outbound_at).days
            hints["days_since_last_outbound"] = days_since
            
            if days_since > 14:
                hints["long_gap"] = True
                hints["mention_previous_outreach"] = True
        
        if contact.last_reply_at:
            days_since = (utcnow() - contact.last_reply_at).days
            hints["days_since_last_reply"] = days_since
        
        return hints
