"""
Conversation-aware follow-up scheduling engine.

Integrates semantic understanding with follow-up timing decisions.
Respects prospect-requested timing and relationship state.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from mailer_agent.models import Campaign, Contact, ContactStatus
from mailer_agent.semantic_models import BuyingStage, IntentType, SemanticIntent, UrgencyLevel
from mailer_agent.state_machine import can_send_followup, should_cancel_scheduled_followup

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
        next_time = datetime.utcnow() + timedelta(days=days_to_wait)
        
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
        if contact.next_action_at > datetime.utcnow():
            wait_seconds = (contact.next_action_at - datetime.utcnow()).total_seconds()
            return False, f"Not due yet (wait {wait_seconds:.0f}s)"
        
        # Check if recently replied
        if contact.last_reply_at:
            time_since_reply = datetime.utcnow() - contact.last_reply_at
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
        - Schedule for long-term nurture
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
            # Long-term opportunity - schedule far out
            nurture_days = 90  # 3 months
            next_time = datetime.utcnow() + timedelta(days=nurture_days)
            logger.info(
                f"Contact {contact.id} long-term opportunity - scheduling {nurture_days} days out"
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
        
        # Default: resume normal cadence after short delay
        delay_days = 3  # Standard delay after reply
        next_time = datetime.utcnow() + timedelta(days=delay_days)
        logger.debug(
            f"Contact {contact.id} neutral reply - resuming cadence in {delay_days} days"
        )
        return next_time
    
    def _extract_requested_timing(
        self,
        semantic_intent: SemanticIntent
    ) -> Optional[datetime]:
        """
        Extract specific datetime from requested_timing string.
        
        Examples:
        - "next month" → 30 days from now
        - "Q3" → July 1
        - "18 months" → 540 days from now
        - "next week" → 7 days from now
        """
        
        if not semantic_intent.requested_timing:
            return None
        
        timing_str = semantic_intent.requested_timing.lower()
        now = datetime.utcnow()
        
        # Explicit month references
        if "next month" in timing_str or "in a month" in timing_str:
            return now + timedelta(days=30)
        
        # Week references
        if "next week" in timing_str or "in a week" in timing_str:
            return now + timedelta(days=7)
        
        # Quarter references (approximate)
        quarters = {
            "q1": (1, 1),  # Jan 1
            "q2": (4, 1),  # Apr 1
            "q3": (7, 1),  # Jul 1
            "q4": (10, 1), # Oct 1
        }
        for quarter, (month, day) in quarters.items():
            if quarter in timing_str:
                year = now.year
                if month < now.month:  # Quarter already passed this year
                    year += 1
                return datetime(year, month, day)
        
        # Month count (e.g., "18 months", "6 months")
        import re
        month_match = re.search(r'(\d+)\s*months?', timing_str)
        if month_match:
            months = int(month_match.group(1))
            return now + timedelta(days=months * 30)
        
        # Week count (e.g., "2 weeks", "3 weeks")
        week_match = re.search(r'(\d+)\s*weeks?', timing_str)
        if week_match:
            weeks = int(week_match.group(1))
            return now + timedelta(weeks=weeks)
        
        # Day count (e.g., "5 days", "10 days")
        day_match = re.search(r'(\d+)\s*days?', timing_str)
        if day_match:
            days = int(day_match.group(1))
            return now + timedelta(days=days)
        
        # Year references
        if "next year" in timing_str:
            return datetime(now.year + 1, 1, 1)
        
        # Default: if we can't parse, schedule for 1 month
        logger.warning(
            f"Could not parse timing '{timing_str}', defaulting to 30 days"
        )
        return now + timedelta(days=30)
    
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
            days_since = (datetime.utcnow() - contact.last_outbound_at).days
            hints["days_since_last_outbound"] = days_since
            
            if days_since > 14:
                hints["long_gap"] = True
                hints["mention_previous_outreach"] = True
        
        if contact.last_reply_at:
            days_since = (datetime.utcnow() - contact.last_reply_at).days
            hints["days_since_last_reply"] = days_since
        
        return hints
