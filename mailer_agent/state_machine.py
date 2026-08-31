"""
Relationship state machine for B2B sales contacts.

Manages legal state transitions based on events and semantic analysis.
Prevents invalid transitions and enforces business rules.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from mailer_agent.models import Contact, ContactStatus
from mailer_agent.semantic_models import BuyingStage, IntentType, SemanticIntent
from mailer_agent.utils.datetime_utils import utcnow

logger = logging.getLogger("mailer_agent.state_machine")


class StateTransitionEvent(str, Enum):
    """Events that can trigger state transitions."""
    INITIAL_SENT = "initial_sent"
    FOLLOWUP_SENT = "followup_sent"
    REPLY_RECEIVED = "reply_received"
    POSITIVE_INTEREST = "positive_interest"
    MEETING_REQUESTED = "meeting_requested"
    MEETING_CONFIRMED = "meeting_confirmed"
    PRICING_DISCUSSED = "pricing_discussed"
    OBJECTION_RAISED = "objection_raised"
    NOT_INTERESTED = "not_interested"
    UNSUBSCRIBED = "unsubscribed"
    OUT_OF_OFFICE = "out_of_office"
    SEQUENCE_EXHAUSTED = "sequence_exhausted"
    MANUAL_PAUSE = "manual_pause"
    MANUAL_RESUME = "manual_resume"
    MANUAL_WON = "manual_won"
    MANUAL_LOST = "manual_lost"
    NEEDS_HUMAN = "needs_human"


# Valid state transitions map: current_state -> {event -> new_state}
STATE_TRANSITIONS: dict[str, dict[StateTransitionEvent, str]] = {
    ContactStatus.NEW.value: {
        StateTransitionEvent.INITIAL_SENT: ContactStatus.ACTIVE.value,
        StateTransitionEvent.MANUAL_PAUSE: ContactStatus.PAUSED.value,
    },
    
    ContactStatus.ACTIVE.value: {
        StateTransitionEvent.FOLLOWUP_SENT: ContactStatus.ACTIVE.value,  # Stay active
        StateTransitionEvent.REPLY_RECEIVED: ContactStatus.REPLIED.value,
        StateTransitionEvent.SEQUENCE_EXHAUSTED: ContactStatus.SEQUENCE_COMPLETE.value,
        StateTransitionEvent.MANUAL_PAUSE: ContactStatus.PAUSED.value,
        StateTransitionEvent.MANUAL_WON: ContactStatus.CLOSED_WON.value,
        StateTransitionEvent.MANUAL_LOST: ContactStatus.CLOSED_LOST.value,
    },
    
    ContactStatus.REPLIED.value: {
        StateTransitionEvent.POSITIVE_INTEREST: ContactStatus.ENGAGED.value,
        StateTransitionEvent.MEETING_REQUESTED: ContactStatus.MEETING_REQUESTED.value,
        StateTransitionEvent.OBJECTION_RAISED: ContactStatus.ENGAGED.value,  # Still engaged
        StateTransitionEvent.NOT_INTERESTED: ContactStatus.CLOSED_LOST.value,
        StateTransitionEvent.UNSUBSCRIBED: ContactStatus.SUPPRESSED.value,
        StateTransitionEvent.OUT_OF_OFFICE: ContactStatus.ACTIVE.value,  # Back to active
        StateTransitionEvent.NEEDS_HUMAN: ContactStatus.NEEDS_REVIEW.value,
        StateTransitionEvent.FOLLOWUP_SENT: ContactStatus.ACTIVE.value,
    },
    
    ContactStatus.ENGAGED.value: {
        StateTransitionEvent.REPLY_RECEIVED: ContactStatus.REPLIED.value,
        StateTransitionEvent.MEETING_REQUESTED: ContactStatus.MEETING_REQUESTED.value,
        StateTransitionEvent.PRICING_DISCUSSED: ContactStatus.EVALUATING.value,
        StateTransitionEvent.NOT_INTERESTED: ContactStatus.CLOSED_LOST.value,
        StateTransitionEvent.UNSUBSCRIBED: ContactStatus.SUPPRESSED.value,
        StateTransitionEvent.OBJECTION_RAISED: ContactStatus.ENGAGED.value,  # Stay engaged
        StateTransitionEvent.MANUAL_WON: ContactStatus.CLOSED_WON.value,
        StateTransitionEvent.MANUAL_LOST: ContactStatus.CLOSED_LOST.value,
    },
    
    ContactStatus.EVALUATING.value: {
        StateTransitionEvent.REPLY_RECEIVED: ContactStatus.REPLIED.value,
        StateTransitionEvent.MEETING_REQUESTED: ContactStatus.MEETING_REQUESTED.value,
        StateTransitionEvent.POSITIVE_INTEREST: ContactStatus.NEGOTIATING.value,
        StateTransitionEvent.NOT_INTERESTED: ContactStatus.CLOSED_LOST.value,
        StateTransitionEvent.UNSUBSCRIBED: ContactStatus.SUPPRESSED.value,
        StateTransitionEvent.MANUAL_WON: ContactStatus.CLOSED_WON.value,
        StateTransitionEvent.MANUAL_LOST: ContactStatus.CLOSED_LOST.value,
    },
    
    ContactStatus.MEETING_REQUESTED.value: {
        StateTransitionEvent.MEETING_CONFIRMED: ContactStatus.MEETING_SCHEDULED.value,
        StateTransitionEvent.REPLY_RECEIVED: ContactStatus.REPLIED.value,
        StateTransitionEvent.NOT_INTERESTED: ContactStatus.CLOSED_LOST.value,
        StateTransitionEvent.UNSUBSCRIBED: ContactStatus.SUPPRESSED.value,
        StateTransitionEvent.NEEDS_HUMAN: ContactStatus.NEEDS_REVIEW.value,
    },
    
    ContactStatus.MEETING_SCHEDULED.value: {
        StateTransitionEvent.POSITIVE_INTEREST: ContactStatus.NEGOTIATING.value,
        StateTransitionEvent.NOT_INTERESTED: ContactStatus.CLOSED_LOST.value,
        StateTransitionEvent.MANUAL_WON: ContactStatus.CLOSED_WON.value,
        StateTransitionEvent.MANUAL_LOST: ContactStatus.CLOSED_LOST.value,
    },
    
    ContactStatus.NEGOTIATING.value: {
        StateTransitionEvent.REPLY_RECEIVED: ContactStatus.REPLIED.value,
        StateTransitionEvent.MANUAL_WON: ContactStatus.CLOSED_WON.value,
        StateTransitionEvent.MANUAL_LOST: ContactStatus.CLOSED_LOST.value,
        StateTransitionEvent.NOT_INTERESTED: ContactStatus.CLOSED_LOST.value,
    },
    
    ContactStatus.NURTURE.value: {
        StateTransitionEvent.REPLY_RECEIVED: ContactStatus.REPLIED.value,
        StateTransitionEvent.POSITIVE_INTEREST: ContactStatus.ENGAGED.value,
        StateTransitionEvent.MANUAL_RESUME: ContactStatus.ACTIVE.value,
        StateTransitionEvent.NOT_INTERESTED: ContactStatus.CLOSED_LOST.value,
        StateTransitionEvent.UNSUBSCRIBED: ContactStatus.SUPPRESSED.value,
    },
    
    ContactStatus.SEQUENCE_COMPLETE.value: {
        StateTransitionEvent.REPLY_RECEIVED: ContactStatus.REPLIED.value,
        StateTransitionEvent.MANUAL_RESUME: ContactStatus.ACTIVE.value,
        StateTransitionEvent.MANUAL_WON: ContactStatus.CLOSED_WON.value,
        StateTransitionEvent.MANUAL_LOST: ContactStatus.CLOSED_LOST.value,
    },
    
    ContactStatus.NEEDS_REVIEW.value: {
        StateTransitionEvent.MANUAL_RESUME: ContactStatus.ACTIVE.value,
        StateTransitionEvent.MANUAL_WON: ContactStatus.CLOSED_WON.value,
        StateTransitionEvent.MANUAL_LOST: ContactStatus.CLOSED_LOST.value,
        StateTransitionEvent.MANUAL_PAUSE: ContactStatus.PAUSED.value,
    },
    
    ContactStatus.PAUSED.value: {
        StateTransitionEvent.MANUAL_RESUME: ContactStatus.ACTIVE.value,
        StateTransitionEvent.MANUAL_WON: ContactStatus.CLOSED_WON.value,
        StateTransitionEvent.MANUAL_LOST: ContactStatus.CLOSED_LOST.value,
    },
    
    # Terminal states - very limited transitions
    ContactStatus.CLOSED_WON.value: {},
    ContactStatus.CLOSED_LOST.value: {
        StateTransitionEvent.MANUAL_RESUME: ContactStatus.ACTIVE.value,  # Allow reopening
    },
    ContactStatus.SUPPRESSED.value: {},  # No transitions out - permanent
}


class StateTransitionError(Exception):
    """Raised when attempting an invalid state transition."""
    pass


def transition_contact_state(
    contact: Contact,
    event: StateTransitionEvent,
    reason: Optional[str] = None
) -> bool:
    """
    Attempt to transition contact to new state based on event.
    
    Returns True if transition occurred, False if not allowed.
    Raises StateTransitionError if transition is invalid.
    """
    current_state = contact.status
    
    # Check if transition is valid
    valid_transitions = STATE_TRANSITIONS.get(current_state, {})
    
    if event not in valid_transitions:
        # Not a valid transition
        logger.debug(
            f"Invalid transition for contact {contact.id}: "
            f"{current_state} + {event.value} (no valid transition)"
        )
        return False
    
    new_state = valid_transitions[event]
    
    logger.info(
        f"Contact {contact.id} state transition: {current_state} -> {new_state} "
        f"(event: {event.value})"
        + (f", reason: {reason}" if reason else "")
    )
    
    contact.status = new_state
    contact.updated_at = utcnow()
    
    return True


def infer_event_from_semantic_intent(intent: SemanticIntent) -> StateTransitionEvent:
    """
    Infer state transition event from semantic analysis.
    
    Maps multi-dimensional intent to primary state-changing event.
    """
    # Priority order matters
    
    # Terminal/critical intents first
    if IntentType.UNSUBSCRIBE in intent.intents:
        return StateTransitionEvent.UNSUBSCRIBED
    
    if IntentType.NOT_INTERESTED in intent.intents:
        return StateTransitionEvent.NOT_INTERESTED
    
    if IntentType.OUT_OF_OFFICE in intent.intents:
        return StateTransitionEvent.OUT_OF_OFFICE
    
    # High-value intents
    if IntentType.MEETING_REQUEST in intent.intents:
        return StateTransitionEvent.MEETING_REQUESTED
    
    if IntentType.PRICING_REQUEST in intent.intents and intent.has_budget_signal:
        return StateTransitionEvent.PRICING_DISCUSSED
    
    # Interest signals
    if IntentType.POSITIVE_INTEREST in intent.intents:
        # Check buying stage to determine if this is engagement or negotiation
        if intent.buying_stage in [BuyingStage.EVALUATING, BuyingStage.DECIDING]:
            return StateTransitionEvent.PRICING_DISCUSSED
        return StateTransitionEvent.POSITIVE_INTEREST
    
    # Objection handling
    if IntentType.OBJECTION in intent.intents:
        return StateTransitionEvent.OBJECTION_RAISED
    
    # Default: generic reply received
    return StateTransitionEvent.REPLY_RECEIVED


def should_cancel_scheduled_followup(contact: Contact) -> bool:
    """
    Determine if scheduled follow-up should be cancelled based on current state.
    
    Prevents follow-ups when prospect replied or conversation is active.
    """
    # Cancel follow-up if in these states
    cancel_states = {
        ContactStatus.REPLIED.value,
        ContactStatus.ENGAGED.value,
        ContactStatus.EVALUATING.value,
        ContactStatus.MEETING_REQUESTED.value,
        ContactStatus.MEETING_SCHEDULED.value,
        ContactStatus.NEGOTIATING.value,
        ContactStatus.NEEDS_REVIEW.value,
        ContactStatus.CLOSED_WON.value,
        ContactStatus.CLOSED_LOST.value,
        ContactStatus.SUPPRESSED.value,
        ContactStatus.PAUSED.value,
    }
    
    return contact.status in cancel_states


def can_send_followup(contact: Contact) -> bool:
    """
    Determine if it's safe to send a follow-up to this contact.
    
    Checks both state and timing.
    """
    # Only send follow-ups when ACTIVE
    if contact.status != ContactStatus.ACTIVE.value:
        return False
    
    # Must have next_action_at scheduled
    if not contact.next_action_at:
        return False
    
    # Must be due
    if contact.next_action_at > utcnow():
        return False
    
    return True
