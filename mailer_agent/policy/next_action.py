"""
Next-best-action policy layer.

Separates semantic understanding (what the prospect meant) from action
selection (what we should do about it). This layer applies business rules
and policies to determine appropriate responses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from mailer_agent.models import Contact, ContactStatus
from mailer_agent.semantic_models import BuyingStage, IntentType, SemanticIntent, UrgencyLevel

logger = logging.getLogger("mailer_agent.policy")


class ActionType(str, Enum):
    """Types of actions the system can take."""
    NO_ACTION = "no_action"                    # Do nothing (e.g., OOO)
    DRAFT_REPLY = "draft_reply"                # Generate reply, await human approval
    AUTO_REPLY = "auto_reply"                  # Generate and send reply automatically
    REQUEST_INFORMATION = "request_information"  # Ask prospect for more details
    ANSWER_QUESTION = "answer_question"        # Answer prospect's question
    HANDLE_OBJECTION = "handle_objection"      # Address objection
    PROPOSE_MEETING = "propose_meeting"        # Suggest call/meeting
    PROVIDE_PRICING = "provide_pricing"        # Share pricing information
    SEND_MATERIALS = "send_materials"          # Send case studies, docs, etc.
    NURTURE_SCHEDULE = "nurture_schedule"      # Schedule future follow-up
    ESCALATE_HUMAN = "escalate_human"          # Require human intervention
    CLOSE_DEAL = "close_deal"                  # Move to close
    MARK_LOST = "mark_lost"                    # End pursuit
    SUPPRESS = "suppress"                      # Add to suppression list


@dataclass
class NextBestAction:
    """
    Recommended next action based on semantic analysis + business policy.
    """
    
    # Primary action
    action: ActionType
    
    # Action parameters/context
    message_type: Optional[str] = None  # "reply", "closing", "nurture", etc.
    requires_human_approval: bool = False
    approval_reason: Optional[str] = None
    
    # Timing
    execute_immediately: bool = True
    schedule_for: Optional[str] = None  # Future datetime or relative ("next_month")
    
    # Content guidance
    content_hints: dict = None  # Hints for message generation
    
    # Confidence and reasoning
    confidence: float = 0.0
    reasoning: str = ""
    alternative_actions: list[ActionType] = None
    
    def __post_init__(self):
        if self.content_hints is None:
            self.content_hints = {}
        if self.alternative_actions is None:
            self.alternative_actions = []


class NextBestActionPolicy:
    """
    Policy engine for determining next best action.
    
    Applies business rules to semantic understanding.
    """
    
    def __init__(self, auto_reply_enabled: bool = False):
        self.auto_reply_enabled = auto_reply_enabled
    
    def determine_next_action(
        self,
        *,
        contact: Contact,
        semantic_intent: SemanticIntent,
        has_approved_pricing: bool = False,
        has_case_studies: bool = False
    ) -> NextBestAction:
        """
        Determine next best action based on semantic understanding and context.
        
        Args:
            contact: The contact/prospect
            semantic_intent: Semantic analysis of their reply
            has_approved_pricing: Whether campaign has approved pricing info
            has_case_studies: Whether campaign has case studies/materials
        """
        
        # Rule 1: Terminal/critical intents override everything
        if IntentType.UNSUBSCRIBE in semantic_intent.intents:
            return NextBestAction(
                action=ActionType.SUPPRESS,
                execute_immediately=True,
                confidence=1.0,
                reasoning="Explicit unsubscribe request"
            )
        
        if IntentType.NOT_INTERESTED in semantic_intent.intents:
            # But check confidence - might be misclassified
            if semantic_intent.confidence >= 0.7:
                return NextBestAction(
                    action=ActionType.MARK_LOST,
                    requires_human_approval=True,
                    approval_reason="Confirm prospect truly not interested",
                    confidence=semantic_intent.confidence,
                    reasoning="Prospect explicitly not interested"
                )
            else:
                # Low confidence - human should review
                return NextBestAction(
                    action=ActionType.ESCALATE_HUMAN,
                    requires_human_approval=True,
                    approval_reason="Low confidence 'not interested' classification",
                    confidence=semantic_intent.confidence,
                    reasoning="Ambiguous rejection, needs human review"
                )
        
        # Rule 2: Out of office - do nothing
        if IntentType.OUT_OF_OFFICE in semantic_intent.intents:
            return NextBestAction(
                action=ActionType.NO_ACTION,
                execute_immediately=False,
                confidence=0.95,
                reasoning="Out of office auto-reply, will wait for real response"
            )
        
        # Rule 3: Meeting requests get highest priority
        if IntentType.MEETING_REQUEST in semantic_intent.intents:
            return self._handle_meeting_request(contact, semantic_intent)
        
        # Rule 4: Pricing requests need careful handling
        if IntentType.PRICING_REQUEST in semantic_intent.intents or semantic_intent.has_pricing_question:
            return self._handle_pricing_request(
                contact,
                semantic_intent,
                has_approved_pricing
            )
        
        # Rule 5: Information requests
        if IntentType.INFORMATION_REQUEST in semantic_intent.intents:
            return self._handle_information_request(
                contact,
                semantic_intent,
                has_case_studies
            )
        
        # Rule 6: Objections require thoughtful response
        if IntentType.OBJECTION in semantic_intent.intents:
            return self._handle_objection(contact, semantic_intent)
        
        # Rule 7: Positive interest -> move forward
        if IntentType.POSITIVE_INTEREST in semantic_intent.intents:
            return self._handle_positive_interest(contact, semantic_intent)
        
        # Rule 8: Timing constraints
        if IntentType.TIMING_CONSTRAINT in semantic_intent.intents:
            return self._handle_timing_constraint(contact, semantic_intent)
        
        # Rule 9: Questions that don't fit other categories
        if IntentType.QUESTION in semantic_intent.intents:
            return self._handle_question(contact, semantic_intent)
        
        # Rule 10: Low confidence or ambiguous - human review
        if semantic_intent.confidence < 0.5 or semantic_intent.requires_human_review:
            return NextBestAction(
                action=ActionType.ESCALATE_HUMAN,
                requires_human_approval=True,
                approval_reason=semantic_intent.human_review_reason or f"Low confidence ({semantic_intent.confidence:.2f})",
                confidence=semantic_intent.confidence,
                reasoning="Ambiguous intent, requires human judgment"
            )
        
        # Default: Draft reply with human approval
        return NextBestAction(
            action=ActionType.DRAFT_REPLY,
            message_type="reply",
            requires_human_approval=True,
            approval_reason="Neutral/unclear intent",
            confidence=semantic_intent.confidence,
            reasoning="No clear high-value action identified"
        )
    
    def _handle_meeting_request(
        self,
        contact: Contact,
        intent: SemanticIntent
    ) -> NextBestAction:
        """Handle meeting/call requests."""
        
        # Meeting requests should always involve human for scheduling
        return NextBestAction(
            action=ActionType.PROPOSE_MEETING,
            message_type="closing",
            requires_human_approval=True,  # Human needs to confirm availability
            approval_reason="Meeting scheduling requires calendar coordination",
            content_hints={
                "acknowledge_interest": True,
                "confirm_meeting_request": True,
                "requested_timing": intent.requested_timing,
                "urgency": intent.urgency.value
            },
            confidence=intent.confidence,
            reasoning="Prospect wants to meet - high-value opportunity"
        )
    
    def _handle_pricing_request(
        self,
        contact: Contact,
        intent: SemanticIntent,
        has_approved_pricing: bool
    ) -> NextBestAction:
        """Handle pricing questions."""
        
        if has_approved_pricing:
            # We have approved pricing - can respond but should review
            return NextBestAction(
                action=ActionType.PROVIDE_PRICING,
                message_type="reply",
                requires_human_approval=True,  # Pricing is sensitive
                approval_reason="Pricing discussion requires approval",
                content_hints={
                    "has_pricing_data": True,
                    "pricing_context": intent.requested_information,
                    "has_budget_signal": intent.has_budget_signal
                },
                confidence=intent.confidence,
                reasoning="Pricing request with approved data available"
            )
        else:
            # No approved pricing - must escalate
            return NextBestAction(
                action=ActionType.ESCALATE_HUMAN,
                requires_human_approval=True,
                approval_reason="Pricing request but no approved pricing data",
                content_hints={
                    "needs_pricing_data": True,
                    "pricing_context": intent.requested_information
                },
                confidence=intent.confidence,
                reasoning="Cannot auto-respond to pricing without approved data"
            )
    
    def _handle_information_request(
        self,
        contact: Contact,
        intent: SemanticIntent,
        has_materials: bool
    ) -> NextBestAction:
        """Handle requests for information, case studies, etc."""
        
        if not intent.requested_information:
            # Vague request
            return NextBestAction(
                action=ActionType.REQUEST_INFORMATION,
                message_type="reply",
                requires_human_approval=False if self.auto_reply_enabled and intent.confidence >= 0.7 else True,
                approval_reason="Clarifying information request" if not self.auto_reply_enabled else None,
                content_hints={
                    "ask_for_specifics": True
                },
                confidence=intent.confidence,
                reasoning="Vague information request, need clarification"
            )
        
        # Specific request
        return NextBestAction(
            action=ActionType.SEND_MATERIALS,
            message_type="reply",
            requires_human_approval=True,  # Materials should be reviewed
            approval_reason="Verify materials match request",
            content_hints={
                "requested_items": intent.requested_information,
                "has_materials": has_materials
            },
            confidence=intent.confidence,
            reasoning=f"Specific information requested: {', '.join(intent.requested_information[:3])}"
        )
    
    def _handle_objection(
        self,
        contact: Contact,
        intent: SemanticIntent
    ) -> NextBestAction:
        """Handle objections."""
        
        # Objections always require thoughtful responses
        return NextBestAction(
            action=ActionType.HANDLE_OBJECTION,
            message_type="reply",
            requires_human_approval=True,
            approval_reason="Objections require careful, accurate responses",
            content_hints={
                "objections": intent.objections_raised,
                "sentiment": intent.sentiment.value,
                "has_interest": IntentType.POSITIVE_INTEREST in intent.intents
            },
            confidence=intent.confidence,
            reasoning=f"Objection raised: {', '.join(intent.objections_raised[:2])}"
        )
    
    def _handle_positive_interest(
        self,
        contact: Contact,
        intent: SemanticIntent
    ) -> NextBestAction:
        """Handle positive interest."""
        
        # Check buying stage to determine aggressiveness
        if intent.buying_stage in [BuyingStage.DECIDING, BuyingStage.COMMITTED]:
            # Hot lead - move to close
            return NextBestAction(
                action=ActionType.CLOSE_DEAL,
                message_type="closing",
                requires_human_approval=True,
                approval_reason="High-value closing opportunity",
                content_hints={
                    "buying_stage": intent.buying_stage.value,
                    "urgency": intent.urgency.value,
                    "push_for_commitment": True
                },
                confidence=intent.confidence,
                reasoning="Strong buying signals, ready to close"
            )
        
        elif intent.buying_stage in [BuyingStage.EVALUATING]:
            # Evaluation stage - provide value
            auto_send = self.auto_reply_enabled and intent.confidence >= 0.75
            
            return NextBestAction(
                action=ActionType.AUTO_REPLY if auto_send else ActionType.DRAFT_REPLY,
                message_type="reply",
                requires_human_approval=not auto_send,
                approval_reason=None if auto_send else "Evaluation stage, high-value conversation",
                content_hints={
                    "buying_stage": intent.buying_stage.value,
                    "provide_proof": True,
                    "address_questions": intent.questions_asked
                },
                confidence=intent.confidence,
                reasoning="Prospect evaluating, continue value conversation"
            )
        
        else:
            # Early stage interest
            auto_send = self.auto_reply_enabled and intent.confidence >= 0.7
            
            return NextBestAction(
                action=ActionType.AUTO_REPLY if auto_send else ActionType.DRAFT_REPLY,
                message_type="reply",
                requires_human_approval=not auto_send,
                content_hints={
                    "buying_stage": intent.buying_stage.value,
                    "nurture_interest": True
                },
                confidence=intent.confidence,
                reasoning="Early interest, nurture relationship"
            )
    
    def _handle_timing_constraint(
        self,
        contact: Contact,
        intent: SemanticIntent
    ) -> NextBestAction:
        """Handle timing constraints (contact me later)."""
        
        return NextBestAction(
            action=ActionType.NURTURE_SCHEDULE,
            message_type="nurture",
            execute_immediately=False,
            schedule_for=intent.requested_timing,
            requires_human_approval=True,
            approval_reason="Verify timing interpretation",
            content_hints={
                "requested_timing": intent.requested_timing,
                "acknowledge_timing": True
            },
            confidence=intent.confidence,
            reasoning=f"Prospect requested contact at: {intent.requested_timing}"
        )
    
    def _handle_question(
        self,
        contact: Contact,
        intent: SemanticIntent
    ) -> NextBestAction:
        """Handle general questions."""
        
        # Questions can be auto-answered if high confidence and simple
        if intent.confidence >= 0.75 and len(intent.questions_asked) <= 2:
            auto_send = self.auto_reply_enabled
            
            return NextBestAction(
                action=ActionType.ANSWER_QUESTION if auto_send else ActionType.DRAFT_REPLY,
                message_type="reply",
                requires_human_approval=not auto_send,
                approval_reason=None if auto_send else "Review answer accuracy",
                content_hints={
                    "questions": intent.questions_asked,
                    "provide_direct_answers": True
                },
                confidence=intent.confidence,
                reasoning=f"Direct questions: {', '.join(intent.questions_asked[:2])}"
            )
        else:
            # Complex or low-confidence questions need review
            return NextBestAction(
                action=ActionType.DRAFT_REPLY,
                message_type="reply",
                requires_human_approval=True,
                approval_reason="Complex/ambiguous questions",
                content_hints={
                    "questions": intent.questions_asked
                },
                confidence=intent.confidence,
                reasoning="Multiple or complex questions require careful response"
            )
