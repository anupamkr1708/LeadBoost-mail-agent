"""
Semantic classifier for multi-dimensional intent analysis.

This replaces the single-intent classification with rich, structured
understanding of prospect communication.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from mailer_agent.config import get_settings
from mailer_agent.llm import provider_v2 as llm_provider
from mailer_agent.models import Campaign, Contact
from mailer_agent.semantic_models import (
    BuyingStage,
    ClassificationFailureReason,
    ClassificationResult,
    IntentType,
    SemanticIntent,
    SentimentType,
    UrgencyLevel,
)

logger = logging.getLogger("mailer_agent.semantic.classifier")
settings = get_settings()


# Enhanced classification prompt
SEMANTIC_CLASSIFIER_SYSTEM_PROMPT = """You are a B2B sales intelligence analyst. Your job is to understand prospect email replies in depth, extracting multiple dimensions of meaning simultaneously.

Analyze the prospect's reply and respond with a JSON object containing:

{
  "intents": ["list", "of", "intent_types"],
  "sentiment": "positive|neutral|negative|mixed",
  "buying_stage": "unaware|aware|considering|evaluating|deciding|committed|rejected|nurture",
  "urgency": "immediate|near_term|long_term|no_timeline",
  "requested_timing": "optional string like 'next month', 'Q4', null if none",
  "has_pricing_question": boolean,
  "has_budget_signal": boolean,
  "has_decision_maker_signal": boolean,
  "has_commitment_signal": boolean,
  "requested_information": ["list", "of", "specific", "requests"],
  "objections_raised": ["list", "of", "objections"],
  "questions_asked": ["list", "of", "questions"],
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "requires_human_review": boolean,
  "human_review_reason": "optional reason if requires_human_review is true"
}

**Intent types** (can have multiple):
- positive_interest: Expressed interest, curiosity, positive sentiment
- question: Asked one or more questions
- objection: Raised concerns, objections, pushback
- not_interested: Explicit rejection
- meeting_request: Wants to schedule a call/meeting
- pricing_request: Asking about pricing, costs, budget
- information_request: Requesting more details, materials, case studies
- out_of_office: Automated OOO reply
- unsubscribe: Wants to opt out
- referral: Pointed to another person/department
- timing_constraint: Mentioned specific timing ("contact me later", "next quarter")
- neutral: No clear signal

**Buying stage**:
- unaware: Doesn't know they have a problem
- aware: Knows problem, learning about solutions
- considering: Actively looking at options
- evaluating: Comparing specific vendors/solutions
- deciding: Making final decision
- committed: Ready to buy / already agreed
- rejected: Decided against
- nurture: Interested but not ready (future opportunity)

**Key rules**:
1. MULTIPLE INTENTS: Most real emails have 2-3 intents. e.g., "This looks interesting (positive_interest) but what's the pricing? (pricing_request + question)"
2. COMMERCIAL SIGNALS: Set has_pricing_question, has_budget_signal, etc. based on actual content
3. EXTRACT SPECIFICS: List actual questions, objections, and requests
4. CONFIDENCE: Be honest - low confidence means ambiguous or unclear
5. HUMAN REVIEW: Required for objections, unsupported questions, low confidence, complex situations

Examples:

Input: "This looks interesting -- can we do a call next week? Also, what does pricing look like for a team of 50?"
Output: {
  "intents": ["positive_interest", "meeting_request", "pricing_request", "question"],
  "sentiment": "positive",
  "buying_stage": "evaluating",
  "urgency": "near_term",
  "requested_timing": "next week",
  "has_pricing_question": true,
  "has_budget_signal": true,
  "has_decision_maker_signal": false,
  "has_commitment_signal": false,
  "requested_information": ["pricing for 50-person team"],
  "objections_raised": [],
  "questions_asked": ["pricing for team of 50", "availability for call next week"],
  "confidence": 0.9,
  "reasoning": "Clear positive interest with specific meeting request and pricing question. Strong buying signal.",
  "requires_human_review": false,
  "human_review_reason": null
}

Input: "Not interested, please remove me from your list."
Output: {
  "intents": ["not_interested", "unsubscribe"],
  "sentiment": "negative",
  "buying_stage": "rejected",
  "urgency": "immediate",
  "requested_timing": null,
  "has_pricing_question": false,
  "has_budget_signal": false,
  "has_decision_maker_signal": false,
  "has_commitment_signal": false,
  "requested_information": [],
  "objections_raised": [],
  "questions_asked": [],
  "confidence": 1.0,
  "reasoning": "Explicit rejection and unsubscribe request.",
  "requires_human_review": false,
  "human_review_reason": null
}

Input: "Interesting idea, but we're locked into our current vendor for the next 18 months."
Output: {
  "intents": ["positive_interest", "objection", "timing_constraint"],
  "sentiment": "mixed",
  "buying_stage": "nurture",
  "urgency": "long_term",
  "requested_timing": "18 months",
  "has_pricing_question": false,
  "has_budget_signal": false,
  "has_decision_maker_signal": false,
  "has_commitment_signal": false,
  "requested_information": [],
  "objections_raised": ["locked into current vendor for 18 months"],
  "questions_asked": [],
  "confidence": 0.85,
  "reasoning": "Genuine interest but strong timing objection. Future opportunity.",
  "requires_human_review": false,
  "human_review_reason": null
}

Respond ONLY with the JSON object, no other text."""


def classify_prospect_reply(
    *,
    campaign: Campaign,
    contact: Contact,
    inbound_body: str,
    conversation_context: str
) -> ClassificationResult:
    """
    Classify prospect reply with multi-dimensional semantic analysis.
    
    Returns ClassificationResult with explicit success/failure states.
    """
    start_time = time.time()
    
    # Check for critical keywords first (rule-based safety net)
    rule_based_result = _check_critical_keywords(inbound_body)
    if rule_based_result:
        return rule_based_result
    
    # Attempt LLM classification
    if not llm_provider.is_llm_available():
        return _fallback_classification(inbound_body, "LLM not configured")
    
    try:
        # Build classification prompt
        human_prompt = _build_classification_prompt(
            inbound_body,
            conversation_context,
            contact
        )
        
        # Call LLM
        response = llm_provider.call_llm_json(
            SEMANTIC_CLASSIFIER_SYSTEM_PROMPT,
            human_prompt,
            max_tokens=500,
            temperature=0.3  # Lower temperature for more consistent classification
        )
        
        # Parse response into semantic intent
        semantic_intent = _parse_semantic_response(response)
        
        # Record success
        llm_provider.llm_metrics.record_success()
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return ClassificationResult.from_semantic_intent(
            semantic_intent,
            source="llm",
            model_used=settings.llm_model
        )
        
    except llm_provider.LLMProviderError as e:
        # Explicit provider failure - NOT a semantic result
        llm_provider.llm_metrics.record_failure(e)
        
        logger.warning(
            f"LLM classification failed for contact {contact.id}: {e.reason.value} - {e}"
        )
        
        return ClassificationResult.from_failure(
            reason=e.reason,
            details=str(e),
            source="llm"
        )
    
    except Exception as e:
        # Unexpected error
        logger.exception(f"Unexpected error classifying reply for contact {contact.id}: {e}")
        
        return ClassificationResult.from_failure(
            reason=ClassificationFailureReason.UNKNOWN_ERROR,
            details=str(e),
            source="classifier"
        )


def _check_critical_keywords(body: str) -> Optional[ClassificationResult]:
    """
    Rule-based check for critical keywords that should never be missed.
    
    Returns ClassificationResult if matched, None otherwise.
    """
    lowered = body.lower()
    
    # Unsubscribe patterns (highest priority)
    unsubscribe_patterns = [
        "unsubscribe",
        "remove me",
        "stop emailing",
        "opt out",
        "opt-out",
        "take me off",
        "do not contact",
    ]
    
    if any(pattern in lowered for pattern in unsubscribe_patterns):
        intent = SemanticIntent(
            intents=[IntentType.UNSUBSCRIBE, IntentType.NOT_INTERESTED],
            sentiment=SentimentType.NEGATIVE,
            buying_stage=BuyingStage.REJECTED,
            urgency=UrgencyLevel.IMMEDIATE,
            confidence=0.95,
            reasoning="Explicit unsubscribe keywords detected",
            requires_human_review=False
        )
        return ClassificationResult.from_semantic_intent(
            intent,
            source="rule_based"
        )
    
    # Out of office patterns
    ooo_patterns = [
        "out of office",
        "out of the office",
        "away from my desk",
        "on vacation",
        "on leave",
        "automatic reply",
        "auto-reply",
        "autoreply",
    ]
    
    if any(pattern in lowered for pattern in ooo_patterns):
        intent = SemanticIntent(
            intents=[IntentType.OUT_OF_OFFICE],
            sentiment=SentimentType.NEUTRAL,
            buying_stage=BuyingStage.UNAWARE,
            confidence=0.9,
            reasoning="Out-of-office auto-reply detected",
            requires_human_review=False
        )
        return ClassificationResult.from_semantic_intent(
            intent,
            source="rule_based"
        )
    
    return None


def _build_classification_prompt(
    inbound_body: str,
    conversation_context: str,
    contact: Contact
) -> str:
    """Build the classification prompt with full context."""
    
    prospect_info = f"Prospect: {contact.name or 'Unknown'}"
    if contact.title:
        prospect_info += f", {contact.title}"
    if contact.company:
        prospect_info += f" at {contact.company}"
    
    return f"""{prospect_info}

Conversation history:
{conversation_context}

Most recent prospect reply to classify:
{inbound_body}

Analyze this reply and provide the structured JSON assessment."""


def _parse_semantic_response(response: dict) -> SemanticIntent:
    """
    Parse LLM response into SemanticIntent model.
    
    Handles missing fields gracefully with defaults.
    """
    # Parse intents
    intent_strings = response.get("intents", ["neutral"])
    intents = []
    for intent_str in intent_strings:
        try:
            intents.append(IntentType(intent_str))
        except ValueError:
            logger.warning(f"Unknown intent type: {intent_str}")
    
    if not intents:
        intents = [IntentType.NEUTRAL]
    
    # Parse enums with defaults
    try:
        sentiment = SentimentType(response.get("sentiment", "neutral"))
    except ValueError:
        sentiment = SentimentType.NEUTRAL
    
    try:
        buying_stage = BuyingStage(response.get("buying_stage", "unaware"))
    except ValueError:
        buying_stage = BuyingStage.UNAWARE
    
    try:
        urgency = UrgencyLevel(response.get("urgency", "no_timeline"))
    except ValueError:
        urgency = UrgencyLevel.NO_TIMELINE
    
    # Build semantic intent
    return SemanticIntent(
        intents=intents,
        sentiment=sentiment,
        buying_stage=buying_stage,
        urgency=urgency,
        requested_timing=response.get("requested_timing"),
        has_pricing_question=response.get("has_pricing_question", False),
        has_budget_signal=response.get("has_budget_signal", False),
        has_decision_maker_signal=response.get("has_decision_maker_signal", False),
        has_commitment_signal=response.get("has_commitment_signal", False),
        requested_information=response.get("requested_information", []),
        objections_raised=response.get("objections_raised", []),
        questions_asked=response.get("questions_asked", []),
        confidence=float(response.get("confidence", 0.5)),
        reasoning=response.get("reasoning", ""),
        requires_human_review=response.get("requires_human_review", False),
        human_review_reason=response.get("human_review_reason")
    )


def _fallback_classification(body: str, reason: str) -> ClassificationResult:
    """
    Fallback classification when LLM unavailable.
    
    Returns neutral classification requiring human review.
    """
    intent = SemanticIntent(
        intents=[IntentType.NEUTRAL],
        sentiment=SentimentType.NEUTRAL,
        buying_stage=BuyingStage.UNAWARE,
        confidence=0.2,
        reasoning=f"Fallback classification: {reason}",
        requires_human_review=True,
        human_review_reason="LLM unavailable for classification"
    )
    
    return ClassificationResult.from_semantic_intent(
        intent,
        source="fallback"
    )
