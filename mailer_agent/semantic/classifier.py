"""
Semantic classifier for multi-dimensional intent analysis.

This replaces the single-intent classification with rich, structured
understanding of prospect communication.

Deterministic exceptions in this module (documented, not hidden)
-------------------------------------------------------------------
Two checks here run before any LLM call, and neither is a semantic
heuristic in the sense this architecture forbids -- both are narrow,
justified safety/metadata checks, not attempts to interpret meaning:

1. Unsubscribe keyword matching (_check_unsubscribe_keywords): a
   deterministic safety net, not the primary interpreter. It exists so
   an opt-out is never missed due to an LLM outage or misclassification
   -- explicitly permitted as a "deterministic policy invariant" rather
   than a semantic classifier. The LLM is still free to also recognize
   unsubscribe intent semantically; this is a floor, not a replacement.

2. Auto-Submitted metadata check (_check_auto_submitted_metadata): this
   reads a structured, protocol-level fact (the RFC 3834 header, when
   the inbound channel provides it) rather than guessing from body text.
   When that signal isn't available, out-of-office is left to genuine
   semantic classification -- there is deliberately no hardcoded phrase
   list for it (there used to be one; removed, see git history / prior
   production-readiness report for why).
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
    Certainty,
    ClassificationFailureReason,
    ClassificationResult,
    IntentType,
    SemanticFact,
    SemanticIntent,
    SentimentType,
    SpeechAct,
    TimingKind,
    TimingSignal,
    UrgencyLevel,
)

logger = logging.getLogger("mailer_agent.semantic.classifier")
settings = get_settings()

# Bump when the prompt/schema changes in a way that would affect how a
# past classification should be interpreted -- recorded on every
# ClassificationResult (see semantic_models.py) for observability/replay.
PROMPT_VERSION = "semantic-classifier-v2"


# Enhanced classification prompt
SEMANTIC_CLASSIFIER_SYSTEM_PROMPT = """You are a B2B sales intelligence analyst. Your job is to understand prospect email replies in depth, extracting multiple dimensions of meaning simultaneously, using the full conversation context provided -- not just the latest message in isolation. The same phrase can mean different things depending on what was said earlier and what the campaign/prospect context is; use that context.

Analyze the prospect's reply and respond with a JSON object containing:

{
  "intents": ["list", "of", "intent_types"],
  "speech_act": "statement|question|request|commitment|rejection|acknowledgment",
  "sentiment": "positive|neutral|negative|mixed",
  "buying_stage": "unaware|aware|considering|evaluating|deciding|committed|rejected|nurture",
  "user_goal": "one short phrase describing what the prospect is trying to accomplish right now, or null if unclear",
  "pain_points": ["list", "of", "problems", "they", "mentioned"],
  "objections_raised": ["list", "of", "objections"],
  "constraints": ["list", "of", "constraints", "e.g. locked into current contract, needs procurement approval"],
  "urgency": "immediate|near_term|long_term|no_timeline",
  "has_pricing_question": boolean,
  "has_budget_signal": boolean,
  "has_decision_maker_signal": boolean,
  "has_commitment_signal": boolean,
  "procurement_signal": boolean,
  "requested_information": ["list", "of", "specific", "requests"],
  "questions_asked": ["list", "of", "questions"],
  "commitments_made": ["list", "of", "promises", "the", "PROSPECT", "made"],
  "current_solution": {"value": "e.g. Salesforce", "certainty": "explicit|strongly_inferred|weakly_inferred|unknown", "evidence": "short quote"} or null if not mentioned,
  "competitors_mentioned": [{"value": "...", "certainty": "...", "evidence": "..."}],
  "new_facts": [{"value": "...", "certainty": "...", "evidence": "..."}],
  "contradicted_facts": ["plain text notes on anything this reply contradicts from earlier in the conversation"],
  "unresolved_items": ["questions or asks still needing an answer after this message"],
  "timing": {
    "expression": "the raw phrase, e.g. 'sometime next quarter'",
    "kind": "specific_date|relative_period|quarter|conditional|vague",
    "normalized_target": "ISO 8601 date ONLY if you can genuinely determine one from context, else null",
    "certainty": "explicit|strongly_inferred|weakly_inferred|unknown",
    "commitment_strength": "weak|moderate|strong",
    "requires_clarification": boolean
  } or null if no timing was mentioned,
  "confidence": 0.0-1.0,
  "uncertain_aspects": ["plain text notes on anything genuinely ambiguous in this interpretation"],
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
2. EXPLICIT VS INFERRED: Only mark something "explicit" if the prospect stated it directly. A conclusion you drew from context is "strongly_inferred" or "weakly_inferred" -- never present an inference as if it were a stated fact. It is normal and expected for current_solution/timing/etc to be null when nothing was said about them -- do not invent a value to fill the field.
3. TIMING: Only set normalized_target when you can genuinely determine a specific date/period from context (e.g. an explicit month, or "next week" relative to today). For vague ("sometime", "eventually") or conditional ("after our renewal") timing, leave normalized_target null and set requires_clarification true -- do not guess a date.
4. COMMERCIAL SIGNALS: Set has_pricing_question, has_budget_signal, etc. based on actual content
5. EXTRACT SPECIFICS: List actual questions, objections, pain points, and requests in the prospect's own terms
6. CONFIDENCE: Be honest - low confidence means ambiguous or unclear
7. HUMAN REVIEW: Required for objections, unsupported questions, low confidence, complex situations

Examples:

Input: "This looks interesting -- can we do a call next week? Also, what does pricing look like for a team of 50?"
Output: {
  "intents": ["positive_interest", "meeting_request", "pricing_request", "question"],
  "speech_act": "request",
  "sentiment": "positive",
  "buying_stage": "evaluating",
  "user_goal": "assess fit and cost before committing time to a call",
  "pain_points": [],
  "objections_raised": [],
  "constraints": [],
  "urgency": "near_term",
  "has_pricing_question": true,
  "has_budget_signal": true,
  "has_decision_maker_signal": false,
  "has_commitment_signal": false,
  "procurement_signal": false,
  "requested_information": ["pricing for 50-person team"],
  "questions_asked": ["pricing for team of 50", "availability for call next week"],
  "commitments_made": [],
  "current_solution": null,
  "competitors_mentioned": [],
  "new_facts": [{"value": "team size is around 50 people", "certainty": "explicit", "evidence": "pricing look like for a team of 50"}],
  "contradicted_facts": [],
  "unresolved_items": ["pricing for team of 50", "call availability next week"],
  "timing": {"expression": "next week", "kind": "relative_period", "normalized_target": null, "certainty": "explicit", "commitment_strength": "moderate", "requires_clarification": false},
  "confidence": 0.9,
  "uncertain_aspects": [],
  "reasoning": "Clear positive interest with specific meeting request and pricing question. Strong buying signal.",
  "requires_human_review": false,
  "human_review_reason": null
}

Input: "Not interested, please remove me from your list."
Output: {
  "intents": ["not_interested", "unsubscribe"],
  "speech_act": "rejection",
  "sentiment": "negative",
  "buying_stage": "rejected",
  "user_goal": "stop receiving these emails",
  "pain_points": [], "objections_raised": [], "constraints": [],
  "urgency": "immediate",
  "has_pricing_question": false, "has_budget_signal": false, "has_decision_maker_signal": false, "has_commitment_signal": false, "procurement_signal": false,
  "requested_information": [], "questions_asked": [], "commitments_made": [],
  "current_solution": null, "competitors_mentioned": [], "new_facts": [], "contradicted_facts": [], "unresolved_items": [],
  "timing": null,
  "confidence": 1.0,
  "uncertain_aspects": [],
  "reasoning": "Explicit rejection and unsubscribe request.",
  "requires_human_review": false,
  "human_review_reason": null
}

Input: "Interesting idea, but we're locked into our current Salesforce contract for the next 18 months."
Output: {
  "intents": ["positive_interest", "objection", "timing_constraint"],
  "speech_act": "statement",
  "sentiment": "mixed",
  "buying_stage": "nurture",
  "user_goal": "flag a real constraint while keeping the door open for later",
  "pain_points": [],
  "objections_raised": ["locked into current vendor contract for 18 months"],
  "constraints": ["existing Salesforce contract has 18 months remaining"],
  "urgency": "long_term",
  "has_pricing_question": false, "has_budget_signal": false, "has_decision_maker_signal": false, "has_commitment_signal": false, "procurement_signal": false,
  "requested_information": [], "questions_asked": [], "commitments_made": [],
  "current_solution": {"value": "Salesforce", "certainty": "explicit", "evidence": "locked into our current Salesforce contract"},
  "competitors_mentioned": [],
  "new_facts": [{"value": "current contract has 18 months remaining", "certainty": "explicit", "evidence": "locked into our current Salesforce contract for the next 18 months"}],
  "contradicted_facts": [],
  "unresolved_items": [],
  "timing": {"expression": "18 months", "kind": "relative_period", "normalized_target": null, "certainty": "explicit", "commitment_strength": "weak", "requires_clarification": false},
  "confidence": 0.85,
  "uncertain_aspects": ["unclear whether they'd revisit at contract renewal or need to be re-approached"],
  "reasoning": "Genuine interest but strong contractual timing objection. Future opportunity, not a rejection.",
  "requires_human_review": false,
  "human_review_reason": null
}

Respond ONLY with the JSON object, no other text."""


def classify_prospect_reply(
    *,
    campaign: Campaign,
    contact: Contact,
    inbound_body: str,
    conversation_context: str,
    auto_submitted: bool = False,
) -> ClassificationResult:
    """
    Classify prospect reply with multi-dimensional semantic analysis.
    
    Returns ClassificationResult with explicit success/failure states.

    `auto_submitted`: pass through InboundEmail.auto_submitted when
    available (see mail/imap_reader.py) -- a real protocol-level signal
    that this is an autoresponder, checked deterministically before any
    LLM call. When False (signal unavailable, not "confirmed human"),
    out-of-office is left entirely to the LLM's semantic judgment.
    """
    start_time = time.time()
    
    # Deterministic safety/metadata checks first (see module docstring
    # for why these two specifically are not semantic heuristics).
    rule_based_result = _check_unsubscribe_keywords(inbound_body)
    if rule_based_result:
        return rule_based_result

    metadata_result = _check_auto_submitted_metadata(auto_submitted)
    if metadata_result:
        return metadata_result
    
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
            max_tokens=900,
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
            model_used=settings.llm_model,
            prompt_version=PROMPT_VERSION,
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


def _check_unsubscribe_keywords(body: str) -> Optional[ClassificationResult]:
    """
    Rule-based safety net for unsubscribe requests -- see module
    docstring for why this specific check is a deterministic policy
    invariant, not a semantic heuristic. It exists so an opt-out is
    never missed due to an LLM outage; it does not replace the LLM's own
    ability to recognize unsubscribe intent semantically for phrasings
    this fixed list doesn't cover.

    Returns ClassificationResult if matched, None otherwise.
    """
    lowered = body.lower()

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
            speech_act=SpeechAct.REJECTION,
            sentiment=SentimentType.NEGATIVE,
            buying_stage=BuyingStage.REJECTED,
            urgency=UrgencyLevel.IMMEDIATE,
            confidence=0.95,
            reasoning="Explicit unsubscribe keywords detected",
            requires_human_review=False
        )
        return ClassificationResult.from_semantic_intent(
            intent,
            source="rule_based",
            prompt_version=PROMPT_VERSION,
        )

    return None


def _check_auto_submitted_metadata(auto_submitted: bool) -> Optional[ClassificationResult]:
    """
    Structured-metadata OOO check (see module docstring). Deliberately
    NOT a body-text keyword list -- when the channel doesn't supply this
    signal, out-of-office is left to genuine semantic classification by
    the LLM (IntentType.OUT_OF_OFFICE is part of the classifier's
    normal vocabulary; see the system prompt).
    """
    if not auto_submitted:
        return None

    intent = SemanticIntent(
        intents=[IntentType.OUT_OF_OFFICE],
        speech_act=SpeechAct.STATEMENT,
        sentiment=SentimentType.NEUTRAL,
        buying_stage=BuyingStage.UNAWARE,
        confidence=0.95,
        reasoning="Auto-Submitted header indicates an automated response",
        requires_human_review=False,
    )
    return ClassificationResult.from_semantic_intent(
        intent,
        source="metadata",
        prompt_version=PROMPT_VERSION,
    )


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


def _parse_certainty(value) -> Certainty:
    try:
        return Certainty(value)
    except (ValueError, TypeError):
        return Certainty.UNKNOWN


def _parse_semantic_fact(raw) -> Optional[SemanticFact]:
    """Defensively parse one {value, certainty, evidence} object. Never raises."""
    if not raw or not isinstance(raw, dict):
        return None
    value = raw.get("value")
    if not value:
        return None
    certainty = _parse_certainty(raw.get("certainty"))
    return SemanticFact(
        value=str(value),
        certainty=certainty,
        explicit=(certainty == Certainty.EXPLICIT),
        evidence=raw.get("evidence"),
    )


def _parse_timing_signal(raw) -> Optional[TimingSignal]:
    """
    Defensively parse the structured timing object. Never raises, never
    invents a normalized_target the model didn't provide -- an absent or
    malformed normalized_target stays None, which is what
    followup/conversation_aware.py treats as "no fallback date, this
    needs clarification" rather than a hardcoded default.
    """
    if not raw or not isinstance(raw, dict):
        return None
    expression = raw.get("expression")
    if not expression:
        return None
    try:
        kind = TimingKind(raw.get("kind")) if raw.get("kind") else None
    except ValueError:
        kind = None
    return TimingSignal(
        expression=str(expression),
        kind=kind,
        normalized_target=raw.get("normalized_target") or None,
        certainty=_parse_certainty(raw.get("certainty")),
        commitment_strength=raw.get("commitment_strength"),
        requires_clarification=bool(raw.get("requires_clarification", False)),
    )


def _parse_semantic_response(response: dict) -> SemanticIntent:
    """
    Parse LLM response into SemanticIntent model.
    
    Handles missing/malformed fields gracefully with safe defaults --
    every sub-parser here is defensive (try/except around enum
    construction, type/truthiness checks before dict access) so a
    partially-malformed response degrades individual fields to
    None/UNKNOWN/empty-list rather than raising and losing the entire
    classification.
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

    try:
        speech_act = SpeechAct(response.get("speech_act")) if response.get("speech_act") else None
    except ValueError:
        speech_act = None

    current_solution = _parse_semantic_fact(response.get("current_solution"))
    competitors_mentioned = [
        f for f in (_parse_semantic_fact(c) for c in response.get("competitors_mentioned", []) or []) if f
    ]
    new_facts = [
        f for f in (_parse_semantic_fact(c) for c in response.get("new_facts", []) or []) if f
    ]
    timing = _parse_timing_signal(response.get("timing"))
    
    # Build semantic intent
    return SemanticIntent(
        intents=intents,
        speech_act=speech_act,
        sentiment=sentiment,
        buying_stage=buying_stage,
        user_goal=response.get("user_goal") or None,
        pain_points=response.get("pain_points", []) or [],
        objections_raised=response.get("objections_raised", []) or [],
        constraints=response.get("constraints", []) or [],
        urgency=urgency,
        has_pricing_question=response.get("has_pricing_question", False),
        has_budget_signal=response.get("has_budget_signal", False),
        has_decision_maker_signal=response.get("has_decision_maker_signal", False),
        has_commitment_signal=response.get("has_commitment_signal", False),
        procurement_signal=response.get("procurement_signal", False),
        requested_information=response.get("requested_information", []) or [],
        questions_asked=response.get("questions_asked", []) or [],
        commitments_made=response.get("commitments_made", []) or [],
        current_solution=current_solution,
        competitors_mentioned=competitors_mentioned,
        new_facts=new_facts,
        contradicted_facts=response.get("contradicted_facts", []) or [],
        unresolved_items=response.get("unresolved_items", []) or [],
        timing=timing,
        confidence=float(response.get("confidence", 0.5)),
        uncertain_aspects=response.get("uncertain_aspects", []) or [],
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
        source="fallback",
        prompt_version=PROMPT_VERSION,
    )
