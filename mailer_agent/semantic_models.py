"""
Structured semantic analysis models for sales intelligence.

These models represent the multi-dimensional understanding of prospect
communication, separating semantic interpretation from provider failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class IntentType(str, Enum):
    """Primary intent categories (non-exclusive)."""
    POSITIVE_INTEREST = "positive_interest"
    QUESTION = "question"
    OBJECTION = "objection"
    NOT_INTERESTED = "not_interested"
    MEETING_REQUEST = "meeting_request"
    PRICING_REQUEST = "pricing_request"
    INFORMATION_REQUEST = "information_request"
    OUT_OF_OFFICE = "out_of_office"
    UNSUBSCRIBE = "unsubscribe"
    REFERRAL = "referral"
    TIMING_CONSTRAINT = "timing_constraint"
    NEUTRAL = "neutral"


class SentimentType(str, Enum):
    """Overall sentiment/tone."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class BuyingStage(str, Enum):
    """B2B buying journey stage."""
    UNAWARE = "unaware"
    AWARE = "aware"
    CONSIDERING = "considering"
    EVALUATING = "evaluating"
    DECIDING = "deciding"
    COMMITTED = "committed"
    REJECTED = "rejected"
    NURTURE = "nurture"


class UrgencyLevel(str, Enum):
    """Timeline urgency."""
    IMMEDIATE = "immediate"        # "this week", "ASAP"
    NEAR_TERM = "near_term"       # "next month", "Q3"
    LONG_TERM = "long_term"       # "next year", "future"
    NO_TIMELINE = "no_timeline"


class ClassificationFailureReason(str, Enum):
    """Explicit failure reasons (not semantic states)."""
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    MALFORMED_OUTPUT = "malformed_output"
    PARSE_ERROR = "parse_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class SemanticIntent:
    """Multi-dimensional intent analysis."""
    
    # Primary intents (can have multiple)
    intents: list[IntentType] = field(default_factory=list)
    
    # Overall sentiment
    sentiment: SentimentType = SentimentType.NEUTRAL
    
    # Buying stage assessment
    buying_stage: BuyingStage = BuyingStage.UNAWARE
    
    # Urgency/timing
    urgency: UrgencyLevel = UrgencyLevel.NO_TIMELINE
    requested_timing: Optional[str] = None  # e.g., "next month", "Q4"
    
    # Commercial signals
    has_pricing_question: bool = False
    has_budget_signal: bool = False
    has_decision_maker_signal: bool = False
    has_commitment_signal: bool = False
    
    # Specific requests
    requested_information: list[str] = field(default_factory=list)
    objections_raised: list[str] = field(default_factory=list)
    questions_asked: list[str] = field(default_factory=list)
    
    # Model confidence and reasoning
    confidence: float = 0.0  # 0.0-1.0
    reasoning: str = ""
    
    # Grounding and safety
    requires_human_review: bool = False
    human_review_reason: Optional[str] = None


@dataclass
class ClassificationResult:
    """
    Complete classification result with explicit success/failure states.
    
    This separates:
    - Semantic analysis (what the prospect meant)
    - Provider reliability (did the LLM work)
    - System confidence (should we act on this)
    """
    
    # Success indicator
    success: bool
    
    # Semantic analysis (only valid if success=True)
    semantic_intent: Optional[SemanticIntent] = None
    
    # Failure information (only valid if success=False)
    failure_reason: Optional[ClassificationFailureReason] = None
    failure_details: Optional[str] = None
    
    # Metadata
    source: str = "unknown"  # "llm", "fallback", "rule_based"
    model_used: Optional[str] = None
    processing_time_ms: Optional[int] = None
    
    @property
    def is_actionable(self) -> bool:
        """Can we safely act on this classification?"""
        if not self.success or not self.semantic_intent:
            return False
        
        # Don't auto-act on low confidence or human-review-required
        if self.semantic_intent.confidence < 0.5:
            return False
        if self.semantic_intent.requires_human_review:
            return False
            
        return True
    
    @property
    def primary_intent(self) -> Optional[IntentType]:
        """Get the strongest/first intent for compatibility."""
        if self.semantic_intent and self.semantic_intent.intents:
            return self.semantic_intent.intents[0]
        return None
    
    @classmethod
    def from_failure(
        cls,
        reason: ClassificationFailureReason,
        details: str,
        source: str = "provider"
    ) -> ClassificationResult:
        """Create a failed classification result."""
        return cls(
            success=False,
            failure_reason=reason,
            failure_details=details,
            source=source
        )
    
    @classmethod
    def from_semantic_intent(
        cls,
        intent: SemanticIntent,
        source: str = "llm",
        model_used: Optional[str] = None
    ) -> ClassificationResult:
        """Create a successful classification result."""
        return cls(
            success=True,
            semantic_intent=intent,
            source=source,
            model_used=model_used
        )


@dataclass
class GroundingValidation:
    """
    Validation of generated content against verified context.
    
    Prevents the LLM from inventing facts not in approved context.
    """
    
    is_grounded: bool
    unsupported_claims: list[str] = field(default_factory=list)
    supported_claims: list[str] = field(default_factory=list)
    confidence: float = 0.0
    validation_notes: str = ""
    
    @property
    def is_safe_to_send(self) -> bool:
        """Should this message be sent or require human review?"""
        return self.is_grounded and len(self.unsupported_claims) == 0
