"""
Structured semantic analysis models for sales intelligence.

These models represent the multi-dimensional understanding of prospect
communication, separating semantic interpretation from provider failures.

Design notes (Phase 2 semantic architecture)
---------------------------------------------
Two changes from the earlier, flatter version of this module, both
driven by concrete production defects found by tracing real call paths
rather than by inspection alone:

1. Certainty is now explicit and travels with individual facts, not just
   the overall classification. `current_solution`, `competitors_mentioned`,
   and every entry in `new_facts` carry a `Certainty` (EXPLICIT /
   STRONGLY_INFERRED / WEAKLY_INFERRED / UNKNOWN) rather than being
   flattened into plain strings the rest of the system might mistake for
   verified facts. This directly answers "explicit vs inferred" -- a
   fact the prospect stated outright ("we use Salesforce") and a
   conclusion the model drew from context ("migration effort is probably
   their real blocker") are represented differently, not collapsed into
   the same shape.

2. `requested_timing` used to be a bare string ("next month", "18
   months", "sometime next quarter maybe") that a downstream module
   (followup/conversation_aware.py) regex-matched against a hardcoded
   phrase list, falling back to a hardcoded 30-day default when nothing
   matched. That's exactly the kind of semantic heuristic this
   architecture is supposed to avoid, and it silently invented a
   business-meaningful date out of nothing. `requested_timing` is now a
   structured `TimingSignal`: the LLM (which actually understands
   natural language) produces `normalized_target` when it can determine
   one, and marks `requires_clarification=True` when it can't --
   deterministic Python downstream only ever *consumes* that decision,
   it never re-derives it from regex.

What this module deliberately does NOT do: force a value into every
field. `Certainty.UNKNOWN` and `None` are first-class, common results,
not error states -- an email that doesn't mention timing has no timing
signal to report, and that's different from a timing signal the model
failed to parse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Core bounded ontology -- the small, stable vocabulary the rest of the
# system (state machine, planner, scheduling) is allowed to branch on.
# Anything more specific than this belongs in the free-form/structured
# knowledge fields on SemanticIntent (objections_raised, entities,
# new_facts, etc), not as a new enum member every time a new phrasing
# shows up.
# ---------------------------------------------------------------------------

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


class SpeechAct(str, Enum):
    """
    What kind of communicative act this message performs -- distinct from
    intent (what it's about). "Can we do a call?" and "We should do a
    call" can share a topic (meeting) but are a QUESTION vs a STATEMENT;
    conflating them loses information the planner needs (a question
    expects an answer, a statement doesn't).
    """
    STATEMENT = "statement"
    QUESTION = "question"
    REQUEST = "request"
    COMMITMENT = "commitment"
    REJECTION = "rejection"
    ACKNOWLEDGMENT = "acknowledgment"


class SentimentType(str, Enum):
    """Overall sentiment/tone. Supplementary, not a proxy for intent or interest."""
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


class Certainty(str, Enum):
    """
    How confident the interpretation is that a given fact/belief is true
    -- distinct from IntentType/SentimentType, which describe *what* was
    communicated. This describes *how sure we are* about a specific piece
    of derived knowledge.

    EXPLICIT: the prospect stated this directly ("we use Salesforce").
    STRONGLY_INFERRED: not stated outright, but a confident reading of
      clear context ("switching would be painful" -> switching_risk
      concern).
    WEAKLY_INFERRED: a plausible but speculative reading ("therefore
      migration effort is probably their primary blocker").
    UNKNOWN: cannot be determined from what's available. This is a valid,
      common result -- not an error.
    """
    EXPLICIT = "explicit"
    STRONGLY_INFERRED = "strongly_inferred"
    WEAKLY_INFERRED = "weakly_inferred"
    UNKNOWN = "unknown"


class TimingKind(str, Enum):
    """What shape of timing expression this is."""
    SPECIFIC_DATE = "specific_date"      # "March 15th", "next Tuesday"
    RELATIVE_PERIOD = "relative_period"  # "next month", "in two weeks"
    QUARTER = "quarter"                  # "Q3", "next quarter"
    CONDITIONAL = "conditional"          # "after our renewal", "once budget opens"
    VAGUE = "vague"                      # "sometime", "eventually", "down the road"


class ClassificationFailureReason(str, Enum):
    """Explicit failure reasons (not semantic states)."""
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    MALFORMED_OUTPUT = "malformed_output"
    PARSE_ERROR = "parse_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"


# ---------------------------------------------------------------------------
# Structured value types
# ---------------------------------------------------------------------------

@dataclass
class TimingSignal:
    """
    Structured interpretation of a timing expression, replacing a bare
    string that downstream code would otherwise have to re-parse with
    regex (which is exactly the semantic-heuristic pattern this
    architecture avoids -- see module docstring).

    `normalized_target` is an ISO 8601 date/datetime string ONLY when the
    model can genuinely determine one from context (e.g. campaign start
    date + "in 3 weeks", or an explicit "March 15th"). For anything
    vague, conditional, or ambiguous, it is None and
    `requires_clarification` is True -- deterministic scheduling code
    must not invent a fallback date when this is the case (see
    followup/conversation_aware.py).
    """
    expression: str                                    # raw phrase, e.g. "sometime next quarter"
    kind: Optional[TimingKind] = None
    normalized_target: Optional[str] = None             # ISO date/datetime, only if determinable
    certainty: Certainty = Certainty.UNKNOWN
    commitment_strength: Optional[str] = None            # "weak" | "moderate" | "strong" -- free text by design, not a forced enum
    requires_clarification: bool = False


@dataclass
class SemanticFact:
    """
    A single piece of knowledge extracted from the conversation, with
    provenance. Used for current_solution, competitors_mentioned, and
    entries in new_facts/changed_facts/entities -- anywhere the system
    would otherwise be tempted to store a bare string and quietly treat
    it as verified.

    Provenance fields (spec section 14, "fact provenance"):
    - `supersedes` is filled BY THE LLM (see semantic/classifier.py's
      SUPERSESSION rule) when it recognizes this fact replaces a
      specific earlier-stated value -- e.g. a new current_solution of
      "none -- evaluating replacement" with supersedes="Salesforce".
      This is the model's job because only it can recognize supersession
      from meaning; deterministic code never guesses semantic
      equivalence between two fact strings (see memory/store.py's
      reconcile_known_facts for how this is consumed).
    - `source_message_id` and `observed_at` are NEVER filled by the LLM
      -- it has no visibility into database row IDs or wall-clock time.
      They are attached deterministically, once, immediately after
      classification succeeds -- see attach_fact_provenance() below and
      its call site in mail/reply_handler_v2.py.
    """
    value: str
    certainty: Certainty = Certainty.UNKNOWN
    explicit: bool = False           # True only for Certainty.EXPLICIT; kept as a fast, obvious check
    evidence: Optional[str] = None   # short quote/paraphrase of what supports this
    supersedes: Optional[str] = None       # exact prior fact value this replaces, if any (LLM-provided)
    source_message_id: Optional[int] = None  # Message.id this fact was extracted from (attached deterministically)
    observed_at: Optional[str] = None        # ISO 8601 timestamp, attached deterministically


def attach_fact_provenance(intent: "SemanticIntent", *, source_message_id: int, observed_at: str) -> None:
    """
    Deterministically stamp source_message_id/observed_at onto every
    SemanticFact this SemanticIntent carries (current_solution,
    competitors_mentioned, new_facts), in place. Idempotent -- only
    fills fields that are still None, so calling this twice (or on an
    intent that was somehow pre-stamped) never overwrites an existing
    value with a different one.

    Must be called exactly once, right after classification succeeds and
    before the intent is serialized/persisted (see
    mail/reply_handler_v2.py) -- this is the one place in the pipeline
    that actually knows the inbound Message's real id and timestamp; the
    LLM does not and must never be asked to invent them.
    """
    def _stamp(f: Optional[SemanticFact]) -> None:
        if f is None:
            return
        if f.source_message_id is None:
            f.source_message_id = source_message_id
        if f.observed_at is None:
            f.observed_at = observed_at

    _stamp(intent.current_solution)
    for fact in intent.competitors_mentioned:
        _stamp(fact)
    for fact in intent.new_facts:
        _stamp(fact)


# ---------------------------------------------------------------------------
# The semantic interpretation itself
# ---------------------------------------------------------------------------

@dataclass
class SemanticIntent:
    """
    Multi-dimensional interpretation of a single prospect message, always
    produced in the context of the conversation so far (see
    semantic/classifier.py -- the classifier receives conversation
    history and campaign/contact context, not just the latest message in
    isolation).

    Grouped by the same logical categories the LLM is asked to reason
    about, matching semantic/classifier.py's prompt structure:
    communicative meaning, prospect state, commercial signals,
    conversation content, knowledge, temporal, uncertainty.
    """

    # -- Communicative meaning ---------------------------------------
    intents: list[IntentType] = field(default_factory=list)
    speech_act: Optional[SpeechAct] = None
    sentiment: SentimentType = SentimentType.NEUTRAL

    # -- Prospect state -----------------------------------------------
    buying_stage: BuyingStage = BuyingStage.UNAWARE
    user_goal: Optional[str] = None                      # free text: what the prospect is trying to accomplish right now
    pain_points: list[str] = field(default_factory=list)
    objections_raised: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)  # e.g. "locked into current contract", "needs procurement sign-off"

    # -- Commercial signals ---------------------------------------------
    urgency: UrgencyLevel = UrgencyLevel.NO_TIMELINE
    has_pricing_question: bool = False
    has_budget_signal: bool = False
    has_decision_maker_signal: bool = False
    has_commitment_signal: bool = False
    procurement_signal: bool = False

    # -- Conversation content -------------------------------------------
    requested_information: list[str] = field(default_factory=list)
    questions_asked: list[str] = field(default_factory=list)
    commitments_made: list[str] = field(default_factory=list)   # promises the PROSPECT made ("I'll loop in our IT lead")

    # -- Knowledge (provenance-tracked, never silently treated as verified) --
    current_solution: Optional[SemanticFact] = None
    competitors_mentioned: list[SemanticFact] = field(default_factory=list)
    new_facts: list[SemanticFact] = field(default_factory=list)
    contradicted_facts: list[str] = field(default_factory=list)   # plain-text notes on what this contradicts, if anything
    unresolved_items: list[str] = field(default_factory=list)     # questions/asks that still need an answer

    # -- Temporal ---------------------------------------------------------
    timing: Optional[TimingSignal] = None

    # -- Uncertainty --------------------------------------------------
    confidence: float = 0.0   # 0.0-1.0, overall confidence in this interpretation as a whole
    uncertain_aspects: list[str] = field(default_factory=list)  # plain-text notes on what's genuinely ambiguous
    reasoning: str = ""

    # -- Review/safety ------------------------------------------------
    requires_human_review: bool = False
    human_review_reason: Optional[str] = None

    # -- Backward-compatible convenience accessors ------------------------
    @property
    def requested_timing(self) -> Optional[str]:
        """
        Deprecated convenience accessor for the raw timing expression.
        Prefer `.timing` (a TimingSignal) for anything that needs to
        reason about certainty or a normalized target -- this only
        returns the original phrase, with none of that structure.
        """
        return self.timing.expression if self.timing else None


SEMANTIC_SCHEMA_VERSION = "3"
# v3 (spec sections 14-16, fact provenance): SemanticFact gained
# supersedes/source_message_id/observed_at. Purely additive and
# backward-compatible -- readers of v2 records simply see these three
# keys as None/absent, never misinterpreted as something else. Bumped
# anyway so any future reader that branches on semantic_schema_version
# can tell "this record predates provenance tracking" from "this record
# has provenance fields that are genuinely all null" if that distinction
# ever matters.


def serialize_semantic_intent(intent: SemanticIntent) -> dict:
    """
    Convert a SemanticIntent into a plain JSON-serializable dict, for
    storage in Message.semantic_analysis (a native JSON column -- see
    models.py) and for building conversation context/memory.

    Centralized here (not duplicated at each call site) so every
    consumer -- storage, memory context assembly, observability logging
    -- agrees on one shape. Includes semantic_schema_version so a future
    prompt/schema change doesn't silently reinterpret old stored records
    under a different meaning.
    """
    def _fact(f: Optional[SemanticFact]) -> Optional[dict]:
        if f is None:
            return None
        return {
            "value": f.value,
            "certainty": f.certainty.value,
            "explicit": f.explicit,
            "evidence": f.evidence,
            "supersedes": f.supersedes,
            "source_message_id": f.source_message_id,
            "observed_at": f.observed_at,
        }

    def _timing(t: Optional[TimingSignal]) -> Optional[dict]:
        if t is None:
            return None
        return {
            "expression": t.expression,
            "kind": t.kind.value if t.kind else None,
            "normalized_target": t.normalized_target,
            "certainty": t.certainty.value,
            "commitment_strength": t.commitment_strength,
            "requires_clarification": t.requires_clarification,
        }

    return {
        "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
        "intents": [i.value for i in intent.intents],
        "speech_act": intent.speech_act.value if intent.speech_act else None,
        "sentiment": intent.sentiment.value,
        "buying_stage": intent.buying_stage.value,
        "user_goal": intent.user_goal,
        "pain_points": intent.pain_points,
        "objections_raised": intent.objections_raised,
        "constraints": intent.constraints,
        "urgency": intent.urgency.value,
        "has_pricing_question": intent.has_pricing_question,
        "has_budget_signal": intent.has_budget_signal,
        "has_decision_maker_signal": intent.has_decision_maker_signal,
        "has_commitment_signal": intent.has_commitment_signal,
        "procurement_signal": intent.procurement_signal,
        "requested_information": intent.requested_information,
        "questions_asked": intent.questions_asked,
        "commitments_made": intent.commitments_made,
        "current_solution": _fact(intent.current_solution),
        "competitors_mentioned": [_fact(f) for f in intent.competitors_mentioned],
        "new_facts": [_fact(f) for f in intent.new_facts],
        "contradicted_facts": intent.contradicted_facts,
        "unresolved_items": intent.unresolved_items,
        "timing": _timing(intent.timing),
        "confidence": intent.confidence,
        "uncertain_aspects": intent.uncertain_aspects,
        "reasoning": intent.reasoning,
        "requires_human_review": intent.requires_human_review,
        "human_review_reason": intent.human_review_reason,
    }


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
    prompt_version: Optional[str] = None
    # Model-routing observability (spec: never leave model_used wrong/
    # misleading when a fallback model actually answered the request).
    used_fallback: Optional[bool] = None
    response_mode: Optional[str] = None  # "strict_schema" | "json_object" | "lenient"
    attempts: Optional[int] = None

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
        model_used: Optional[str] = None,
        prompt_version: Optional[str] = None,
        used_fallback: Optional[bool] = None,
        response_mode: Optional[str] = None,
        attempts: Optional[int] = None,
    ) -> ClassificationResult:
        """Create a successful classification result."""
        return cls(
            success=True,
            semantic_intent=intent,
            source=source,
            model_used=model_used,
            prompt_version=prompt_version,
            used_fallback=used_fallback,
            response_mode=response_mode,
            attempts=attempts,
        )


@dataclass
class GroundingValidation:
    """
    Validation of generated content against verified context.

    This is claim-RISK validation, not semantic fact verification: it
    checks whether numeric/quantitative claims in a draft (percentages,
    dollar figures, headcount, guarantee language, pricing/availability
    terms) are supported by text present in the campaign's approved
    proof_points/value_prop or the conversation transcript. It works by
    extracting claim-shaped substrings and confirming the same tokens
    appear in the approved source text -- it cannot judge whether a
    claim is true, only whether it's traceable to something the campaign
    owner actually approved. A draft that paraphrases an approved fact
    using different numbers/wording than the source, or that makes a
    qualitative (non-numeric) unsupported claim, may not be caught.
    Treat this as a floor, not a ceiling: it prevents the most common and
    most damaging failure (inventing a number/guarantee/customer that
    was never approved), not general hallucination.
    """
    
    is_grounded: bool
    unsupported_claims: list[str] = field(default_factory=list)
    supported_claims: list[str] = field(default_factory=list)
    confidence: float = 0.0
    validation_notes: str = ""
    # Sensitive-but-not-necessarily-false content (pricing/availability
    # mentions) that a human reviewer can authorize -- distinct from
    # unsupported_claims, which is reserved for content that is actually
    # untraceable to any approved source. See hard_block/is_safe_to_send.
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)

    @property
    def is_safe_to_send(self) -> bool:
        """Eligible for silent auto-send with no human in the loop."""
        return (
            self.is_grounded
            and len(self.unsupported_claims) == 0
            and not self.review_required
        )

    @property
    def hard_block(self) -> bool:
        """
        True only when a claim is actually untraceable to any approved
        source (a fabricated number, price, guarantee, or customer claim).

        This is deliberately independent of review_required: a draft that
        merely mentions pricing/availability but invents no specific figure
        is review_required (a human should look at it) but NOT hard_block
        (a human CAN authorize sending it). Human approval must never be
        able to turn a hard_block claim into a sendable one -- that
        distinction is enforced by callers checking hard_block, not
        is_safe_to_send, at the approval gate.
        """
        return not self.is_grounded or len(self.unsupported_claims) > 0
