"""
Planner: decides WHAT the agent should try to accomplish next.

This module used to be a 428-line rule-based policy engine (fixed
priority chain: meeting request > pricing > information request >
objection > interest > ...) that was never actually called by any
production code path -- its only reference anywhere in the codebase was
a dead import-check inside a health endpoint. That's exactly the "Zheimon
orphan architecture" failure mode: a subsystem that exists, is
internally coherent, and is completely disconnected from the real
system. See docs/FINAL_PRODUCTION_READINESS.md for how that was found.

This rewrite does two things differently:

1. It's a genuine reasoning step, not a bigger if/elif tree. Planning --
   "given this business objective, this prospect's own goal, and what we
   currently understand, what's the single most useful next action?" --
   is exactly the kind of judgment call a fixed priority chain gets
   wrong in ways that are hard to enumerate in advance (see this
   module's docstring history for the removed
   "positive_interest -> pricing_discussed" style shortcuts). So the
   planner is LLM-backed, structurally separate from classification
   (semantic/classifier.py answers "what did they mean?"; this answers
   "what should we do about it?") and structurally separate from wording
   (llm/agent.py's Responder answers "how do we say it?").

2. It is actually called by mail/reply_handler_v2.py's real reply path.
   No new subsystem ships without a production caller in this codebase
   going forward -- that's the whole point of this rewrite.

The planner PROPOSES. It is not the safety authority: policy/guardrails.py
authorizes or rejects the proposal deterministically before anything is
drafted or sent, and grounding/suppression checks downstream are
unaffected by anything the planner said. See guardrails.py's docstring
for that boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from mailer_agent.llm.prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from mailer_agent.llm.provider import LLMOutputError, LLMUnavailableError, call_llm_json, is_llm_available
from mailer_agent.semantic_models import SemanticIntent

logger = logging.getLogger("mailer_agent.policy.next_action")

PLANNER_PROMPT_VERSION = "planner-v1"

# The universal objective for this kind of B2B outreach agent. Not a
# per-campaign DB field (yet) -- campaigns don't currently express a
# distinct objective beyond "have a legitimate commercial conversation
# about value_prop", and adding a new required column for a single
# constant string didn't seem justified. If campaigns ever need
# genuinely different objectives (e.g. "collect a referral" vs "book a
# meeting"), this is the one place that would need to become
# campaign-aware.
DEFAULT_BUSINESS_OBJECTIVE = (
    "Progress toward a qualified next step (a call, meeting, or other "
    "concrete conversation) by being genuinely useful and honest in "
    "response to what the prospect actually says -- never by pushing "
    "toward that step at the cost of relevance or honesty."
)


class ActionType(str, Enum):
    ACKNOWLEDGE = "acknowledge"
    ANSWER = "answer"
    CLARIFY = "clarify"
    ASK_TARGETED_QUESTION = "ask_targeted_question"
    ADDRESS_OBJECTION = "address_objection"
    PROVIDE_REQUESTED_INFORMATION = "provide_requested_information"
    PROPOSE_NEXT_STEP = "propose_next_step"
    REQUEST_MISSING_INFORMATION = "request_missing_information"
    DEFER = "defer"
    NURTURE = "nurture"
    ESCALATE = "escalate"


@dataclass
class NextActionProposal:
    """
    Typed output of the planner. Crossing this typed contract (rather
    than the LLM's raw JSON) is what lets policy/guardrails.py validate
    a fixed, known shape instead of trusting arbitrary output -- see
    guardrails.py.
    """
    action_type: ActionType
    objective: str
    reason: str
    required_information: list[str] = field(default_factory=list)
    confidence: float = 0.0
    requires_human_review: bool = False
    review_reason: Optional[str] = None
    source: str = "llm"  # "llm" or "fallback"


# Fallback proposal used when the LLM is unavailable. Deliberately the
# most conservative possible plan -- it does not guess at what action
# would be useful (that would just be a smaller, hidden version of the
# heuristic-planning problem this module exists to avoid), it defers to
# a human.
_FALLBACK_PROPOSAL = NextActionProposal(
    action_type=ActionType.ESCALATE,
    objective="Have a human review this reply and decide the right response.",
    reason="Planner LLM unavailable -- no safe automated plan.",
    confidence=0.0,
    requires_human_review=True,
    review_reason="Planner unavailable",
    source="fallback",
)


def _semantic_summary(intent: SemanticIntent) -> str:
    """Compact plain-text summary of the classifier's structured output, for the planner prompt."""
    lines = [
        f"intents: {[i.value for i in intent.intents]}",
        f"speech_act: {intent.speech_act.value if intent.speech_act else 'unknown'}",
        f"sentiment: {intent.sentiment.value}",
        f"buying_stage: {intent.buying_stage.value}",
        f"urgency: {intent.urgency.value}",
    ]
    if intent.user_goal:
        lines.append(f"prospect's stated/inferred goal: {intent.user_goal}")
    if intent.objections_raised:
        lines.append(f"objections: {intent.objections_raised}")
    if intent.questions_asked:
        lines.append(f"questions asked: {intent.questions_asked}")
    if intent.requested_information:
        lines.append(f"information requested: {intent.requested_information}")
    if intent.has_pricing_question:
        lines.append("has_pricing_question: true")
    if intent.timing:
        lines.append(
            f"timing: expression={intent.timing.expression!r} "
            f"normalized_target={intent.timing.normalized_target} "
            f"requires_clarification={intent.timing.requires_clarification}"
        )
    if intent.current_solution:
        lines.append(
            f"current_solution: {intent.current_solution.value} "
            f"(certainty={intent.current_solution.certainty.value})"
        )
    lines.append(f"classifier confidence: {intent.confidence}")
    if intent.uncertain_aspects:
        lines.append(f"classifier-flagged uncertainty: {intent.uncertain_aspects}")
    return "\n".join(lines)


def plan_next_action(
    *,
    intent: SemanticIntent,
    context_transcript: str,
    known_facts: str = "",
    business_objective: str = DEFAULT_BUSINESS_OBJECTIVE,
) -> NextActionProposal:
    """
    Propose the next conversational action. Never raises -- returns the
    conservative ESCALATE fallback if the LLM is unavailable or fails,
    since a wrong guess about what to do next is worse than asking a
    human (this mirrors llm/agent.py's ContextualFallbackUnavailable
    reasoning, but the planner has a real fallback value to return where
    the responder deliberately does not, because "have a human look at
    it" is itself always a valid plan, unlike invented reply content).
    """
    if not is_llm_available():
        logger.info("Planner: LLM unavailable, returning conservative escalate fallback")
        return _FALLBACK_PROPOSAL

    try:
        human_prompt = build_planner_prompt(
            business_objective=business_objective,
            prospect_goal=intent.user_goal,
            semantic_summary=_semantic_summary(intent),
            known_facts=known_facts,
            unresolved_items="\n".join(intent.unresolved_items) if intent.unresolved_items else "",
            context_transcript=context_transcript,
        )
        payload = call_llm_json(PLANNER_SYSTEM_PROMPT, human_prompt, max_tokens=400, temperature=0.3)

        try:
            action_type = ActionType(payload.get("action_type"))
        except ValueError:
            logger.warning("Planner returned unknown action_type %r, escalating", payload.get("action_type"))
            return _FALLBACK_PROPOSAL

        objective = (payload.get("objective") or "").strip()
        if not objective:
            logger.warning("Planner returned empty objective, escalating")
            return _FALLBACK_PROPOSAL

        return NextActionProposal(
            action_type=action_type,
            objective=objective,
            reason=payload.get("reason", ""),
            required_information=payload.get("required_information", []) or [],
            confidence=float(payload.get("confidence", 0.5)),
            requires_human_review=bool(payload.get("requires_human_review", False)),
            review_reason=payload.get("review_reason"),
            source="llm",
        )
    except (LLMUnavailableError, LLMOutputError) as e:
        logger.warning("Planner LLM call failed, escalating: %s", e)
        return _FALLBACK_PROPOSAL
