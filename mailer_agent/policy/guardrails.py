"""
Guardrails: deterministic authorization of the planner's proposed action.

The planner (policy/next_action.py) proposes what to do. This module is
the safety authority that decides whether that proposal may actually be
auto-sent, or must be held for human review -- plain Python boolean
logic, no LLM call, no semantic interpretation of its own. It only ever
asks yes/no questions about things that are already known/structured
(is auto-reply enabled, is confidence above a threshold, did the planner
itself flag this as needing review, is this an action category we've
decided is never safe to auto-send).

What this module deliberately does NOT do: check suppression or run
grounding validation. Those remain exactly where they already were
(mail/reply_handler_v2.py checks suppression immediately before sending;
llm/agent.py runs grounding on every draft; api/messages.py re-checks
both at approval time) -- duplicating them here would just create two
places that could drift out of sync. This module answers one narrower
question: "is this category of action, from this planner with this
confidence, ever eligible for auto-send at all?" Suppression and
grounding are independent, unconditional gates that still apply
regardless of what this module decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mailer_agent.policy.next_action import ActionType, NextActionProposal
from mailer_agent.semantic_models import SemanticIntent

# Action categories that are never eligible for auto-send, regardless of
# confidence or settings -- a direct, action-driven replacement for the
# old ALWAYS_REQUIRE_APPROVAL_INTENTS set (which gated on raw intent
# type; this gates on what the planner actually decided to DO, which is
# the more meaningful signal once a planner exists at all).
NEVER_AUTO_SEND_ACTIONS = {
    ActionType.ESCALATE,        # planner itself said this needs a human
    ActionType.ADDRESS_OBJECTION,  # objections deserve a human's judgment call
}

MIN_AUTO_SEND_CONFIDENCE = 0.6


@dataclass
class AuthorizedAction:
    proposal: NextActionProposal
    can_auto_send: bool
    review_reason: Optional[str]
    # Maps the planner's ActionType onto the existing drafting entry
    # point's action_type parameter (llm.agent.draft_message expects
    # "reply" or "closing" -- see llm/prompts.py's build_action_instruction).
    # "closing" is used only when the planner explicitly proposed
    # PROPOSE_NEXT_STEP -- a real decision the planner made, not a
    # raw intent+confidence heuristic re-derived here.
    draft_action_type: str


def authorize_action(
    proposal: NextActionProposal,
    *,
    intent: SemanticIntent,
    auto_reply_enabled: bool,
) -> AuthorizedAction:
    """
    Decide whether `proposal` may be auto-sent once drafted, or must be
    held for human approval. This is checked BEFORE drafting (to decide
    what kind of draft to even write) -- the actual send path still
    independently re-checks suppression and grounding immediately before
    any external side effect, regardless of what this function returns.
    """
    draft_action_type = "closing" if proposal.action_type == ActionType.PROPOSE_NEXT_STEP else "reply"

    reasons: list[str] = []

    if not auto_reply_enabled:
        reasons.append("AUTO_REPLY_ENABLED is false")

    if intent.confidence < MIN_AUTO_SEND_CONFIDENCE:
        reasons.append(f"Classifier confidence too low ({intent.confidence:.2f})")

    if intent.requires_human_review:
        reasons.append(intent.human_review_reason or "Semantic analysis requires human review")

    if proposal.confidence < MIN_AUTO_SEND_CONFIDENCE:
        reasons.append(f"Planner confidence too low ({proposal.confidence:.2f})")

    if proposal.requires_human_review:
        reasons.append(proposal.review_reason or "Planner requires human review")

    if proposal.action_type in NEVER_AUTO_SEND_ACTIONS:
        reasons.append(f"Action type {proposal.action_type.value} always requires human review")

    if proposal.action_type == ActionType.PROVIDE_REQUESTED_INFORMATION and intent.has_pricing_question:
        reasons.append("Pricing questions are never auto-answered without human review")

    return AuthorizedAction(
        proposal=proposal,
        can_auto_send=len(reasons) == 0,
        review_reason="; ".join(reasons) if reasons else None,
        draft_action_type=draft_action_type,
    )
