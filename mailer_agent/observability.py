"""
Observability and replay for one semantic processing turn.

Not an event-sourcing system, not a new persistence layer -- this is
exactly what the spec asked for and no more: a structured snapshot of
one inbound-reply turn (classification -> plan -> guardrail decision ->
draft -> send/hold), logged at INFO level with enough fields to answer
"why did the agent do that?" after the fact, and shaped so the same
snapshot could be fed back through the pipeline later for a deterministic
replay (same inputs, same FakeLLMProvider fixtures reconstructed from
the recorded outputs) -- e.g. to check whether a prompt change would
have produced a different result for a past real conversation.

This module only builds and logs the trace; it doesn't decide anything.
Wired into mail/reply_handler_v2.py::_draft_and_maybe_send_reply, which
is the one place per inbound reply where every piece of this snapshot is
already available.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger("mailer_agent.observability")


@dataclass
class TurnTrace:
    """
    One inbound-reply turn, end to end. Every field here is either a
    plain value or something already JSON-serializable (semantic_intent/
    planner_proposal are passed in as the dicts semantic_models.
    serialize_semantic_intent() and the planner's own dataclass ->dict
    conversion produce, not the dataclass instances themselves) so this
    can be logged directly as structured data.
    """
    # Identity -- how to find this turn again later.
    contact_id: int
    campaign_id: int
    inbound_message_id: Optional[int] = None
    outbound_message_id: Optional[int] = None

    # What came in.
    inbound_body_excerpt: str = ""  # truncated -- see build_turn_trace(); never the full raw body in logs

    # Classification (semantic/classifier.py).
    classification_source: Optional[str] = None  # "llm" | "rule_based" | "metadata" | "fallback"
    classification_success: Optional[bool] = None
    semantic_intent: Optional[dict] = None  # serialize_semantic_intent() output
    prompt_version_classifier: Optional[str] = None

    # Planning (policy/next_action.py).
    planner_action_type: Optional[str] = None
    planner_objective: Optional[str] = None
    planner_confidence: Optional[float] = None
    planner_source: Optional[str] = None
    prompt_version_planner: Optional[str] = None

    # Guardrails (policy/guardrails.py).
    guardrail_can_auto_send: Optional[bool] = None
    guardrail_review_reason: Optional[str] = None

    # Drafting/grounding (llm/agent.py).
    draft_source: Optional[str] = None  # "llm" | "fallback"
    grounding_safe: Optional[bool] = None
    grounding_notes: Optional[str] = None

    # Final outcome.
    final_action: Optional[str] = None  # result["action"] from process_inbound_email_v2

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_and_log_turn_trace(
    *,
    contact_id: int,
    campaign_id: int,
    inbound_body: str,
    inbound_message_id: Optional[int] = None,
    outbound_message_id: Optional[int] = None,
    classification_source: Optional[str] = None,
    classification_success: Optional[bool] = None,
    semantic_intent_dict: Optional[dict] = None,
    prompt_version_classifier: Optional[str] = None,
    planner_action_type: Optional[str] = None,
    planner_objective: Optional[str] = None,
    planner_confidence: Optional[float] = None,
    planner_source: Optional[str] = None,
    prompt_version_planner: Optional[str] = None,
    guardrail_can_auto_send: Optional[bool] = None,
    guardrail_review_reason: Optional[str] = None,
    draft_source: Optional[str] = None,
    grounding_safe: Optional[bool] = None,
    grounding_notes: Optional[str] = None,
    final_action: Optional[str] = None,
) -> TurnTrace:
    """
    Build a TurnTrace and log it at INFO. Never raises -- observability
    must not be able to break the pipeline it's observing.

    inbound_body is truncated to 200 chars for the log line (this is
    business email content, not a secret, but there's no reason to put
    full prospect message bodies in application logs when an excerpt is
    enough to recognize which turn this was -- see spec's "do not
    persist sensitive raw data unnecessarily").
    """
    trace = TurnTrace(
        contact_id=contact_id,
        campaign_id=campaign_id,
        inbound_message_id=inbound_message_id,
        outbound_message_id=outbound_message_id,
        inbound_body_excerpt=(inbound_body or "")[:200],
        classification_source=classification_source,
        classification_success=classification_success,
        semantic_intent=semantic_intent_dict,
        prompt_version_classifier=prompt_version_classifier,
        planner_action_type=planner_action_type,
        planner_objective=planner_objective,
        planner_confidence=planner_confidence,
        planner_source=planner_source,
        prompt_version_planner=prompt_version_planner,
        guardrail_can_auto_send=guardrail_can_auto_send,
        guardrail_review_reason=guardrail_review_reason,
        draft_source=draft_source,
        grounding_safe=grounding_safe,
        grounding_notes=grounding_notes,
        final_action=final_action,
    )
    try:
        logger.info("turn_trace contact_id=%s campaign_id=%s trace=%s", contact_id, campaign_id, trace.as_dict())
    except Exception:  # pragma: no cover -- observability must never break the pipeline
        logger.exception("Failed to log turn trace for contact %s", contact_id)
    return trace
