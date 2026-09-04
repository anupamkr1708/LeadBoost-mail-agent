"""
SalesAgent: the decision-making + drafting core.

One entry point used by the rest of the system:
  - draft_message(...)   -- write the next outbound email (initial /
                             follow-up / reply / closing push)

Prefers the LLM path. If the LLM is unavailable or returns something
unusable, behavior depends on what kind of message this is:

  - initial_outreach / follow_up: falls back to a single deterministic
    path -- not a library of per-industry templates, just one honest,
    data-driven message built from the same campaign/contact fields the
    LLM would have used. These are safe to auto-generate without the LLM
    because they don't need to interpret anything the prospect said --
    there's no specific claim/question/objection in view to get wrong.

  - reply / closing (anything responding to a specific prospect
    message): raises ContextualFallbackUnavailable instead of inventing
    content. A reply has to be *about* what the prospect actually said,
    and without the LLM there is no way to know that -- generating a
    generic "let's hop on a call" reply regardless of whether the
    prospect asked a pricing question, raised an objection, or asked for
    a referral is not a safe degraded mode, it's a wrong answer that
    happens to be well-formed English. Callers (mail/reply_handler_v2.py)
    catch this and route to human review with no auto-generated body,
    rather than persisting a draft that looks ready to approve but isn't
    actually responsive to anything.

After every draft that IS produced (LLM or initial/follow-up fallback),
grounding validation runs synchronously before the result is returned.
The validation uses only keyword/regex heuristics (no LLM) and never
blocks the pipeline -- it sets AgentDraft.grounding so callers can decide
whether to auto-send or route to human review.

Note on inbound classification: reply intent is handled by
semantic/classifier.py (classify_prospect_reply), which produces a
multi-dimensional SemanticIntent rather than a single label. An earlier,
single-intent classify_reply() used to live in this module; it was never
called anywhere once the semantic classifier replaced it, so it has been
removed rather than kept as unused dead code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from mailer_agent.llm.grounding import validate_grounding
from mailer_agent.llm.prompts import (
    AGENT_SYSTEM_PROMPT,
    build_action_instruction,
    build_context_block,
)
from mailer_agent.llm.provider import LLMOutputError, LLMUnavailableError, call_llm_json, is_llm_available
from mailer_agent.models import Campaign, Contact
from mailer_agent.semantic_models import GroundingValidation

logger = logging.getLogger("mailer_agent.agent")

# Action types that respond to something specific the prospect said --
# these have no safe deterministic fallback (see module docstring).
# Everything else (initial_outreach, follow_up) is an agent-initiated
# nudge that doesn't need to interpret prospect content, so a grounded
# deterministic fallback remains safe for those.
CONTEXTUAL_REPLY_ACTION_TYPES = {"reply", "closing"}


class ContextualFallbackUnavailable(Exception):
    """
    Raised by draft_message() instead of inventing reply content when
    the LLM path failed/is unavailable AND action_type is one that
    requires understanding what the prospect said (see
    CONTEXTUAL_REPLY_ACTION_TYPES). Callers must route to human review,
    not substitute a generic CTA.
    """
    def __init__(self, action_type: str, cause: Exception | None = None):
        self.action_type = action_type
        self.cause = cause
        super().__init__(
            f"No safe deterministic fallback for action_type={action_type!r} "
            f"-- this message must respond to something specific the prospect "
            f"said, which requires the LLM. Human must draft this reply."
        )


@dataclass
class AgentDraft:
    subject: str | None
    body: str
    reasoning: str
    source: str  # "llm" or "fallback"
    # Grounding validation result — always set, never None after draft_message()
    grounding: Optional[GroundingValidation] = field(default=None)


def draft_message(
    *,
    campaign: Campaign,
    contact: Contact,
    action_type: str,
    context_transcript: str,
    days_waited: int | None = None,
    planner_objective: str | None = None,
    planner_reason: str | None = None,
) -> AgentDraft:
    """
    Draft the next outbound message and validate it against approved sources.

    `planner_objective`/`planner_reason`: the specific objective from
    policy/next_action.py's Planner, when this draft is a reply
    (mail/reply_handler_v2.py always runs the planner first for replies
    and passes its output through here). This is the PLAN/WORDING
    separation: the planner already decided *what* this message should
    accomplish; this function (the Responder) only decides *how* to say
    it. Not used for initial_outreach/follow_up, which are
    agent-initiated and don't go through the planner.

    Grounding validation runs on every draft regardless of whether the
    LLM or the deterministic fallback produced it.  Callers inspect
    ``draft.grounding.is_safe_to_send`` to decide whether to auto-send or
    hold for human approval.

    Raises ContextualFallbackUnavailable (see class docstring) instead of
    returning a draft when action_type needs the LLM to be safe and the
    LLM path isn't available -- callers must handle this explicitly, not
    just take whatever AgentDraft comes back.
    """
    if is_llm_available():
        try:
            draft = _draft_with_llm(
                campaign, contact, action_type, context_transcript, days_waited,
                planner_objective=planner_objective, planner_reason=planner_reason,
            )
        except (LLMUnavailableError, LLMOutputError) as e:
            if action_type in CONTEXTUAL_REPLY_ACTION_TYPES:
                logger.warning(
                    "LLM drafting failed for contact %s (%s) -- no safe "
                    "fallback for a contextual reply, raising for human "
                    "review instead of generating a generic response: %s",
                    contact.id, action_type, e,
                )
                raise ContextualFallbackUnavailable(action_type, cause=e) from e
            logger.warning(
                "LLM drafting failed for contact %s (%s), using deterministic fallback: %s",
                contact.id, action_type, e,
            )
            draft = _draft_fallback(campaign, contact, action_type)
    else:
        if action_type in CONTEXTUAL_REPLY_ACTION_TYPES:
            logger.warning(
                "LLM unavailable and action_type=%r requires it -- no safe "
                "fallback, raising for human review.",
                action_type,
            )
            raise ContextualFallbackUnavailable(action_type)
        draft = _draft_fallback(campaign, contact, action_type)

    # --- Grounding validation (always runs, no LLM involved) ---
    draft.grounding = _run_grounding(draft, campaign, contact, context_transcript)

    if not draft.grounding.is_safe_to_send:
        logger.info(
            "Contact %s draft flagged by grounding validation: %s",
            contact.id,
            draft.grounding.validation_notes,
        )

    return draft


def _run_grounding(
    draft: AgentDraft,
    campaign: Campaign,
    contact: Contact,
    context_transcript: str,
) -> GroundingValidation:
    """Run grounding checks; never raises — always returns a result."""
    try:
        return validate_grounding(
            draft.body,
            proof_points=campaign.proof_points,
            context_notes=contact.context_notes,
            conversation_transcript=context_transcript,
            value_prop=campaign.value_prop,
        )
    except Exception as exc:  # pragma: no cover — defensive catch
        logger.error("Grounding validation raised unexpectedly: %s", exc)
        # Conservative: treat unexpected errors as ungrounded
        return GroundingValidation(
            is_grounded=False,
            unsupported_claims=["grounding-check-error"],
            validation_notes=f"Grounding check raised: {exc}",
            confidence=0.0,
        )


def _draft_with_llm(
    campaign: Campaign,
    contact: Contact,
    action_type: str,
    context_transcript: str,
    days_waited: int | None,
    *,
    planner_objective: str | None = None,
    planner_reason: str | None = None,
) -> AgentDraft:
    instruction = build_action_instruction(
        action_type, contact.follow_up_index + 1, days_waited,
        planner_objective=planner_objective, planner_reason=planner_reason,
    )
    human_prompt = (
        f"{build_context_block(campaign, contact)}\n\n"
        f"Task: {instruction}\n\n"
        f"Conversation history:\n{context_transcript}"
    )
    payload = call_llm_json(AGENT_SYSTEM_PROMPT, human_prompt, max_tokens=650)

    body = (payload.get("body") or "").strip()
    if not body:
        raise LLMOutputError("LLM returned an empty body")

    subject = (payload.get("subject") or "").strip() or None
    if not subject:
        # The prompt always asks for a subject now (rule 9) -- this is a
        # pure safety net for the rare case the model omits it anyway,
        # not the primary path.
        subject = f"{campaign.sender_org} <> {contact.company or contact.name or 'quick question'}"

    return AgentDraft(
        subject=subject,
        body=body,
        reasoning=payload.get("reasoning", ""),
        source="llm",
    )


def _draft_fallback(campaign: Campaign, contact: Contact, action_type: str) -> AgentDraft:
    """
    Single, non-branching safety net for agent-initiated messages
    (initial outreach / follow-up nudges) that don't need to interpret
    anything the prospect said. Uses only real campaign/contact fields --
    no per-industry copy.

    Never called for action_type in CONTEXTUAL_REPLY_ACTION_TYPES --
    draft_message() raises ContextualFallbackUnavailable for those
    instead (see module docstring for why a reply specifically has no
    safe fallback).
    """
    name = contact.name or "there"
    company = contact.company or "your company"
    sender = campaign.sender_name
    org = campaign.sender_org
    offer = campaign.value_prop.strip()

    if action_type == "initial_outreach":
        subject = f"{org} <> {company}"
        body = (
            f"Hi {name},\n\n"
            f"I'm {sender} with {org}. {offer}\n\n"
            f"Worth a short call to see if it's relevant for {company}?\n\n"
            f"Best,\n{sender}\n{org}"
        )
    elif action_type == "follow_up":
        subject = None
        body = (
            f"Hi {name},\n\n"
            f"Following up on my last note -- still think this could be worth "
            f"a quick conversation for {company}. {offer}\n\n"
            f"Happy to work around your schedule if you're open to a short call.\n\n"
            f"Best,\n{sender}\n{org}"
        )
    else:
        # Should be unreachable: draft_message() raises
        # ContextualFallbackUnavailable before calling this function for
        # any action_type outside {initial_outreach, follow_up}. Fails
        # loudly rather than silently generating unsafe content if that
        # invariant is ever violated by a future change.
        raise AssertionError(
            f"_draft_fallback called with action_type={action_type!r}, which "
            f"has no safe deterministic fallback and should have raised "
            f"ContextualFallbackUnavailable in draft_message() instead."
        )

    logger.info("Used deterministic fallback draft for contact %s (%s)", contact.id, action_type)
    return AgentDraft(subject=subject, body=body, reasoning="LLM unavailable; deterministic fallback used.", source="fallback")
