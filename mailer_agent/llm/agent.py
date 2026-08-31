"""
SalesAgent: the decision-making + drafting core.

Two entry points used by the rest of the system:
  - draft_message(...)   -- write the next outbound email (initial /
                             follow-up / reply / closing push)
  - classify_reply(...)  -- figure out what an inbound reply actually means

Both prefer the LLM path. If the LLM is unavailable or returns something
unusable, both fall back to a single deterministic path -- not a library
of per-industry templates, just one honest, data-driven message built
from the same campaign/contact fields the LLM would have used, so a
transient outage degrades gracefully instead of blocking the send.

After every draft (LLM or fallback), grounding validation runs
synchronously before the result is returned.  The validation uses only
keyword/regex heuristics (no LLM) and never blocks the pipeline -- it
sets AgentDraft.grounding so callers can decide whether to auto-send or
route to human review.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from mailer_agent.llm.grounding import validate_grounding
from mailer_agent.llm.prompts import (
    AGENT_SYSTEM_PROMPT,
    REPLY_CLASSIFIER_SYSTEM_PROMPT,
    build_action_instruction,
    build_classifier_prompt,
    build_context_block,
)
from mailer_agent.llm.provider import LLMOutputError, LLMUnavailableError, call_llm_json, is_llm_available
from mailer_agent.models import Campaign, Contact
from mailer_agent.semantic_models import GroundingValidation

logger = logging.getLogger("mailer_agent.agent")

VALID_INTENTS = {
    "interested",
    "question",
    "objection",
    "not_interested",
    "unsubscribe",
    "out_of_office",
    "neutral",
}


@dataclass
class AgentDraft:
    subject: str | None
    body: str
    reasoning: str
    source: str  # "llm" or "fallback"
    # Grounding validation result — always set, never None after draft_message()
    grounding: Optional[GroundingValidation] = field(default=None)


@dataclass
class ReplyClassification:
    intent: str
    confidence: float
    reasoning: str
    source: str


def draft_message(
    *,
    campaign: Campaign,
    contact: Contact,
    action_type: str,
    context_transcript: str,
    days_waited: int | None = None,
) -> AgentDraft:
    """
    Draft the next outbound message and validate it against approved sources.

    Grounding validation runs on every draft regardless of whether the
    LLM or the deterministic fallback produced it.  Callers inspect
    ``draft.grounding.is_safe_to_send`` to decide whether to auto-send or
    hold for human approval.
    """
    if is_llm_available():
        try:
            draft = _draft_with_llm(campaign, contact, action_type, context_transcript, days_waited)
        except (LLMUnavailableError, LLMOutputError) as e:
            logger.warning(
                "LLM drafting failed for contact %s (%s), using fallback: %s",
                contact.id, action_type, e,
            )
            draft = _draft_fallback(campaign, contact, action_type)
    else:
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
) -> AgentDraft:
    human_prompt = (
        f"{build_context_block(campaign, contact)}\n\n"
        f"Task: {build_action_instruction(action_type, contact.follow_up_index + 1, days_waited)}\n\n"
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
    Single, non-branching safety net. Uses only real campaign/contact
    fields -- no per-industry copy. This exists so an LLM outage
    degrades to "plain but true" rather than blocking the pipeline.
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
    else:  # reply / closing -- can't safely auto-generate contextual reply content
        subject = None
        body = (
            f"Hi {name},\n\n"
            f"Thanks for getting back to me. I'd love to continue this over a "
            f"quick call so I can answer your questions properly -- do you have "
            f"15 minutes this week?\n\n"
            f"Best,\n{sender}\n{org}"
        )

    logger.info("Used deterministic fallback draft for contact %s (%s)", contact.id, action_type)
    return AgentDraft(subject=subject, body=body, reasoning="LLM unavailable; deterministic fallback used.", source="fallback")


def classify_reply(
    *, campaign: Campaign, contact: Contact, inbound_body: str, context_transcript: str
) -> ReplyClassification:
    if is_llm_available():
        try:
            human_prompt = build_classifier_prompt(inbound_body, context_transcript)
            payload = call_llm_json(REPLY_CLASSIFIER_SYSTEM_PROMPT, human_prompt, max_tokens=150)
            intent = payload.get("intent", "neutral")
            if intent not in VALID_INTENTS:
                intent = "neutral"
            confidence = float(payload.get("confidence", 0.5))
            return ReplyClassification(
                intent=intent,
                confidence=confidence,
                reasoning=payload.get("reasoning", ""),
                source="llm",
            )
        except (LLMUnavailableError, LLMOutputError) as e:
            logger.warning("Reply classification failed for contact %s: %s", contact.id, e)

    # Fallback: crude keyword check, purely so an unsubscribe request is
    # never silently dropped even if the LLM is down. Everything else
    # defaults to "neutral" -> routed to human review by the caller.
    lowered = inbound_body.lower()
    if any(w in lowered for w in ("unsubscribe", "remove me", "stop emailing", "opt out", "opt-out")):
        return ReplyClassification(
            intent="unsubscribe", confidence=0.9, reasoning="Keyword match (LLM unavailable).", source="fallback"
        )
    return ReplyClassification(
        intent="neutral", confidence=0.2, reasoning="LLM unavailable; defaulted to neutral for human review.", source="fallback"
    )
