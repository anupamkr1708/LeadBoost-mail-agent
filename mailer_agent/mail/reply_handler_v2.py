"""
Enhanced reply handler with semantic intelligence.

Improvements over original:
1. Uses multi-dimensional semantic classification
2. Explicit failure handling (provider failures ≠ semantic neutrality)
3. State machine-driven transitions
4. Message deduplication
5. Proper timestamp tracking
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone

from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mailer_agent.config import get_settings
from mailer_agent.llm.agent import ContextualFallbackUnavailable, draft_message
from mailer_agent.mail.imap_reader import InboundEmail, as_reply_subject
from mailer_agent.mail.sender import SendOutcome, send_email
from mailer_agent.memory.store import build_conversation_context, maybe_summarize_older_messages
from mailer_agent.policy.guardrails import authorize_action
from mailer_agent.policy.next_action import PLANNER_PROMPT_VERSION, plan_next_action
from mailer_agent.models import (
    Contact,
    ContactStatus,
    Message,
    MessageDirection,
    MessageStatus,
    MessageType,
    SuppressionEntry,
    contact_is_integration_managed,
)
from mailer_agent.semantic.classifier import classify_prospect_reply
from mailer_agent.semantic_models import IntentType, attach_fact_provenance, serialize_semantic_intent
from mailer_agent.state_machine import (
    StateTransitionEvent,
    infer_event_from_semantic_intent,
    transition_contact_state,
)
from mailer_agent.utils.datetime_utils import utcnow

logger = logging.getLogger("mailer_agent.mail.reply_handler_v2")
settings = get_settings()


def _build_references_header(
    parent_references: list[str], parent_in_reply_to: str | None, parent_message_id: str | None
) -> str | None:
    """
    RFC 5322 References for a reply: the parent's own References chain
    (its ancestors), then the parent's own In-Reply-To, then the
    parent's own Message-ID -- in order, with no duplicates.

    In-Reply-To is included as well as References because real mail
    clients don't always populate References even when they do set
    In-Reply-To (Gmail's first reply in a thread, in particular, sends
    In-Reply-To with References empty) -- using References alone would
    silently drop the immediate ancestor from the chain in exactly that
    case. The immediate parent (parent_message_id) stays the last entry
    here (and is separately set as In-Reply-To by the caller) -- this
    just makes sure the rest of the ancestor chain isn't dropped on the
    next hop.

    Deterministic and protocol-level: never derived from anything the
    LLM wrote.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for msg_id in (*parent_references, parent_in_reply_to, parent_message_id):
        if msg_id and msg_id not in seen:
            seen.add(msg_id)
            ordered.append(msg_id)
    return " ".join(ordered) if ordered else None


def _normalize_message_id(value: str | None) -> str | None:
    """
    Canonicalize a Message-ID-like header value to RFC 5322's
    angle-bracketed form, e.g. "abc@x" and "<abc@x>" both become
    "<abc@x>".

    Why this matters: IMAP polling extracts the raw `Message-ID` header
    via Python's email module, which normally keeps the angle brackets
    intact. A webhook payload has no such guarantee -- different
    inbound-parse providers (and hand-rolled payload templates) may or
    may not include them, and may add stray whitespace. Without
    normalizing at the one point both channels converge
    (process_inbound_email_v2), the *same physical email* delivered via
    webhook and then again via IMAP fallback (or vice versa) could be
    dedup-mismatched purely due to string formatting, and get processed
    -- and possibly auto-replied to -- twice.
    """
    if not value:
        return value
    v = value.strip()
    if not v:
        return None
    if not v.startswith("<"):
        v = f"<{v}"
    if not v.endswith(">"):
        v = f"{v}>"
    return v


def process_inbound_email_v2(db: Session, email_in: InboundEmail) -> dict:
    """
    Enhanced inbound email processing with semantic intelligence.
    
    Returns dict with processing result.

    This is the one convergence point for both inbound channels
    (api/webhooks.py and the IMAP poll job in followup/scheduler.py both
    call through mail.reply_handler.process_inbound_email, which is a
    thin pass-through to this function) -- so Message-ID normalization
    happens here, once, rather than duplicated in each channel's own
    parsing code.
    """
    # Normalize threading identifiers before anything else touches them,
    # so dedup (Step 1) and threading correlation (Step 2) both compare
    # apples to apples regardless of which channel this arrived through.
    normalized_message_id = _normalize_message_id(email_in.message_id)
    normalized_in_reply_to = _normalize_message_id(email_in.in_reply_to)
    normalized_references = [
        r for r in (_normalize_message_id(ref) for ref in email_in.references) if r
    ]
    email_in = replace(
        email_in,
        message_id=normalized_message_id,
        in_reply_to=normalized_in_reply_to,
        references=normalized_references,
    )

    # Step 1: Deduplicate by Message-ID
    if email_in.message_id:
        existing = db.query(Message).filter(
            Message.message_id_header == email_in.message_id,
            Message.direction == MessageDirection.INBOUND.value
        ).first()
        
        if existing:
            logger.info(
                f"Duplicate inbound email detected (Message-ID: {email_in.message_id}), skipping"
            )
            return {
                "matched": True,
                "contact_id": existing.contact_id,
                "action": "skipped_duplicate",
                "message_id": existing.id
            }
    
    # Step 2: Correlate to contact
    contact = _find_contact_by_message_id(db, email_in) or _find_contact_by_email(db, email_in)
    
    if not contact:
        logger.info(
            "No matching contact for inbound email from %s (to=%s) -- "
            "either genuinely unknown or ambiguous correlation (see warning "
            "above if this was an ambiguous-email-match case); skipped, not persisted.",
            email_in.from_email, email_in.to_email,
        )
        return {"matched": False, "from_email": email_in.from_email}
    
    campaign = contact.campaign
    
    # Step 3: Persist inbound message
    inbound_msg = Message(
        contact_id=contact.id,
        direction=MessageDirection.INBOUND.value,
        message_type=None,
        subject=email_in.subject,
        body=email_in.body_text,
        status=MessageStatus.RECEIVED.value,
        message_id_header=email_in.message_id,
        in_reply_to_header=email_in.in_reply_to,
        references_header=" ".join(email_in.references) if email_in.references else None,
    )
    db.add(inbound_msg)
    try:
        db.flush()
    except IntegrityError:
        # The Step 1 check-then-insert dedup check above has a race
        # window under true concurrency: two transactions can both see
        # "not found" before either commits (e.g. the same email
        # arriving via webhook and IMAP nearly simultaneously). The
        # uq_messages_message_id_header constraint (see models.py,
        # migrations/003_message_id_unique_constraint.py) is what
        # actually closes that race -- this except clause is what makes
        # hitting it a graceful "someone else already handled this"
        # instead of an unhandled 500. Roll back this transaction (now
        # aborted after the IntegrityError) and re-query for the row the
        # winning transaction inserted.
        db.rollback()
        logger.info(
            "Concurrent duplicate detected for Message-ID %s (lost the "
            "race to insert, another transaction won) -- treating as "
            "already-handled, not an error.",
            email_in.message_id,
        )
        existing = db.query(Message).filter(
            Message.message_id_header == email_in.message_id,
            Message.direction == MessageDirection.INBOUND.value
        ).first()
        return {
            "matched": True,
            "contact_id": existing.contact_id if existing else contact.id,
            "action": "skipped_duplicate",
            "message_id": existing.id if existing else None,
        }
    
    # Update contact timestamps
    contact.last_reply_at = utcnow()
    
    # Step 4: Semantic classification
    context_transcript = build_conversation_context(db, contact)
    
    classification_result = classify_prospect_reply(
        campaign=campaign,
        contact=contact,
        inbound_body=email_in.body_text,
        conversation_context=context_transcript,
        auto_submitted=email_in.auto_submitted,
    )
    
    # Store classification result in message
    inbound_msg.classification_success = classification_result.success
    
    if classification_result.success and classification_result.semantic_intent:
        intent = classification_result.semantic_intent

        # Deterministic provenance stamp (spec section 14) -- must happen
        # here, not in the classifier: this is the one place that
        # actually knows the inbound Message's real database id and
        # observation time. The LLM only ever fills `supersedes` (see
        # semantic/classifier.py's SUPERSESSION rule) -- it has no
        # visibility into row ids or wall-clock time and must never be
        # asked to invent them.
        attach_fact_provenance(
            intent,
            source_message_id=inbound_msg.id,
            observed_at=utcnow().isoformat(),
        )

        # Native dict into the JSON column -- semantic_analysis is
        # Column(JSON), and SQLAlchemy's JSON type handles
        # serialization itself. Passing json.dumps(...) here used to
        # double-encode it (a JSON string *containing* a JSON string),
        # which breaks any JSON-path query against the column and means
        # any future reader has to json.loads() a value that should
        # already be a dict.
        inbound_msg.semantic_analysis = serialize_semantic_intent(intent)
        
        # Backward compatibility: set detected_intent to primary intent
        inbound_msg.detected_intent = intent.intents[0].value if intent.intents else "neutral"
        inbound_msg.intent_confidence = intent.confidence
        
        # Update contact buying stage
        contact.buying_stage = intent.buying_stage.value
        
    else:
        # Classification failed
        inbound_msg.classification_failure_reason = (
            classification_result.failure_reason.value if classification_result.failure_reason else "unknown"
        )
        inbound_msg.detected_intent = "unknown"
        inbound_msg.intent_confidence = 0.0
    
    result = {
        "matched": True,
        "contact_id": contact.id,
        "classification_success": classification_result.success,
    }
    
    # Step 5: Handle based on classification
    if not classification_result.success:
        # Classification failed - require human review
        logger.warning(
            f"Classification failed for contact {contact.id}: "
            f"{classification_result.failure_reason} - routing to human review"
        )
        
        transition_contact_state(
            contact,
            StateTransitionEvent.NEEDS_HUMAN,
            reason=f"Classification failed: {classification_result.failure_reason}"
        )
        contact.next_action_at = None
        result["action"] = "needs_review_classification_failed"
        
    elif not classification_result.semantic_intent:
        # Should not happen if success=True, but handle gracefully
        logger.error(f"Classification success but no semantic_intent for contact {contact.id}")
        transition_contact_state(contact, StateTransitionEvent.NEEDS_HUMAN, reason="Missing semantic intent")
        result["action"] = "needs_review_missing_intent"
        
    else:
        # Successful classification
        intent = classification_result.semantic_intent
        result["primary_intent"] = intent.intents[0].value if intent.intents else "neutral"
        result["confidence"] = intent.confidence
        
        # Infer state transition event
        event = infer_event_from_semantic_intent(intent)
        
        # Handle terminal intents
        if IntentType.UNSUBSCRIBE in intent.intents:
            _handle_unsubscribe(db, contact, result)
            
        elif IntentType.NOT_INTERESTED in intent.intents:
            transition_contact_state(contact, StateTransitionEvent.NOT_INTERESTED)
            contact.next_action_at = None
            result["action"] = "closed_lost"
            
        elif IntentType.OUT_OF_OFFICE in intent.intents:
            # No state change, no action - just wait
            result["action"] = "ignored_oos"
            
        elif contact_is_integration_managed(contact):
            # LeadBoost-owned contact. The reply is stored, classified and
            # state-tracked above/here, but Mailer-native automation must not
            # turn it into an outbound email nobody authorized: no native
            # follow-up rescheduling, no drafting, and in particular no
            # automatic send -- AUTO_REPLY_ENABLED does not apply to this
            # contact. next_action_at is pinned to None so the contact can
            # never become native-scheduler-eligible. The bilateral reply
            # contract is C9.3.
            transition_contact_state(contact, event, reason=f"Semantic: {intent.reasoning[:100]}")
            contact.next_action_at = None
            result["action"] = "recorded_integration_managed"

        else:
            # Regular reply - transition state
            transition_contact_state(contact, event, reason=f"Semantic: {intent.reasoning[:100]}")
            
            # Reschedule follow-up based on conversation context
            from mailer_agent.followup.conversation_aware import FollowUpScheduler
            scheduler = FollowUpScheduler()
            contact.next_action_at = scheduler.reschedule_after_reply(
                contact, campaign, intent
            )
            
            # Draft reply
            _draft_and_maybe_send_reply(
                db, contact, campaign, email_in, intent, result,
                classification_source=classification_result.source,
                classification_model_used=classification_result.model_used,
                classification_used_fallback=classification_result.used_fallback,
                classification_response_mode=classification_result.response_mode,
                inbound_message_id=inbound_msg.id,
            )
    
    db.add(contact)
    db.add(inbound_msg)
    db.flush()
    
    # Step 6: Update conversation memory
    maybe_summarize_older_messages(db, contact)
    
    return result


def _find_contact_by_message_id(db: Session, email_in: InboundEmail) -> Contact | None:
    """Find contact by threading headers (reliable method)."""
    candidate_ids = set()
    if email_in.in_reply_to:
        candidate_ids.add(email_in.in_reply_to)
    candidate_ids.update(email_in.references)
    
    if not candidate_ids:
        return None
    
    msg = (
        db.query(Message)
        .filter(Message.message_id_header.in_(candidate_ids))
        .order_by(Message.created_at.desc())
        .first()
    )
    
    return msg.contact if msg else None


def _find_contact_by_email(db: Session, email_in: InboundEmail) -> Contact | None:
    """
    Find contact by email address (fallback method, used only when
    threading-header correlation in _find_contact_by_message_id fails).

    Tenant-safe by construction, not by convention:
    - If to_email is known (webhook told us which inbox received this,
      or IMAP polled a specific mailbox), resolution is deterministic:
      joined to Campaign.sender_email == to_email, which scopes to
      exactly one organization before any row is even considered.
    - If to_email is NOT known, this never guesses across multiple
      candidates. If more than one contact anywhere matches from_email,
      correlation is genuinely ambiguous (could be two different
      organizations that both have a contact with this address) and
      this returns None -- unresolved -- rather than silently picking
      the most-recently-updated one, which is what this function used
      to do and which risked attributing a reply to the wrong tenant.
      The caller (process_inbound_email_v2) treats None the same as "no
      match": the inbound email is not persisted against any contact,
      logged, and left for human/ops follow-up instead of guessed at.
    """
    query = db.query(Contact).filter(
        Contact.email == email_in.from_email,
        Contact.status != ContactStatus.SUPPRESSED.value,
    )

    if email_in.to_email:
        from mailer_agent.models import Campaign
        # to_email scopes this to (at most) one organization before we
        # pick among its contacts, so a most-recently-updated tiebreaker
        # here is choosing among that ONE org's own re-contacts of the
        # same address (e.g. a re-run campaign) -- not guessing across
        # tenants.
        query = (
            query.join(Campaign, Contact.campaign_id == Campaign.id)
            .filter(Campaign.sender_email == email_in.to_email)
        )
        return query.order_by(Contact.updated_at.desc()).first()

    # No to_email available: we cannot resolve to a specific
    # organization ourselves. Only proceed if the match is already
    # unambiguous (exactly one contact anywhere has this email) --
    # otherwise this is exactly the cross-tenant collision scenario
    # that must not be guessed at.
    candidates = query.all()
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logger.warning(
            "Ambiguous inbound correlation: %d contacts match from_email=%s "
            "with no to_email or thread-header to disambiguate -- treating "
            "as unresolved rather than guessing which organization owns it.",
            len(candidates), email_in.from_email,
        )
    return None


def _handle_unsubscribe(db: Session, contact: Contact, result: dict):
    """Handle unsubscribe request."""
    # Add to suppression list if not already there
    # CRITICAL: Include organization_id to properly scope suppression
    from mailer_agent.models import Campaign
    campaign = db.query(Campaign).filter_by(id=contact.campaign_id).first()
    org_id = campaign.organization_id if campaign else None
    
    existing = (
        db.query(SuppressionEntry)
        .filter_by(email=contact.email, organization_id=org_id)
        .first()
    )
    
    if not existing:
        db.add(SuppressionEntry(
            email=contact.email,
            organization_id=org_id,
            reason="unsubscribed"
        ))
    
    # Transition to suppressed
    transition_contact_state(contact, StateTransitionEvent.UNSUBSCRIBED)
    contact.next_action_at = None
    
    result["action"] = "suppressed"


def _build_known_facts_summary(intent, contact: Contact, campaign) -> str:
    """
    Facts for the planner prompt: what's actually APPROVED to draw from
    (the campaign's value_prop/proof_points and the contact's
    context_notes -- the same sources llm/agent.py's Responder and
    grounding.py's validator treat as authoritative) plus accumulated
    conversation-wide facts (memory.store.build_known_facts_context,
    pulled from every prior classified inbound message's stored
    semantic_analysis) plus anything new from THIS turn's intent that
    hasn't been persisted yet at the point this runs.

    The approved-content block matters specifically for
    PLANNER_SYSTEM_PROMPT's rule 2 ("only propose
    provide_requested_information if it's the kind of thing that could
    plausibly be answered from approved campaign materials") -- without
    it, the planner was being asked to judge plausibility with no view
    of what's actually approved, so it could only guess. It now sees the
    same approved facts the Responder and grounding gate already use, so
    its own instruction has something to check itself against instead
    of assuming.

    Deduplicated by simple text match -- this-turn facts are appended
    after the accumulated ones so the planner sees the most current
    information last.
    """
    from mailer_agent.memory.store import build_known_facts_context

    approved_lines = [
        f"value proposition: {campaign.value_prop}",
        "proof points: "
        + (campaign.proof_points or "(none approved -- nothing beyond the value proposition above is available)"),
        "approved facts about this contact: " + (contact.context_notes or "(none recorded)"),
    ]
    approved = "Approved campaign/contact content (the only source of business facts a reply may draw from):\n" + "\n".join(
        approved_lines
    )

    accumulated = build_known_facts_context(contact)
    this_turn_lines = []
    if intent.current_solution:
        this_turn_lines.append(
            f"current_solution (this message): {intent.current_solution.value} "
            f"(certainty={intent.current_solution.certainty.value})"
        )
    for fact in intent.new_facts:
        this_turn_lines.append(f"fact (this message): {fact.value} (certainty={fact.certainty.value})")
    if intent.contradicted_facts:
        this_turn_lines.append(f"contradicts earlier (this message): {intent.contradicted_facts}")

    parts = [p for p in (approved, accumulated, "\n".join(this_turn_lines)) if p]
    return "\n".join(parts)


def _draft_and_maybe_send_reply(
    db: Session,
    contact: Contact,
    campaign,
    email_in: InboundEmail,
    intent,
    result: dict,
    *,
    classification_source: str = "unknown",
    classification_model_used: str | None = None,
    classification_used_fallback: bool | None = None,
    classification_response_mode: str | None = None,
    inbound_message_id: int | None = None,
):
    """
    Plan, draft, and (if authorized) send a reply.

    Pipeline: Planner (LLM proposes what to do) -> Guardrails
    (deterministic authorization of that proposal) -> Responder
    (llm.agent.draft_message writes the actual wording, grounded in the
    planner's objective) -> suppression/grounding gates (independent,
    unconditional, checked regardless of what the planner/guardrails
    decided) -> send or hold for approval.

    This replaced a single inline line (`action_type = "closing" if
    POSITIVE_INTEREST and confidence >= 0.7 else "reply"`) plus a fixed
    ALWAYS_REQUIRE_APPROVAL_INTENTS set-membership check. Both are gone
    now, not because they were exactly wrong, but because they were the
    seed of exactly the kind of hardcoded, intent-keyed branching this
    architecture is supposed to avoid -- see policy/next_action.py and
    policy/guardrails.py for where that reasoning now lives instead.
    """
    context_transcript = build_conversation_context(db, contact)

    # --- Planner: what should this message accomplish? ---
    proposal = plan_next_action(
        intent=intent,
        context_transcript=context_transcript,
        known_facts=_build_known_facts_summary(intent, contact, campaign),
    )

    # --- Guardrails: is that proposal eligible for auto-send? ---
    authorized = authorize_action(proposal, intent=intent, auto_reply_enabled=settings.auto_reply_enabled)
    action_type = authorized.draft_action_type

    logger.info(
        "Contact %s planner proposal: action_type=%s objective=%r "
        "confidence=%.2f source=%s | guardrail: can_auto_send=%s reason=%r",
        contact.id, proposal.action_type.value, proposal.objective,
        proposal.confidence, proposal.source,
        authorized.can_auto_send, authorized.review_reason,
    )

    # --- Responder: write the actual wording, grounded in the plan. ---
    # draft_message() raises ContextualFallbackUnavailable instead of
    # returning a draft when the LLM is unavailable/failed -- a reply
    # has to respond to something specific the prospect said, and there
    # is no safe deterministic way to fabricate that (see
    # llm/agent.py's module docstring). No draft body is invented here:
    # route straight to human review instead of persisting a generic,
    # not-actually-responsive message that a reviewer might approve
    # without noticing it doesn't address what was asked.
    try:
        draft = draft_message(
            campaign=campaign,
            contact=contact,
            action_type=action_type,
            context_transcript=context_transcript,
            planner_objective=proposal.objective,
            planner_reason=proposal.reason,
        )
    except ContextualFallbackUnavailable as e:
        logger.warning(
            "Contact %s reply requires human drafting -- no LLM-generated "
            "content available: %s", contact.id, e,
        )
        transition_contact_state(
            contact,
            StateTransitionEvent.NEEDS_HUMAN,
            reason=f"LLM unavailable for contextual reply (action_type={action_type})",
        )
        result["action"] = "reply_needs_manual_draft"
        result["approval_reason"] = (
            "LLM was unavailable, so no automated draft could be safely "
            "generated for this reply -- a human must write it from scratch."
        )
        return result

    # Grounding gate: if unsupported claims found, force draft status
    # regardless of the planner/guardrail decision. Independent of, and
    # downstream of, everything above -- a well-authorized action can
    # still produce an ungrounded draft if the model invents a detail
    # while executing an otherwise-fine plan.
    grounding_blocked = draft.grounding and not draft.grounding.is_safe_to_send
    if grounding_blocked:
        logger.info(
            "Contact %s reply held for grounding review: %s",
            contact.id, draft.grounding.validation_notes,
        )

    # Suppression gate: every other send path (initial outreach,
    # follow-up, and the manual-approval endpoint) checks suppression
    # immediately before the external send call -- checked here too,
    # immediately before deciding whether to auto-send, independent of
    # the planner/guardrail decision above.
    from mailer_agent.followup.engine import is_suppressed
    suppressed = is_suppressed(db, contact.email, org_id=campaign.organization_id)
    if suppressed:
        logger.info(
            "Contact %s reply held for review instead of auto-sent: address is suppressed",
            contact.id,
        )

    can_auto_send = authorized.can_auto_send and not grounding_blocked and not suppressed

    # Subject and threading are transport/application-layer facts, not
    # wording -- the responder proposes draft.subject for content it
    # originates (initial outreach, follow-ups), but a contextual reply
    # stays in the prospect's existing thread deterministically. Letting
    # an LLM-generated subject replace it would silently start a new
    # thread in the recipient's mail client even though In-Reply-To/
    # References still point at the right parent (see docs for the
    # observed live-E2E case this fixes). Do not use draft.subject here.
    reply_subject = as_reply_subject(email_in.subject)
    reply_references = _build_references_header(
        email_in.references, email_in.in_reply_to, email_in.message_id
    )

    # Create reply message
    reply_msg = Message(
        contact_id=contact.id,
        direction=MessageDirection.OUTBOUND.value,
        message_type=MessageType.REPLY.value if action_type == "reply" else MessageType.CLOSING.value,
        subject=reply_subject,
        body=draft.body,
        status=MessageStatus.DRAFT.value,
        in_reply_to_header=email_in.message_id,
        references_header=reply_references,
    )

    if can_auto_send:
        # Auto-send
        send_result = send_email(
            to_email=contact.email,
            from_email=campaign.sender_email,
            from_name=campaign.sender_name,
            subject=reply_subject,
            body_text=draft.body,
            reply_to=campaign.reply_to_email,
            in_reply_to_header=email_in.message_id,
            references_header=reply_references,
        )

        # Status mirrors send_result.outcome directly (sent/failed/unknown)
        # -- an ambiguous SMTP outcome must never be recorded as a plain
        # "failed" that a caller might treat as safe to blindly retry.
        reply_msg.status = send_result.outcome.value
        reply_msg.message_id_header = send_result.message_id
        reply_msg.error_message = send_result.error

        if send_result.success:
            contact.last_outbound_at = utcnow()
        elif send_result.outcome == SendOutcome.UNKNOWN:
            # Ambiguous delivery -- escalate instead of leaving the
            # contact in a state where an automatic follow-up cycle
            # could duplicate a reply that may have already gone out.
            transition_contact_state(
                contact,
                StateTransitionEvent.NEEDS_HUMAN,
                reason=f"Auto-reply send outcome unknown: {send_result.error}",
            )
            contact.next_action_at = None

        result["action"] = {
            "sent": "auto_replied",
            "failed": "auto_reply_failed",
            "unknown": "auto_reply_unknown",
        }[send_result.outcome.value]
        result["message_id"] = reply_msg.id
        result["planner_action_type"] = proposal.action_type.value

    else:
        # Requires human approval
        result["action"] = "reply_drafted_awaiting_approval"
        result["message_id"] = reply_msg.id
        result["planner_action_type"] = proposal.action_type.value
        if suppressed:
            result["approval_reason"] = "Contact is on the suppression list"
        elif grounding_blocked and draft.grounding:
            result["approval_reason"] = (
                f"Grounding: {draft.grounding.validation_notes}"
            )
        else:
            result["approval_reason"] = authorized.review_reason or "Requires review"

    db.add(reply_msg)

    # Full structured trace of this turn -- see observability.py. Logged
    # here rather than at each intermediate step so it captures the
    # actual final outcome, not just the plan.
    from mailer_agent.observability import build_and_log_turn_trace
    build_and_log_turn_trace(
        contact_id=contact.id,
        campaign_id=campaign.id,
        inbound_body=email_in.body_text,
        inbound_message_id=inbound_message_id,
        outbound_message_id=reply_msg.id,
        classification_source=classification_source,
        classification_success=True,
        semantic_intent_dict=serialize_semantic_intent(intent),
        classifier_model_used=classification_model_used,
        classifier_used_fallback=classification_used_fallback,
        classifier_response_mode=classification_response_mode,
        prompt_version_planner=PLANNER_PROMPT_VERSION,
        planner_action_type=proposal.action_type.value,
        planner_objective=proposal.objective,
        planner_confidence=proposal.confidence,
        planner_source=proposal.source,
        planner_model_used=proposal.model_used,
        planner_used_fallback=proposal.used_fallback,
        planner_response_mode=proposal.response_mode,
        guardrail_can_auto_send=authorized.can_auto_send,
        guardrail_review_reason=authorized.review_reason,
        draft_source=draft.source,
        grounding_safe=draft.grounding.is_safe_to_send if draft.grounding else None,
        grounding_notes=draft.grounding.validation_notes if draft.grounding else None,
        draft_model_used=draft.model_used,
        draft_used_fallback=draft.used_fallback,
        draft_response_mode=draft.response_mode,
        final_action=result.get("action"),
    )
