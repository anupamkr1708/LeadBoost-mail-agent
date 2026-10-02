"""
LeadBoost generated-outreach intake (C9.2).

    POST /integrations/leadboost/outreach-requests

LeadBoost has already authorized the outreach action. Mailer receives the
verified business context for that one action, generates the message with its
existing drafting stack (llm.agent.draft_message, which also runs the existing
grounding validator), and durably records ONE Message + ONE ExternalDispatch.
The C6-C8 worker (mail/external_dispatch_worker.py) still owns delivery; the
C9.1 reconciliation GET still observes it.

Why a separate module from api/integrations.py
----------------------------------------------
integrations.py holds the exact-message route and the C9.1 read-only
reconciliation handler, and a frozen C9.1 test asserts that module imports
nothing from the LLM / follow-up / sender stack. This route necessarily
imports the drafting stack, so it lives here. It reuses (imports) the
campaign/contact get-or-create helpers and the integration-source constant
from integrations.py rather than duplicating them.

What this endpoint is NOT
-------------------------
Not a second campaign API and not a way to drive Mailer-native behaviour: no
sender/SMTP/mailbox/credential, subject/body, tone, proof_points, action type,
cadence or tenant field exists in the request (schemas.py, extra="forbid").
Sender identity is the deployment-level integration sender, via the existing
integration Campaign.

Request-local generation context
--------------------------------
The offer (value_proposition), recipient_facts and recipient title/company are
fed to draft_message() through TRANSIENT, never-persisted Campaign/Contact
instances. They are never written to the shared integration Campaign or
Contact rows, so concurrent requests cannot see each other's context. The
only durable copy is ExternalDispatch.grounding_context, an immutable
per-dispatch snapshot the worker grounds against:

    Request A context -> Message A -> ExternalDispatch A -> grounding A

Conversation history is Mailer's own: build_conversation_context() over the
Message table (empty for a new contact). Nothing is accepted from the caller.

TRANSITIONAL: synchronous generation
------------------------------------
The LLM is called inside the request, before the dispatch is accepted, so a
request can take as long as one LLM call and a provider outage surfaces as an
error to the caller (who retries with the same idempotency_key). That is a
known limitation of this stage, not the target design: M1/M2 move generation
into mailbox-owned durable work. It is deliberately NOT solved here with a new
queue, and the dispatch worker deliberately does not import the LLM.

Concurrency and idempotency
---------------------------
Identity is (organization_id, idempotency_key) -- DB unique constraint -- and
the operation fingerprint is built only from the caller's external_action_id
and recipient email, never from generated text, so a retry that regenerates a
different body is still the same operation. Two identical concurrent requests
may each generate a draft (wasted LLM work, bounded to the race window); only
one final commit can win, the loser rolls back its Message and returns the
winner's dispatch. A replay of an accepted key never calls the LLM.

Transaction boundaries match the exact-message route: Campaign and Contact are
reusable get-or-create rows committed on their own; Message + ExternalDispatch
(with its grounding snapshot) are created in ONE final commit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mailer_agent.api.deps import get_integration_org_id, require_api_key
from mailer_agent.api.integrations import (
    _get_or_create_integration_campaign,
    _get_or_create_integration_contact,
)
from mailer_agent.db import get_db
from mailer_agent.llm.agent import draft_message
from mailer_agent.mail.exact_message import create_authorized_message
from mailer_agent.memory.store import build_conversation_context
from mailer_agent.models import (
    Campaign,
    Contact,
    ExternalDispatch,
    ExternalDispatchState,
)
from mailer_agent.schemas import (
    LeadBoostOutreachActionAccepted,
    LeadBoostOutreachRequestIn,
)

logger = logging.getLogger("mailer_agent.api.integrations_generated")

router = APIRouter(
    prefix="/integrations/leadboost",
    tags=["integrations"],
    dependencies=[Depends(require_api_key)],
)

GROUNDING_CONTEXT_VERSION = 1

_IDEMPOTENCY_CONFLICT_DETAIL = (
    "idempotency_key has already been used with a different request. "
    "Use a new idempotency_key for a genuinely new action."
)


def _compute_generated_request_fingerprint(*, external_action_id: str, recipient_email: str) -> str:
    """
    SHA-256 over the canonical JSON of the operation's identity: which
    LeadBoost action, for which recipient. Stable across safe retries.

    Deliberately excludes everything generated or descriptive (subject, body,
    value_proposition, recipient_facts, name/title/company, correlation_id):
    the same logical operation must keep the same fingerprint however its
    message was worded or its context refined on a retry. The "kind" tag keeps
    these fingerprints disjoint from the exact-message route's, so a key
    first used there can never read as a replay here.
    """
    canonical = json.dumps(
        {
            "kind": "leadboost.generated_outreach.v1",
            "external_action_id": external_action_id,
            "recipient_email": recipient_email,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _find_dispatch(db: Session, org_id: str, idempotency_key: str) -> ExternalDispatch | None:
    # Tenant predicate lives in the query itself, never fetch-then-check.
    return (
        db.query(ExternalDispatch)
        .filter(
            ExternalDispatch.organization_id == org_id,
            ExternalDispatch.idempotency_key == idempotency_key,
        )
        .first()
    )


def _accepted(dispatch: ExternalDispatch) -> LeadBoostOutreachActionAccepted:
    return LeadBoostOutreachActionAccepted(
        accepted=True, mailing_agent_reference=dispatch.public_reference
    )


def _replay_or_conflict(
    existing: ExternalDispatch, fingerprint: str
) -> LeadBoostOutreachActionAccepted:
    if existing.request_fingerprint == fingerprint:
        return _accepted(existing)
    raise HTTPException(status_code=409, detail=_IDEMPOTENCY_CONFLICT_DETAIL)


def _format_recipient_facts(facts: list[str]) -> str | None:
    """Same free-text shape the existing prompt already presents as
    "Verified facts about this contact/company" (Contact.context_notes)."""
    if not facts:
        return None
    return "\n".join(f"- {fact}" for fact in facts)


@router.post(
    "/outreach-requests",
    response_model=LeadBoostOutreachActionAccepted,
    status_code=202,
)
def create_leadboost_generated_outreach(
    payload: LeadBoostOutreachRequestIn,
    org_id: str = Depends(get_integration_org_id),
    db: Session = Depends(get_db),
):
    """
    Accept an already-authorized LeadBoost outreach action, generate its
    message from the supplied context, and durably queue it. Sends nothing.

    1. org_id from the API key only (fail-closed dependency); payload strictly
       validated (unknown fields -> 422).
    2. Idempotency lookup scoped to org_id: same fingerprint -> return the
       existing operation (no LLM, no mutation); different -> 409.
    3. get-or-create the integration Campaign, then the Contact (own short
       commits; a reused Contact with an active follow-up -> 409, nothing
       mutated). Request context is NOT written to either row.
    4. Generate via draft_message() with transient Campaign/Contact carrying
       THIS request's context. A draft the existing grounding validator
       hard-blocks is not accepted (422, nothing persisted): a message that
       would be failed unsent by the worker is never queued.
    5. ONE commit creates Message + ExternalDispatch (incl. the immutable
       grounding snapshot). The unique (organization_id, idempotency_key)
       constraint is the race backstop.
    """
    email = str(payload.recipient.email)
    fingerprint = _compute_generated_request_fingerprint(
        external_action_id=payload.external_action_id, recipient_email=email
    )

    existing = _find_dispatch(db, org_id, payload.idempotency_key)
    if existing is not None:
        logger.info(
            "Idempotent replay: org=%s idempotency_key=%s (no LLM, no mutation)",
            org_id, payload.idempotency_key,
        )
        return _replay_or_conflict(existing, fingerprint)

    campaign = _get_or_create_integration_campaign(db, org_id)
    contact = _get_or_create_integration_contact(db, campaign.id, email, payload.recipient.name)

    # Everything the rest of the request needs, as plain values, so no ORM
    # state is relied on across the (possibly slow) LLM call below.
    campaign_id, contact_id = campaign.id, contact.id
    sender = {
        "sender_name": campaign.sender_name,
        "sender_org": campaign.sender_org,
        "sender_email": campaign.sender_email,
        "sender_title": campaign.sender_title,
        "tone": campaign.tone,
    }
    # Mailer's own conversation history (new contact -> the first-contact
    # placeholder). Never caller-supplied.
    transcript = build_conversation_context(db, contact)
    # Release the read transaction before a potentially multi-second LLM call.
    db.rollback()

    context_notes = _format_recipient_facts(payload.context.recipient_facts)
    value_prop = payload.context.value_proposition

    # Transient, never added to a session: request-local generation context.
    gen_campaign = Campaign(
        name="LeadBoost generated outreach (transient)",
        value_prop=value_prop,
        proof_points=None,
        **sender,
    )
    gen_contact = Contact(
        name=payload.recipient.name,
        email=email,
        title=payload.recipient.title,
        company=payload.recipient.company,
        context_notes=context_notes,
        follow_up_index=0,
    )

    try:
        draft = draft_message(
            campaign=gen_campaign,
            contact=gen_contact,
            action_type="initial_outreach",
            context_transcript=transcript,
        )
    except Exception:
        logger.exception(
            "Generation failed for org=%s idempotency_key=%s", org_id, payload.idempotency_key
        )
        raise HTTPException(
            status_code=503,
            detail="Message generation failed; nothing was queued. Retry with the same idempotency_key.",
        )

    if draft.grounding is None or draft.grounding.hard_block:
        # Same predicate the worker gates on. Notes are counted, not echoed.
        logger.warning(
            "Generated draft rejected by grounding for org=%s idempotency_key=%s: %s",
            org_id, payload.idempotency_key,
            draft.grounding.validation_notes if draft.grounding else "no grounding result",
        )
        raise HTTPException(
            status_code=422,
            detail=(
                "A grounded message could not be generated from the supplied context; "
                "nothing was queued. Check that the context supports every claim."
            ),
        )

    message = create_authorized_message(
        contact_id=contact_id, subject=draft.subject, body=draft.body
    )
    db.add(message)
    db.flush()  # assign message.id without committing

    dispatch = ExternalDispatch(
        organization_id=org_id,
        idempotency_key=payload.idempotency_key,
        external_action_id=payload.external_action_id,
        correlation_id=payload.correlation_id,
        campaign_id=campaign_id,
        contact_id=contact_id,
        message_id=message.id,
        request_fingerprint=fingerprint,
        public_reference=uuid.uuid4().hex,
        state=ExternalDispatchState.QUEUED.value,
        grounding_context={
            "version": GROUNDING_CONTEXT_VERSION,
            "value_prop": value_prop,
            "context_notes": context_notes,
            "conversation_transcript": transcript,
        },
    )
    db.add(dispatch)

    try:
        db.commit()
    except IntegrityError:
        # Lost a concurrent race on (organization_id, idempotency_key): the
        # Message flushed above is rolled back with everything else in this
        # transaction; converge on the winner's row.
        db.rollback()
        winner = _find_dispatch(db, org_id, payload.idempotency_key)
        if winner is None:
            raise
        logger.info(
            "Concurrent request converged on existing dispatch: org=%s idempotency_key=%s",
            org_id, payload.idempotency_key,
        )
        return _replay_or_conflict(winner, fingerprint)

    db.refresh(dispatch)
    logger.info(
        "Accepted LeadBoost generated outreach: org=%s external_action_id=%s "
        "correlation_id=%s -> dispatch=%s contact=%s message=%s source=%s",
        org_id, payload.external_action_id, payload.correlation_id,
        dispatch.public_reference, contact_id, message.id, draft.source,
    )
    return _accepted(dispatch)
