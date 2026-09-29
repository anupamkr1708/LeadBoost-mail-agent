"""
External-integration API boundary (Phase C).

This module is the *only* place an external, already-authorized caller
(currently: LeadBoost) hands off an outreach action to the Mailer Agent.
It is deliberately NOT built on top of any existing endpoint:

  - NOT POST /campaigns/{id}/start -- that's for starting an entire
    human-created campaign's worth of contacts, not accepting one
    already-authorized action.
  - NOT POST /messages/{id}/approve -- that's a second human-approval
    gate. LeadBoost's own OutreachAction.APPROVED state *is* the
    approval; routing this through /messages/{id}/approve a second time
    would recreate the "two competing approval systems" problem this
    design explicitly avoids.

Outcome chain (see ExternalDispatchState in models.py for the full
state machine -- this batch only ever produces QUEUED):

    LeadBoost approved
        != Mailer accepted        <- this endpoint, this batch
        != Mailer worker claimed  <- later phase
        != SMTP attempted         <- later phase
        != SMTP sent               <- later phase
        != recipient received

`accepted=true` in this endpoint's response means only the second of
those six things. Never collapse them.

THIS BATCH (C2-C4) explicitly does NOT:
  - call send_email() (mail/sender.py)
  - call draft_message() or anything under llm/
  - generate a Message-ID
  - claim/dispatch work
A later phase (C5+) adds all of that; this endpoint only durably records
the request.

Tenancy
-------
organization_id is resolved *exclusively* from the authenticated
X-API-Key (get_current_org_id, see api/deps.py) and is never read from
the request body. external_action_id, campaign_id, contact_id -- none of
these select a tenant; they are either pure correlation data or are
themselves scoped by a query that already filters on the authenticated
org_id. See ExternalDispatch's docstring in models.py for why this
matters (the inbound webhook's to_email-based tenant resolution in
api/webhooks.py is the cautionary example of getting this wrong -- a
pre-existing, separately tracked gap this module does not repeat).
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mailer_agent.api.deps import get_current_org_id, require_api_key
from mailer_agent.config import get_settings
from mailer_agent.db import get_db
from mailer_agent.models import (
    Campaign,
    Contact,
    ContactStatus,
    ExternalDispatch,
    ExternalDispatchState,
    Message,
    MessageDirection,
    MessageStatus,
    MessageType,
)
from mailer_agent.schemas import LeadBoostOutreachActionAccepted, LeadBoostOutreachActionIn

logger = logging.getLogger("mailer_agent.api.integrations")

router = APIRouter(
    prefix="/integrations/leadboost",
    tags=["integrations"],
    dependencies=[Depends(require_api_key)],
)

# Fixed value identifying the one, deterministic, get-or-create Campaign
# each organization's LeadBoost-originated sends live under. Not
# user/caller-supplied -- a constant of this integration.
LEADBOOST_INTEGRATION_SOURCE = "leadboost"


# ---------------------------------------------------------------------------
# Idempotency: request fingerprint
# ---------------------------------------------------------------------------

def _compute_request_fingerprint(
    *,
    external_action_id: str | None,
    recipient_email: str,
    recipient_name: str | None,
    subject: str | None,
    body: str,
) -> str:
    """
    SHA-256 hex digest over a canonical (sorted-key, separator-fixed)
    JSON serialization of the fields that define "what this request is
    asking to send". Used to distinguish a safe replay of a known
    idempotency_key (same fingerprint -> return existing operation) from
    caller misuse (different fingerprint -> 409, no mutation).

    Deliberately NOT dependent on correlation_id or idempotency_key
    themselves -- those identify the *operation*, not its content.
    """
    payload = {
        "external_action_id": external_action_id,
        "recipient_email": recipient_email,
        "recipient_name": recipient_name,
        "subject": subject,
        "body": body,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Campaign get-or-create (C4)
# ---------------------------------------------------------------------------

def _get_or_create_integration_campaign(db: Session, org_id: str) -> Campaign:
    """
    Exactly one Campaign per organization carries
    integration_source="leadboost" -- this is the fixed, deterministic
    home for every LeadBoost-originated Contact/Message, never one
    campaign per action. See uq_campaigns_org_integration_source in
    models.py for the DB constraint this relies on as the actual race
    backstop.

    Race-safe by construction, not by timing: query first; if absent,
    attempt an insert and commit it as its own short transaction; if
    that insert loses a race (another concurrent first-use request won),
    the unique index raises IntegrityError, we roll back just that
    attempt, and re-read the winner's row. Exactly one integration
    campaign per org, guaranteed by the database.

    Committing this as its own transaction (rather than folding it into
    the same transaction as the Message/ExternalDispatch write later)
    is deliberate: it keeps this idempotent, reusable get-or-create step
    fully independent of whatever happens afterward in the caller, so a
    later failure elsewhere in the request never needs to roll back a
    Campaign row that's perfectly fine to have created and that the next
    request would just recreate anyway.
    """
    campaign = (
        db.query(Campaign)
        .filter(
            Campaign.organization_id == org_id,
            Campaign.integration_source == LEADBOOST_INTEGRATION_SOURCE,
        )
        .first()
    )
    if campaign is not None:
        return campaign

    # See config.py's leadboost_integration_sender_* docstring for why
    # this fallback exists and why it is deployment-level, not per-org --
    # a known, explicitly-recorded open item, not a silent guess.
    settings = get_settings()
    if not settings.leadboost_integration_sender_email:
        logger.error(
            "LeadBoost integration campaign creation blocked for org=%s: "
            "LEADBOOST_INTEGRATION_SENDER_EMAIL is not configured on this deployment.",
            org_id,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "LeadBoost integration is not configured on this Mailer Agent "
                "deployment (missing sender identity). Contact the operator."
            ),
        )

    campaign = Campaign(
        name="LeadBoost Integration",
        organization_id=org_id,
        integration_source=LEADBOOST_INTEGRATION_SOURCE,
        sender_name=settings.leadboost_integration_sender_name,
        sender_org=settings.leadboost_integration_sender_org,
        sender_email=settings.leadboost_integration_sender_email,
        value_prop=(
            "LeadBoost-authorized outreach -- every message sent under this "
            "campaign is supplied verbatim per action by LeadBoost; this "
            "campaign's own value_prop is never used to draft or alter any "
            "message."
        ),
        follow_up_days=[],
        max_follow_ups=0,
    )
    db.add(campaign)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        campaign = (
            db.query(Campaign)
            .filter(
                Campaign.organization_id == org_id,
                Campaign.integration_source == LEADBOOST_INTEGRATION_SOURCE,
            )
            .first()
        )
        if campaign is None:
            # Genuinely unexpected -- the IntegrityError wasn't the
            # uniqueness race we expected, or the winner's row somehow
            # isn't visible. Don't silently swallow this.
            raise
        return campaign

    db.refresh(campaign)
    return campaign


# ---------------------------------------------------------------------------
# Contact get-or-create (C4)
# ---------------------------------------------------------------------------

def _get_or_create_integration_contact(
    db: Session, campaign_id: int, email: str, name: str | None
) -> Contact:
    """
    Same uq_contacts_campaign_email backstop ingest_leads already relies
    on (models.py, api/campaigns.py) -- reused as-is, not reimplemented.

    CONTACT SAFETY (C4, corrected in Batch 1.1): next_action_at must be
    NULL for every integration contact. The create path below sets it
    directly and that's the end of the story for a brand-new contact.
    The REUSE path cannot simply assume it is still NULL, and an earlier
    version of this function incorrectly did (its docstring claimed the
    invariant held "on both the create and the reuse path" without the
    reuse path actually checking anything). This codebase has real,
    already-shipped paths that can set next_action_at on ANY contact,
    including an integration one, with no awareness that the contact is
    integration-managed:
      - mail/reply_handler_v2.py's automatic reschedule_after_reply()
        (fires whenever this contact replies to a sent message -- not
        reachable yet in this batch since nothing sends real email
        until C5+, but will be live the moment sending exists);
      - POST /contacts/{id}/force-followup (api/contacts.py) -- an
        admin endpoint, callable today, though it requires the contact
        to already be ACTIVE (can_send_followup), so it needs a prior
        manual /resume call first; still a real, reachable path.

    If either has populated next_action_at on an integration contact by
    the time a new LeadBoost dispatch reuses it, that contact is
    genuinely in an active, human-attended follow-up/conversation state.
    Two candidate fixes were considered: silently clear next_action_at
    back to NULL and proceed, or reject the new dispatch. Silently
    clearing it was rejected as the wrong choice: it would destroy a
    real, live follow-up schedule that arose from the contact's own
    reply (or an operator's own manual override) as an invisible side
    effect of an unrelated LeadBoost API call, with no way for anyone to
    notice. Instead, this function REJECTS the dispatch (409), with zero
    mutation -- no Message, no ExternalDispatch, and next_action_at is
    left completely untouched -- so the conflict is surfaced rather than
    silently resolved in either direction. A dispatch that is merely a
    replay of an already-accepted idempotency_key never reaches this
    function at all (the endpoint returns the existing operation before
    any campaign/contact lookup), so this only ever gates genuinely NEW
    dispatch attempts. See tests/test_leadboost_integration.py's
    regression test.
    """
    contact = (
        db.query(Contact)
        .filter(Contact.campaign_id == campaign_id, Contact.email == email)
        .first()
    )
    if contact is not None:
        _reject_if_contact_has_active_followup(contact)
        return contact

    contact = Contact(
        campaign_id=campaign_id,
        name=name,
        email=email,
        status=ContactStatus.NEW.value,
        next_action_at=None,  # explicit, not just relying on the column default -- see docstring
    )
    db.add(contact)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        contact = (
            db.query(Contact)
            .filter(Contact.campaign_id == campaign_id, Contact.email == email)
            .first()
        )
        if contact is None:
            raise
        # Same invariant check applies uniformly to the race-loser path.
        # Extremely unlikely to trip (the winner's row is only
        # microseconds old), but "extremely unlikely" is not "provably
        # impossible", and this must not be a special case.
        _reject_if_contact_has_active_followup(contact)
        return contact

    db.refresh(contact)
    return contact


def _reject_if_contact_has_active_followup(contact: Contact) -> None:
    """See _get_or_create_integration_contact's docstring for the full
    reasoning. Raises 409 with zero mutation; never called on a
    freshly-created contact, only on a reused one."""
    if contact.next_action_at is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "This recipient already has an active, human-attended "
                "follow-up scheduled on the LeadBoost integration campaign "
                "(next_action_at is set). A new LeadBoost-authorized "
                "dispatch cannot be safely applied without risking a "
                "conflicting message to the same contact. No dispatch was "
                "created; next_action_at was not modified."
            ),
        )


# ---------------------------------------------------------------------------
# Response helper
# ---------------------------------------------------------------------------

def _accepted_response(dispatch: ExternalDispatch) -> LeadBoostOutreachActionAccepted:
    return LeadBoostOutreachActionAccepted(
        accepted=True,
        mailing_agent_reference=dispatch.public_reference,
    )


# ---------------------------------------------------------------------------
# POST /integrations/leadboost/outreach-actions
# ---------------------------------------------------------------------------

@router.post(
    "/outreach-actions",
    response_model=LeadBoostOutreachActionAccepted,
    status_code=202,
)
def create_leadboost_outreach_action(
    payload: LeadBoostOutreachActionIn,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    """
    Durably records an already-authorized LeadBoost outreach action for
    later, asynchronous dispatch. Does not send anything.

    TRANSACTION BOUNDARIES (corrected in Batch 1.1 -- this docstring
    previously said "Transaction A" as if this were one atomic SQL
    transaction; it is not, and this is intentional, not an oversight):

    1. org_id resolved from X-API-Key (dependency, above).
    2. payload validated by LeadBoostOutreachActionIn.
    3. idempotency lookup, scoped to org_id in the query predicate itself.
       same fingerprint -> return the existing operation, no mutation.
       different fingerprint -> 409, no mutation. (No commit either way
       -- this is a read.)
    4. get-or-create integration Campaign
       (_get_or_create_integration_campaign) -- its OWN short commit if
       a new row is inserted, independent of everything below.
    5. get-or-create Contact (_get_or_create_integration_contact,
       next_action_at forced NULL on create; rejected with 409 + zero
       mutation if an existing contact has an active follow-up -- see
       that function's docstring) -- its OWN short commit if a new row
       is inserted, independent of everything below.
    6. create Message (DRAFT, exact caller-supplied subject/body) and
       create ExternalDispatch(state=queued) together, and commit BOTH
       in one final, atomic commit -- this is the one commit that
       actually matters for duplicate-prevention (protected by
       uq_external_dispatches_org_idempotency_key).
    7. return 202 {"accepted": true, "mailing_agent_reference": ...}

    Why three separate commits are acceptable, not a bug: Campaign and
    Contact are idempotent, reusable, get-or-create resources -- if
    request processing fails or crashes after step 4 or 5 but before
    step 6 commits, the org is left with a perfectly valid, reusable
    Campaign/Contact row and nothing else. The NEXT request (a legitimate
    retry with the same idempotency_key, or even an unrelated dispatch to
    a different recipient) simply finds and reuses that row via the same
    get-or-create query -- no cleanup needed, no orphaned state that
    causes incorrect behavior. What must never happen -- a duplicate
    Message or a duplicate ExternalDispatch for the same
    (organization_id, idempotency_key) -- is fully prevented by step 6
    being one atomic commit guarded by the unique constraint, regardless
    of how steps 4-5 landed. In short: a failed final commit may leave a
    reusable Campaign/Contact row behind, but it cannot create a
    duplicate dispatch or a duplicate send.
    """
    fingerprint = _compute_request_fingerprint(
        external_action_id=payload.external_action_id,
        recipient_email=payload.recipient.email,
        recipient_name=payload.recipient.name,
        subject=payload.message.subject,
        body=payload.message.body,
    )

    existing = (
        db.query(ExternalDispatch)
        .filter(
            ExternalDispatch.organization_id == org_id,
            ExternalDispatch.idempotency_key == payload.idempotency_key,
        )
        .first()
    )
    if existing is not None:
        if existing.request_fingerprint == fingerprint:
            logger.info(
                "Idempotent replay: org=%s idempotency_key=%s -> existing dispatch %s (no mutation)",
                org_id, payload.idempotency_key, existing.public_reference,
            )
            return _accepted_response(existing)
        logger.warning(
            "Idempotency key reused with a different payload: org=%s idempotency_key=%s",
            org_id, payload.idempotency_key,
        )
        raise HTTPException(
            status_code=409,
            detail=(
                "idempotency_key has already been used with a different request "
                "payload. Use a new idempotency_key for a genuinely new action."
            ),
        )

    campaign = _get_or_create_integration_campaign(db, org_id)
    contact = _get_or_create_integration_contact(
        db, campaign.id, payload.recipient.email, payload.recipient.name
    )

    message = Message(
        contact_id=contact.id,
        direction=MessageDirection.OUTBOUND.value,
        message_type=MessageType.INITIAL.value,
        subject=payload.message.subject,
        body=payload.message.body,
        status=MessageStatus.DRAFT.value,
    )
    db.add(message)
    db.flush()  # assign message.id without committing yet

    dispatch = ExternalDispatch(
        organization_id=org_id,
        idempotency_key=payload.idempotency_key,
        external_action_id=payload.external_action_id,
        correlation_id=payload.correlation_id,
        campaign_id=campaign.id,
        contact_id=contact.id,
        message_id=message.id,
        request_fingerprint=fingerprint,
        public_reference=uuid.uuid4().hex,
        state=ExternalDispatchState.QUEUED.value,
    )
    db.add(dispatch)

    try:
        db.commit()
    except IntegrityError:
        # Lost a concurrent race on (organization_id, idempotency_key)
        # against an identical request -- the Message we just flushed is
        # rolled back along with everything else in this transaction, and
        # we re-read the winner's row instead. See ExternalDispatch's
        # docstring in models.py: this constraint is the actual
        # idempotency backstop, not the pre-check above.
        db.rollback()
        winner = (
            db.query(ExternalDispatch)
            .filter(
                ExternalDispatch.organization_id == org_id,
                ExternalDispatch.idempotency_key == payload.idempotency_key,
            )
            .first()
        )
        if winner is None:
            raise
        if winner.request_fingerprint != fingerprint:
            raise HTTPException(
                status_code=409,
                detail=(
                    "idempotency_key has already been used with a different "
                    "request payload. Use a new idempotency_key for a genuinely "
                    "new action."
                ),
            )
        logger.info(
            "Concurrent identical request converged on existing dispatch: "
            "org=%s idempotency_key=%s -> %s",
            org_id, payload.idempotency_key, winner.public_reference,
        )
        return _accepted_response(winner)

    db.refresh(dispatch)
    logger.info(
        "Accepted LeadBoost outreach action: org=%s external_action_id=%s "
        "correlation_id=%s -> dispatch=%s contact=%s message=%s",
        org_id, payload.external_action_id, payload.correlation_id,
        dispatch.public_reference, contact.id, message.id,
    )
    return _accepted_response(dispatch)
