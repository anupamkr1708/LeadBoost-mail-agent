"""
Campaign management endpoints.

Multi-tenancy: every query is scoped to the calling organization's org_id,
resolved from the X-API-Key header via get_current_org_id().  A campaign
created by org "acme" is invisible to org "leadboost" and vice-versa.
"""

from __future__ import annotations

import logging
from datetime import timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mailer_agent.api.deps import get_current_org_id, require_api_key
from mailer_agent.db import get_db
from mailer_agent.followup.engine import is_suppressed
from mailer_agent.lead_ingestion import LeadIngestionError, normalize_lead_payload
from mailer_agent.models import Campaign, Contact, ContactStatus
from mailer_agent.schemas import (
    CampaignCreate,
    CampaignOut,
    ContactBulkCreate,
    ContactOut,
    LeadIngestRequest,
)
from mailer_agent.utils.datetime_utils import utcnow

logger = logging.getLogger("mailer_agent.api.campaigns")

router = APIRouter(
    prefix="/campaigns",
    tags=["campaigns"],
    dependencies=[Depends(require_api_key)],
)

# ---------------------------------------------------------------------------
# Campaign CRUD
# ---------------------------------------------------------------------------


@router.post("", response_model=CampaignOut, status_code=201)
def create_campaign(
    payload: CampaignCreate,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    campaign = Campaign(
        name=payload.name,
        organization_id=org_id,
        sender_name=payload.sender_name,
        sender_org=payload.sender_org,
        sender_email=payload.sender_email,
        reply_to_email=payload.reply_to_email,
        sender_title=payload.sender_title,
        sender_signature_extra=payload.sender_signature_extra,
        value_prop=payload.value_prop,
        proof_points=payload.proof_points,
        tone=payload.tone,
        follow_up_days=payload.follow_up_days,
        max_follow_ups=len(payload.follow_up_days),
        timezone=payload.timezone,
    )
    db.add(campaign)
    db.commit()
    db.refresh(campaign)
    return campaign


@router.get("", response_model=list[CampaignOut])
def list_campaigns(
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    return (
        db.query(Campaign)
        .filter(Campaign.organization_id == org_id)
        .order_by(Campaign.created_at.desc())
        .all()
    )


@router.get("/{campaign_id}", response_model=CampaignOut)
def get_campaign(
    campaign_id: int,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    campaign = _get_campaign_or_404(db, campaign_id, org_id)
    return campaign


# ---------------------------------------------------------------------------
# Contact / Lead ingestion
# ---------------------------------------------------------------------------


@router.post("/{campaign_id}/contacts", response_model=list[ContactOut], status_code=201)
def add_contacts(
    campaign_id: int,
    payload: ContactBulkCreate,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    campaign = _get_campaign_or_404(db, campaign_id, org_id)

    created: list[Contact] = []
    skipped: list[dict] = []

    for c in payload.contacts:
        if is_suppressed(db, c.email):
            skipped.append({"reason": "suppressed", "email": c.email})
            continue

        # Check for existing contact (application-level; DB constraint is the backstop)
        existing = (
            db.query(Contact)
            .filter(Contact.campaign_id == campaign_id, Contact.email == c.email)
            .first()
        )
        if existing:
            skipped.append({"reason": "already_exists", "email": c.email, "contact_id": existing.id})
            continue

        contact = Contact(
            campaign_id=campaign_id,
            name=c.name,
            email=c.email,
            title=c.title,
            company=c.company,
            context_notes=c.context_notes,
            status=ContactStatus.NEW.value,
        )
        db.add(contact)
        created.append(contact)

    # Flush to catch any DB-level unique constraint violations (race conditions).
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        logger.warning("IntegrityError adding contacts to campaign %s: %s", campaign_id, exc)
        raise HTTPException(
            status_code=409,
            detail=(
                "One or more contacts already exist in this campaign "
                "(duplicate detected at database level). "
                "Re-POST the list without the duplicates, or use the leads/ingest "
                "endpoint which handles duplicates per-item."
            ),
        ) from exc

    db.commit()
    for c in created:
        db.refresh(c)
    return created


@router.get("/{campaign_id}/contacts", response_model=list[ContactOut])
def list_contacts(
    campaign_id: int,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    _get_campaign_or_404(db, campaign_id, org_id)
    return db.query(Contact).filter(Contact.campaign_id == campaign_id).all()


@router.post("/{campaign_id}/leads/ingest")
def ingest_leads(
    campaign_id: int,
    payload: LeadIngestRequest,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    """
    Flexible ingestion for leads in any shape -- accepts LeadBoost's own Lead
    record fields directly (company_name, contact_name, contact_title, email,
    about_text, industry, employees, revenue_band, qualification_label, score,
    ...) or generic aliases; unrecognized fields are folded into context_notes.

    Per-item failures (missing email, already suppressed, already a contact in
    this campaign) are reported individually rather than failing the whole batch.
    """
    _get_campaign_or_404(db, campaign_id, org_id)

    created: list[dict] = []
    skipped: list[dict] = []

    for raw_lead in payload.leads:
        try:
            normalized = normalize_lead_payload(raw_lead)
        except LeadIngestionError as e:
            skipped.append({"reason": str(e), "raw": raw_lead})
            continue

        if is_suppressed(db, normalized["email"]):
            skipped.append({"reason": "suppressed", "email": normalized["email"]})
            continue

        existing = (
            db.query(Contact)
            .filter(Contact.campaign_id == campaign_id, Contact.email == normalized["email"])
            .first()
        )
        if existing:
            skipped.append(
                {
                    "reason": "already_exists",
                    "email": normalized["email"],
                    "contact_id": existing.id,
                }
            )
            continue

        contact = Contact(
            campaign_id=campaign_id,
            name=normalized["name"],
            email=normalized["email"],
            title=normalized["title"],
            company=normalized["company"],
            context_notes=normalized["context_notes"],
            status=ContactStatus.NEW.value,
        )
        db.add(contact)

        # Flush after each contact so we catch constraint violations per-item.
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            skipped.append(
                {
                    "reason": "duplicate_at_db_level",
                    "email": normalized["email"],
                }
            )
            continue

        created.append({"contact_id": contact.id, "email": contact.email})

    db.commit()
    return {"campaign_id": campaign_id, "created": created, "skipped": skipped}


# ---------------------------------------------------------------------------
# Campaign lifecycle
# ---------------------------------------------------------------------------


@router.post("/{campaign_id}/start")
def start_campaign(
    campaign_id: int,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    """
    Start a campaign by marking all NEW contacts as ready for immediate dispatch.

    Creates durable work entries that the scheduler picks up naturally — no
    process-local BackgroundTasks so this is safe with multiple API workers.

    Idempotent: already-ACTIVE contacts are not touched.
    """
    campaign = _get_campaign_or_404(db, campaign_id, org_id)

    new_contacts = (
        db.query(Contact)
        .filter(
            Contact.campaign_id == campaign_id,
            Contact.status == ContactStatus.NEW.value,
        )
        .all()
    )

    if not new_contacts:
        return {"campaign_id": campaign_id, "queued": 0, "message": "No new contacts to start"}

    now = utcnow()
    for contact in new_contacts:
        contact.next_action_at = now

    db.commit()

    from mailer_agent.config import get_settings as _gs

    s = _gs()
    logger.info("Campaign %s started: %d contacts queued for sending", campaign_id, len(new_contacts))
    return {
        "campaign_id": campaign_id,
        "queued": len(new_contacts),
        "message": (
            f"Queued {len(new_contacts)} contact(s) for sending. "
            f"The scheduler will dispatch them with ~{s.send_delay_seconds}s staggering. "
            f"Poll GET /campaigns/{campaign_id}/contacts to track progress."
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_campaign_or_404(db: Session, campaign_id: int, org_id: str) -> Campaign:
    """
    Fetch campaign by id and verify it belongs to the calling org.
    Returns 404 (not 403) for unknown campaigns regardless of ownership,
    so org boundaries are not discoverable via error codes.
    """
    campaign = (
        db.query(Campaign)
        .filter(Campaign.id == campaign_id, Campaign.organization_id == org_id)
        .first()
    )
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign
