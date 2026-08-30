from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from mailer_agent.api.deps import require_api_key
from mailer_agent.db import get_db
from mailer_agent.followup.engine import is_suppressed, send_initial_outreach
from mailer_agent.lead_ingestion import LeadIngestionError, normalize_lead_payload
from mailer_agent.models import Campaign, Contact, ContactStatus
from mailer_agent.schemas import (
    CampaignCreate,
    CampaignOut,
    ContactBulkCreate,
    ContactOut,
    LeadIngestRequest,
)

router = APIRouter(prefix="/campaigns", tags=["campaigns"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=CampaignOut)
def create_campaign(payload: CampaignCreate, db: Session = Depends(get_db)):
    campaign = Campaign(
        name=payload.name,
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
    )
    db.add(campaign)
    db.commit()
    db.refresh(campaign)
    return campaign


@router.get("", response_model=list[CampaignOut])
def list_campaigns(db: Session = Depends(get_db)):
    return db.query(Campaign).order_by(Campaign.created_at.desc()).all()


@router.get("/{campaign_id}", response_model=CampaignOut)
def get_campaign(campaign_id: int, db: Session = Depends(get_db)):
    campaign = db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(404, "Campaign not found")
    return campaign


@router.post("/{campaign_id}/contacts", response_model=list[ContactOut])
def add_contacts(campaign_id: int, payload: ContactBulkCreate, db: Session = Depends(get_db)):
    campaign = db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(404, "Campaign not found")

    created: list[Contact] = []
    for c in payload.contacts:
        if is_suppressed(db, c.email):
            continue  # never add a suppressed address back into a sequence
        existing = (
            db.query(Contact)
            .filter(Contact.campaign_id == campaign_id, Contact.email == c.email)
            .first()
        )
        if existing:
            continue  # idempotent -- re-posting the same list twice is safe
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

    db.commit()
    for c in created:
        db.refresh(c)
    return created


@router.get("/{campaign_id}/contacts", response_model=list[ContactOut])
def list_contacts(campaign_id: int, db: Session = Depends(get_db)):
    return db.query(Contact).filter(Contact.campaign_id == campaign_id).all()


@router.post("/{campaign_id}/leads/ingest")
def ingest_leads(campaign_id: int, payload: LeadIngestRequest, db: Session = Depends(get_db)):
    """
    Flexible ingestion for leads in any shape -- built specifically to
    accept LeadBoost's own Lead record fields (company_name, contact_name,
    contact_title, email, about_text, industry, employees, revenue_band,
    qualification_label, score, ...) directly, alongside generic aliases,
    so the caller doesn't need to pre-transform anything. See
    mailer_agent/lead_ingestion.py for the exact field mapping.

    Per-item failures (missing email, already suppressed, already a
    contact in this campaign) are reported individually rather than
    failing the whole batch.
    """
    campaign = db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(404, "Campaign not found")

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
            skipped.append({"reason": "already_exists", "email": normalized["email"], "contact_id": existing.id})
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
        db.flush()
        created.append({"contact_id": contact.id, "email": contact.email})

    db.commit()
    return {"campaign_id": campaign_id, "created": created, "skipped": skipped}


@router.post("/{campaign_id}/start")
def start_campaign(campaign_id: int, db: Session = Depends(get_db)):
    """
    Immediately sends initial outreach to every NEW contact in this
    campaign (rather than waiting for the next scheduler tick). The
    background scheduler will still pick up any contacts added later
    and all follow-ups/replies from here on.
    """
    campaign = db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(404, "Campaign not found")

    new_contacts = (
        db.query(Contact)
        .filter(Contact.campaign_id == campaign_id, Contact.status == ContactStatus.NEW.value)
        .all()
    )
    results = [send_initial_outreach(db, c) for c in new_contacts]
    db.commit()
    return {"campaign_id": campaign_id, "dispatched": len(results), "results": results}
