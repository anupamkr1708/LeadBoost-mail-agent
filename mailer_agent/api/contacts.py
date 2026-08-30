from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from mailer_agent.api.deps import require_api_key
from mailer_agent.db import get_db
from mailer_agent.followup.engine import send_followup_if_due
from mailer_agent.models import Contact, ContactStatus
from mailer_agent.schemas import ContactOut, ThreadOut

router = APIRouter(prefix="/contacts", tags=["contacts"], dependencies=[Depends(require_api_key)])


@router.get("/{contact_id}", response_model=ContactOut)
def get_contact(contact_id: int, db: Session = Depends(get_db)):
    contact = db.get(Contact, contact_id)
    if not contact:
        raise HTTPException(404, "Contact not found")
    return contact


@router.get("/{contact_id}/thread", response_model=ThreadOut)
def get_thread(contact_id: int, db: Session = Depends(get_db)):
    contact = db.get(Contact, contact_id)
    if not contact:
        raise HTTPException(404, "Contact not found")
    return ThreadOut(contact=contact, messages=contact.messages)


@router.post("/{contact_id}/pause", response_model=ContactOut)
def pause_contact(contact_id: int, db: Session = Depends(get_db)):
    contact = db.get(Contact, contact_id)
    if not contact:
        raise HTTPException(404, "Contact not found")
    contact.status = ContactStatus.PAUSED.value
    contact.next_action_at = None
    db.commit()
    db.refresh(contact)
    return contact


@router.post("/{contact_id}/resume", response_model=ContactOut)
def resume_contact(contact_id: int, db: Session = Depends(get_db)):
    contact = db.get(Contact, contact_id)
    if not contact:
        raise HTTPException(404, "Contact not found")
    if contact.status == ContactStatus.PAUSED.value:
        contact.status = ContactStatus.ACTIVE.value
    db.commit()
    db.refresh(contact)
    return contact


@router.post("/{contact_id}/force-followup")
def force_followup(contact_id: int, db: Session = Depends(get_db)):
    """Bypasses the schedule and sends the next follow-up right now (manual override)."""
    contact = db.get(Contact, contact_id)
    if not contact:
        raise HTTPException(404, "Contact not found")
    from datetime import datetime

    contact.next_action_at = datetime.utcnow()
    db.commit()
    result = send_followup_if_due(db, contact)
    db.commit()
    return result or {"action": "not_due_or_not_active"}


@router.post("/{contact_id}/mark-won", response_model=ContactOut)
def mark_won(contact_id: int, db: Session = Depends(get_db)):
    contact = db.get(Contact, contact_id)
    if not contact:
        raise HTTPException(404, "Contact not found")
    contact.status = ContactStatus.CLOSED_WON.value
    contact.next_action_at = None
    db.commit()
    db.refresh(contact)
    return contact
