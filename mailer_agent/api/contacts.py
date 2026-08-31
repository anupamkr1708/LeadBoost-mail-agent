"""
Contact management endpoints.

Multi-tenancy: contact access is validated through campaign ownership —
a contact belongs to the calling org if and only if its campaign does.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from mailer_agent.api.deps import get_current_org_id, require_api_key
from mailer_agent.db import get_db
from mailer_agent.followup.engine import send_followup_if_due
from mailer_agent.models import Campaign, Contact
from mailer_agent.schemas import ContactOut, ThreadOut
from mailer_agent.state_machine import StateTransitionEvent, can_send_followup, transition_contact_state
from mailer_agent.utils.datetime_utils import utcnow

router = APIRouter(
    prefix="/contacts",
    tags=["contacts"],
    dependencies=[Depends(require_api_key)],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_contact_or_404(db: Session, contact_id: int, org_id: str) -> Contact:
    """
    Fetch contact and verify it belongs to the calling org (via campaign ownership).
    Returns 404 regardless of whether the id exists or belongs to another org,
    so org boundaries are not discoverable via error codes.
    """
    contact = (
        db.query(Contact)
        .join(Campaign, Contact.campaign_id == Campaign.id)
        .filter(Contact.id == contact_id, Campaign.organization_id == org_id)
        .first()
    )
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    return contact


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------


@router.get("/{contact_id}", response_model=ContactOut)
def get_contact(
    contact_id: int,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    return _get_contact_or_404(db, contact_id, org_id)


@router.get("/{contact_id}/thread", response_model=ThreadOut)
def get_thread(
    contact_id: int,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    contact = _get_contact_or_404(db, contact_id, org_id)
    return ThreadOut(contact=contact, messages=contact.messages)


# ---------------------------------------------------------------------------
# Lifecycle actions
# ---------------------------------------------------------------------------


@router.post("/{contact_id}/pause", response_model=ContactOut)
def pause_contact(
    contact_id: int,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    """Manually pause the sequence for a contact (uses state machine)."""
    contact = _get_contact_or_404(db, contact_id, org_id)

    success = transition_contact_state(
        contact,
        StateTransitionEvent.MANUAL_PAUSE,
        reason="User paused via API",
    )
    if not success:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot pause contact in state {contact.status}. "
                "Only NEW or ACTIVE contacts can be paused."
            ),
        )

    contact.next_action_at = None
    db.commit()
    db.refresh(contact)
    return contact


@router.post("/{contact_id}/resume", response_model=ContactOut)
def resume_contact(
    contact_id: int,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    """Manually resume the sequence for a contact (uses state machine)."""
    contact = _get_contact_or_404(db, contact_id, org_id)

    success = transition_contact_state(
        contact,
        StateTransitionEvent.MANUAL_RESUME,
        reason="User resumed via API",
    )
    if not success:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot resume contact in state {contact.status}. "
                "Only PAUSED, NEEDS_REVIEW, or SEQUENCE_COMPLETE contacts can be resumed."
            ),
        )

    db.commit()
    db.refresh(contact)
    return contact


@router.post("/{contact_id}/force-followup")
def force_followup(
    contact_id: int,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    """
    Bypass the schedule and send the next follow-up right now (manual override).
    Still respects state machine rules — won't send if contact is not in a
    sendable state.
    """
    contact = _get_contact_or_404(db, contact_id, org_id)

    if not can_send_followup(contact):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot force follow-up for contact in state {contact.status}. "
                "Contact must be ACTIVE to send follow-ups."
            ),
        )

    contact.next_action_at = utcnow()
    db.commit()

    result = send_followup_if_due(db, contact)
    db.commit()
    return result or {"action": "not_due_or_not_active"}


@router.post("/{contact_id}/mark-won", response_model=ContactOut)
def mark_won(
    contact_id: int,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    """Mark a deal as closed-won (uses state machine)."""
    contact = _get_contact_or_404(db, contact_id, org_id)

    success = transition_contact_state(
        contact,
        StateTransitionEvent.MANUAL_WON,
        reason="User marked won via API",
    )
    if not success:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot mark contact as won from state {contact.status}. "
                "Contact must be in an active sales state."
            ),
        )

    contact.next_action_at = None
    db.commit()
    db.refresh(contact)
    return contact
