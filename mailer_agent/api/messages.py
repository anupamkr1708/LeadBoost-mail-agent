"""
Message and suppression-list endpoints.

Multi-tenancy: message access is validated through contact → campaign ownership.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from mailer_agent.api.deps import get_current_org_id, require_api_key
from mailer_agent.db import get_db
from mailer_agent.followup.engine import is_suppressed
from mailer_agent.mail.sender import send_email
from mailer_agent.models import Campaign, Contact, Message, MessageStatus, SuppressionEntry
from mailer_agent.schemas import MessageOut, SuppressRequest

router = APIRouter(tags=["messages"], dependencies=[Depends(require_api_key)])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_message_or_404(db: Session, message_id: int, org_id: str) -> Message:
    """
    Fetch message and verify it belongs to the calling org.
    Returns 404 for both missing and cross-org messages.
    """
    msg = (
        db.query(Message)
        .join(Contact, Message.contact_id == Contact.id)
        .join(Campaign, Contact.campaign_id == Campaign.id)
        .filter(Message.id == message_id, Campaign.organization_id == org_id)
        .first()
    )
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")
    return msg


# ---------------------------------------------------------------------------
# Message actions
# ---------------------------------------------------------------------------


@router.post("/messages/{message_id}/approve", response_model=MessageOut)
def approve_and_send_draft(
    message_id: int,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    """
    Send a DRAFT message.

    This is the human-in-the-loop control surface: nothing marked draft ever
    reaches a real inbox until this endpoint is called.  Works for:
    - Messages generated when live_sending_enabled=False
    - Replies held for approval because auto_reply_enabled=False or the
      intent required review (objections, pricing questions, etc.)
    """
    msg = _get_message_or_404(db, message_id, org_id)

    if msg.status != MessageStatus.DRAFT.value:
        raise HTTPException(
            status_code=400,
            detail=f"Message is not a draft (status={msg.status})",
        )

    contact = msg.contact
    campaign = contact.campaign

    if is_suppressed(db, contact.email):
        raise HTTPException(
            status_code=400,
            detail="This contact is on the suppression list — cannot send",
        )

    prior = msg.in_reply_to_header
    subject = msg.subject or (
        f"Re: {contact.messages[0].subject}"
        if contact.messages and contact.messages[0].subject
        else campaign.sender_org
    )

    result = send_email(
        to_email=contact.email,
        from_email=campaign.sender_email,
        from_name=campaign.sender_name,
        subject=subject,
        body_text=msg.body,
        reply_to=campaign.reply_to_email,
        in_reply_to_header=prior,
    )
    msg.status = MessageStatus.SENT.value if result.success else MessageStatus.FAILED.value
    msg.message_id_header = result.message_id
    msg.subject = subject
    msg.error_message = result.error
    db.commit()
    db.refresh(msg)
    return msg


# ---------------------------------------------------------------------------
# Suppression list (org-scoped: each org can only see/add their own entries)
# ---------------------------------------------------------------------------


@router.post("/suppress", status_code=201)
def suppress_email(
    payload: SuppressRequest,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    """Add an email address to the suppression list for this organization."""
    if not db.query(SuppressionEntry).filter_by(email=payload.email, organization_id=org_id).first():
        db.add(SuppressionEntry(email=payload.email, reason=payload.reason, organization_id=org_id))
        db.commit()
    return {"email": payload.email, "suppressed": True}


@router.get("/suppress/{email}")
def check_suppressed(
    email: str,
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    """Check if an email address is suppressed for this organization."""
    suppressed = (
        db.query(SuppressionEntry)
        .filter_by(email=email, organization_id=org_id)
        .first()
    ) is not None
    return {"email": email, "suppressed": suppressed}
