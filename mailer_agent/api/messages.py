from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from mailer_agent.api.deps import require_api_key
from mailer_agent.db import get_db
from mailer_agent.followup.engine import is_suppressed
from mailer_agent.mail.sender import send_email
from mailer_agent.models import Message, MessageStatus, SuppressionEntry
from mailer_agent.schemas import MessageOut, SuppressRequest

router = APIRouter(tags=["messages"], dependencies=[Depends(require_api_key)])


@router.post("/messages/{message_id}/approve", response_model=MessageOut)
def approve_and_send_draft(message_id: int, db: Session = Depends(get_db)):
    """
    Sends a DRAFT message (either live_sending_enabled was False at
    generation time, or it was a reply awaiting approval because
    auto_reply_enabled is False / the intent required review). This is
    the human-in-the-loop control surface: nothing marked draft ever
    reaches a real inbox until this is called.
    """
    msg = db.get(Message, message_id)
    if not msg:
        raise HTTPException(404, "Message not found")
    if msg.status != MessageStatus.DRAFT.value:
        raise HTTPException(400, f"Message is not a draft (status={msg.status})")

    contact = msg.contact
    campaign = contact.campaign

    if is_suppressed(db, contact.email):
        raise HTTPException(400, "This contact is on the suppression list -- cannot send")

    prior = msg.in_reply_to_header
    subject = msg.subject or (f"Re: {contact.messages[0].subject}" if contact.messages and contact.messages[0].subject else f"{campaign.sender_org}")

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


@router.post("/suppress")
def suppress_email(payload: SuppressRequest, db: Session = Depends(get_db)):
    if not db.query(SuppressionEntry).filter_by(email=payload.email).first():
        db.add(SuppressionEntry(email=payload.email, reason=payload.reason))
        db.commit()
    return {"email": payload.email, "suppressed": True}


@router.get("/suppress/{email}")
def check_suppressed(email: str, db: Session = Depends(get_db)):
    return {"email": email, "suppressed": is_suppressed(db, email)}
