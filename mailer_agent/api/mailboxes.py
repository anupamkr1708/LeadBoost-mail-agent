"""
Mailbox API (M1): Mailer-owned sending identities with encrypted credentials.

    POST   /mailboxes
    GET    /mailboxes
    GET    /mailboxes/{public_reference}
    PATCH  /mailboxes/{public_reference}     (status, write-only password replacement)

Organization comes only from the authenticated key (get_authenticated_org_id,
fail-closed). A reference owned by another organization is indistinguishable
from an unknown one (404). Responses are the MailboxOut allowlist: no
passwords, no ciphertext. 422 bodies are sanitized for this router only so a
rejected password is never echoed back.
"""

from __future__ import annotations

import logging
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mailer_agent.api.deps import get_authenticated_org_id
from mailer_agent.db import get_db
from mailer_agent.mailbox_secrets import MailboxEncryptionUnavailable, encrypt_secret
from mailer_agent.models import Mailbox
from mailer_agent.schemas import MailboxCreate, MailboxOut, MailboxUpdate

logger = logging.getLogger("mailer_agent.api.mailboxes")


class _RedactedValidationRoute(APIRoute):
    """Return 422s with only type/loc/msg -- FastAPI's default also echoes `input`."""

    def get_route_handler(self) -> Callable:
        handler = super().get_route_handler()

        async def sanitized(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                detail = [
                    {"type": e.get("type"), "loc": list(e.get("loc", ())), "msg": e.get("msg")}
                    for e in exc.errors()
                ]
                return JSONResponse(status_code=422, content={"detail": detail})

        return sanitized


router = APIRouter(prefix="/mailboxes", tags=["mailboxes"], route_class=_RedactedValidationRoute)

_NOT_FOUND = "Mailbox not found"
_DUPLICATE = "A mailbox with this email address already exists for this organization"


def _encryption_unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail="Mailbox credential encryption is not configured")


def _get_owned(db: Session, org_id: str, public_reference: str) -> Mailbox:
    mailbox = (
        db.query(Mailbox)
        .filter(Mailbox.public_reference == public_reference, Mailbox.organization_id == org_id)
        .first()
    )
    if mailbox is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return mailbox


@router.post("", response_model=MailboxOut, status_code=201)
def create_mailbox(
    payload: MailboxCreate,
    org_id: str = Depends(get_authenticated_org_id),
    db: Session = Depends(get_db),
):
    try:
        smtp_enc = encrypt_secret(payload.smtp_password.get_secret_value())
        imap_enc = (
            encrypt_secret(payload.imap_password.get_secret_value())
            if payload.imap_password is not None
            else None
        )
    except MailboxEncryptionUnavailable:
        logger.error("Mailbox create refused: encryption key missing or invalid")
        raise _encryption_unavailable() from None

    exists = (
        db.query(Mailbox.id)
        .filter(Mailbox.organization_id == org_id, Mailbox.email_address == payload.email_address)
        .first()
    )
    if exists is not None:
        raise HTTPException(status_code=409, detail=_DUPLICATE)

    mailbox = Mailbox(
        organization_id=org_id,
        email_address=payload.email_address,
        smtp_host=payload.smtp_host,
        smtp_port=payload.smtp_port,
        smtp_use_tls=payload.smtp_use_tls,
        smtp_username=payload.smtp_username,
        smtp_password_enc=smtp_enc,
        imap_host=payload.imap_host,
        imap_port=payload.imap_port,
        imap_username=payload.imap_username,
        imap_password_enc=imap_enc,
    )
    db.add(mailbox)
    try:
        db.commit()
    except IntegrityError:
        # Lost a race against the UNIQUE(organization_id, email_address) backstop.
        db.rollback()
        raise HTTPException(status_code=409, detail=_DUPLICATE) from None
    db.refresh(mailbox)
    logger.info("Mailbox created: %s", mailbox.public_reference)
    return mailbox


@router.get("", response_model=list[MailboxOut])
def list_mailboxes(
    org_id: str = Depends(get_authenticated_org_id),
    db: Session = Depends(get_db),
):
    return (
        db.query(Mailbox)
        .filter(Mailbox.organization_id == org_id)
        .order_by(Mailbox.created_at, Mailbox.id)
        .all()
    )


@router.get("/{public_reference}", response_model=MailboxOut)
def get_mailbox(
    public_reference: str,
    org_id: str = Depends(get_authenticated_org_id),
    db: Session = Depends(get_db),
):
    return _get_owned(db, org_id, public_reference)


@router.patch("/{public_reference}", response_model=MailboxOut)
def update_mailbox(
    public_reference: str,
    payload: MailboxUpdate,
    org_id: str = Depends(get_authenticated_org_id),
    db: Session = Depends(get_db),
):
    mailbox = _get_owned(db, org_id, public_reference)

    if payload.imap_password is not None and mailbox.imap_host is None:
        raise HTTPException(status_code=409, detail="Mailbox has no IMAP configuration")

    # Encrypt everything before mutating anything so a 503 leaves the row untouched.
    try:
        smtp_enc = (
            encrypt_secret(payload.smtp_password.get_secret_value())
            if payload.smtp_password is not None
            else None
        )
        imap_enc = (
            encrypt_secret(payload.imap_password.get_secret_value())
            if payload.imap_password is not None
            else None
        )
    except MailboxEncryptionUnavailable:
        logger.error("Mailbox update refused: encryption key missing or invalid")
        raise _encryption_unavailable() from None

    if payload.status is not None:
        mailbox.status = payload.status.value
    if smtp_enc is not None:
        mailbox.smtp_password_enc = smtp_enc
    if imap_enc is not None:
        mailbox.imap_password_enc = imap_enc

    db.commit()
    db.refresh(mailbox)
    logger.info("Mailbox updated: %s", mailbox.public_reference)
    return mailbox
