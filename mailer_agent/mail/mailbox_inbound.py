"""
M3: mailbox-bound inbound polling.

Each eligible Mailer Mailbox is polled with ITS OWN IMAP configuration and ITS OWN
encrypted credential, and everything it receives is bound to that Mailbox's
organization before any message content is interpreted.

    for each eligible mailbox:
        fresh DB read  ->  (mailbox_id, organization_id, imap config, ciphertext)
        decrypt password, connect, LOGIN         <- the only place a credential exists
        UID SEARCH UNSEEN -> UID FETCH (BODY.PEEK[]) -> parse
        per message:  own session -> process_mailbox_inbound -> COMMIT -> UID STORE Seen flag

Separate from the legacy deployment-global poll (mail/imap_reader.fetch_unseen_replies
+ followup/scheduler.poll_replies_job), which is kept for non-integrated campaigns.
Nothing here ever reads settings.imap_host/imap_username/imap_password: the global
IMAP identity is NOT a fallback for mailbox-bound work. Do not configure the same
physical account both as the global IMAP identity and as a Mailbox's IMAP identity --
the two paths dedupe independently.

Eligibility (re-evaluated from the database immediately before each mailbox is
polled, so a mailbox disabled or re-credentialed mid-cycle is honoured, and no
credential is ever cached across cycles): status ACTIVE, imap_host / imap_port /
imap_username / imap_password_enc all present. Anything else is skipped without a
connection attempt.

Isolation: every mailbox runs in its own try/except and every message in its own
database session/transaction, so one bad mailbox, credential or message never stops
another. No database transaction is held across any IMAP network call. A mailbox's
failure is reported through its MailboxPollOutcome and a WARNING that carries the
mailbox reference and an exception TYPE only (never a password, address, subject or
body).

Concurrency: one logical poll per mailbox at a time comes from the existing topology
(one scheduler process, max_instances=1). If two pollers ever overlap, correctness
still holds because inbound persistence is idempotent on (mailbox_id, Message-ID).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy.orm import Session

from mailer_agent.config import get_settings
from mailer_agent.mail.imap_reader import InboundEmail, drain_unseen, open_authenticated_connection
from mailer_agent.mail.reply_handler_v2 import process_mailbox_inbound
from mailer_agent.mailbox_secrets import (
    MailboxDecryptionError,
    MailboxEncryptionUnavailable,
    decrypt_secret,
)
from mailer_agent.models import Mailbox, MailboxStatus

logger = logging.getLogger("mailer_agent.mail.mailbox_inbound")

SessionFactory = Callable[[], Session]

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped_ineligible"
STATUS_CREDENTIAL_UNAVAILABLE = "credential_unavailable"
STATUS_IMAP_ERROR = "imap_error"
STATUS_UNEXPECTED = "unexpected_error"


@dataclass(frozen=True)
class MailboxImapTarget:
    """What one poll needs. The password exists only as ciphertext here."""

    mailbox_id: int
    organization_id: str
    public_reference: str
    imap_host: str
    imap_port: int
    imap_username: str
    imap_password_enc: str = field(repr=False)


@dataclass
class MailboxPollOutcome:
    mailbox_id: int
    public_reference: str | None = None
    status: str = STATUS_OK
    processed: int = 0   # messages durably handled and marked Seen
    left_unseen: int = 0  # messages that will be offered again next poll
    error_type: str | None = None


def _default_session_factory() -> Session:
    from mailer_agent.db import SessionLocal

    return SessionLocal()


def _eligible(mailbox: Mailbox) -> bool:
    return bool(
        mailbox.status == MailboxStatus.ACTIVE.value
        and mailbox.imap_host and mailbox.imap_host.strip()
        and mailbox.imap_port
        and mailbox.imap_username and mailbox.imap_username.strip()
        and mailbox.imap_password_enc
    )


def list_pollable_mailbox_ids(session_factory: SessionFactory | None = None) -> list[int]:
    db = (session_factory or _default_session_factory)()
    try:
        return [m.id for m in db.query(Mailbox).order_by(Mailbox.id).all() if _eligible(m)]
    finally:
        db.close()


def load_pollable_target(
    mailbox_id: int, session_factory: SessionFactory | None = None
) -> MailboxImapTarget | None:
    """Fresh read of one mailbox; None if it is no longer eligible."""
    db = (session_factory or _default_session_factory)()
    try:
        mailbox = db.get(Mailbox, mailbox_id)
        if mailbox is None or not _eligible(mailbox):
            return None
        return MailboxImapTarget(
            mailbox_id=mailbox.id,
            organization_id=mailbox.organization_id,
            public_reference=mailbox.public_reference,
            imap_host=mailbox.imap_host.strip(),
            imap_port=mailbox.imap_port,
            imap_username=mailbox.imap_username.strip(),
            imap_password_enc=mailbox.imap_password_enc,
        )
    finally:
        db.close()  # no transaction or session is held while we talk IMAP


def _persist_inbound(
    session_factory: SessionFactory, target: MailboxImapTarget, email_in: InboundEmail
) -> dict:
    """One message, one session, one transaction. Returns only after COMMIT."""
    db = session_factory()
    try:
        result = process_mailbox_inbound(
            db,
            email_in,
            mailbox_id=target.mailbox_id,
            organization_id=target.organization_id,
        )
        db.commit()
        return result
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def poll_mailbox(
    target: MailboxImapTarget, session_factory: SessionFactory | None = None
) -> MailboxPollOutcome:
    """Poll one mailbox. Never raises: every failure becomes the outcome."""
    factory = session_factory or _default_session_factory
    outcome = MailboxPollOutcome(mailbox_id=target.mailbox_id, public_reference=target.public_reference)
    conn = None
    try:
        # --- the credential boundary: decrypt immediately before LOGIN ---------
        try:
            password = decrypt_secret(target.imap_password_enc)
        except (MailboxEncryptionUnavailable, MailboxDecryptionError) as e:
            outcome.status = STATUS_CREDENTIAL_UNAVAILABLE
            outcome.error_type = type(e).__name__
            logger.warning(
                "Mailbox %s not polled: IMAP credential unavailable (%s)",
                target.public_reference, outcome.error_type,
            )
            return outcome

        try:
            conn = open_authenticated_connection(
                target.imap_host, target.imap_port, target.imap_username, password,
                get_settings().imap_timeout_seconds,
            )
        finally:
            password = None  # noqa: F841 - drop the only plaintext reference
        # ----------------------------------------------------------------------

        stats = drain_unseen(conn, lambda email_in: _persist_inbound(factory, target, email_in))
        outcome.processed = stats.seen_marked
        outcome.left_unseen = stats.left_unseen
    except Exception as e:  # noqa: BLE001 - isolate this mailbox; type name only
        outcome.status = STATUS_IMAP_ERROR
        outcome.error_type = type(e).__name__
        logger.warning(
            "Mailbox %s poll failed (%s); other mailboxes are unaffected",
            target.public_reference, outcome.error_type,
        )
    finally:
        if conn is not None:
            try:
                conn.logout()
            except Exception:  # noqa: BLE001
                pass
    return outcome


def poll_all_mailboxes(session_factory: SessionFactory | None = None) -> list[MailboxPollOutcome]:
    """One cycle over every currently eligible mailbox. Never raises."""
    outcomes: list[MailboxPollOutcome] = []
    try:
        ids = list_pollable_mailbox_ids(session_factory)
    except Exception as e:  # noqa: BLE001
        logger.error("Mailbox inbound cycle could not list mailboxes (%s)", type(e).__name__)
        return outcomes
    for mailbox_id in ids:
        try:
            target = load_pollable_target(mailbox_id, session_factory)  # fresh: disabled/rotated mid-cycle
            if target is None:
                outcomes.append(MailboxPollOutcome(mailbox_id=mailbox_id, status=STATUS_SKIPPED))
                continue
            outcomes.append(poll_mailbox(target, session_factory))
        except Exception as e:  # noqa: BLE001
            logger.error("Mailbox %s poll crashed unexpectedly (%s)", mailbox_id, type(e).__name__)
            outcomes.append(
                MailboxPollOutcome(mailbox_id=mailbox_id, status=STATUS_UNEXPECTED, error_type=type(e).__name__)
            )
    return outcomes
