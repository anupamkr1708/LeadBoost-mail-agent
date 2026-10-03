"""
Shared fixtures/builders for the async-dispatch (C6-C8) tests.

Kept as a plain module (not conftest) so each test file states explicitly
what it seeds. Everything here is deterministic and offline: no real SMTP,
no LLM, no network.
"""

from __future__ import annotations

import socketserver
import threading
import uuid
from datetime import datetime, timedelta, timezone

from mailer_agent.mail.exact_message import create_authorized_message
from mailer_agent.mailbox_secrets import encrypt_secret
from mailer_agent.models import (
    Campaign,
    Contact,
    ExternalDispatch,
    ExternalDispatchState,
    Mailbox,
    MailboxStatus,
)


def naive_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


MAILBOX_SMTP_PASSWORD = "mbx-smtp-secret-9f2c"


def seed_mailbox(
    db,
    *,
    org: str = "org-a",
    email: str = "outreach@sender.example.org",
    status: str = MailboxStatus.ACTIVE.value,
    smtp_host: str = "smtp.mailbox.example",
    smtp_port: int = 2525,
    smtp_use_tls: bool = False,
    password: str = MAILBOX_SMTP_PASSWORD,
) -> Mailbox:
    """Get-or-create the org's mailbox (unique per org+address). Needs
    MAILBOX_ENCRYPTION_KEY, which tests/conftest.py provides for every test."""
    mailbox = (
        db.query(Mailbox)
        .filter(Mailbox.organization_id == org, Mailbox.email_address == email)
        .first()
    )
    if mailbox is None:
        mailbox = Mailbox(
            organization_id=org,
            email_address=email,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_use_tls=smtp_use_tls,
            smtp_username=email,
            smtp_password_enc=encrypt_secret(password),
            status=status,
        )
        db.add(mailbox)
        db.flush()
    return mailbox


def seed_org_mailboxes(
    db, orgs=("org-a", "org-b"), email: str = "outreach@mailer.example.com"
) -> None:
    """Give each org exactly one ACTIVE mailbox (what intake now requires)."""
    for org in orgs:
        seed_mailbox(db, org=org, email=email)
    db.commit()


def seed_dispatch(
    db,
    *,
    org: str = "org-a",
    idem: str | None = None,
    email: str = "lead@example.com",
    subject: str | None = "Quick question",
    body: str = "Hello there,\n\nWould a call next week suit you?\n",
    state: str = ExternalDispatchState.QUEUED.value,
    claimed_by: str | None = None,
    claimed_at: datetime | None = None,
    sender_email: str = "outreach@sender.example.org",
    proof_points: str | None = None,
    commit: bool = True,
    mailbox: bool = True,
    mailbox_status: str = MailboxStatus.ACTIVE.value,
    smtp_host: str = "smtp.mailbox.example",
    smtp_port: int = 2525,
) -> ExternalDispatch:
    """One integration Campaign per org (get-or-create), one Contact, one
    DRAFT Message and one ExternalDispatch -- the same shape the HTTP
    endpoint commits in its final transaction. next_action_at stays NULL."""
    campaign = (
        db.query(Campaign)
        .filter(Campaign.organization_id == org, Campaign.integration_source == "leadboost")
        .first()
    )
    if campaign is None:
        campaign = Campaign(
            name=f"leadboost-{org}",
            organization_id=org,
            integration_source="leadboost",
            sender_name="LeadBoost Outreach",
            sender_org="LeadBoost",
            sender_email=sender_email,
            value_prop="placeholder value prop",
            proof_points=proof_points,
        )
        db.add(campaign)
        db.flush()
    contact = Contact(campaign_id=campaign.id, name="Jane", email=email, next_action_at=None)
    db.add(contact)
    db.flush()
    message = create_authorized_message(contact_id=contact.id, subject=subject, body=body)
    db.add(message)
    db.flush()
    mailbox_row = (
        seed_mailbox(
            db, org=org, email=sender_email, status=mailbox_status,
            smtp_host=smtp_host, smtp_port=smtp_port,
        )
        if mailbox
        else None
    )
    idem = idem or f"idem-{uuid.uuid4().hex[:10]}"
    dispatch = ExternalDispatch(
        organization_id=org,
        idempotency_key=idem,
        external_action_id="481",
        correlation_id="corr-1",
        campaign_id=campaign.id,
        contact_id=contact.id,
        message_id=message.id,
        mailbox_id=mailbox_row.id if mailbox_row else None,
        request_fingerprint="f" * 64,
        public_reference=uuid.uuid4().hex,
        state=state,
        claimed_by=claimed_by,
        claimed_at=claimed_at,
    )
    db.add(dispatch)
    db.flush()
    if commit:
        db.commit()
    return dispatch


def age(seconds: int) -> datetime:
    """A naive-UTC timestamp `seconds` in the past."""
    return naive_now() - timedelta(seconds=seconds)


class FakeSMTPServer:
    """
    Minimal deterministic SMTP server on an ephemeral loopback port.

    Counts messages it ACCEPTED at end-of-DATA and can be told to answer
    QUIT with a non-221 code (the F1 scenario: accepted, then QUIT fails).
    """

    def __init__(self, *, quit_reply: str = "221 bye"):
        outer = self
        self.accepted: list[str] = []
        self.quit_reply = quit_reply
        self.connections = 0

        class Handler(socketserver.StreamRequestHandler):
            def _w(self, line: str) -> None:
                self.wfile.write((line + "\r\n").encode())
                self.wfile.flush()

            def handle(self) -> None:
                outer.connections += 1
                self._w("220 fake ESMTP")
                while True:
                    raw = self.rfile.readline()
                    if not raw:
                        return
                    cmd = raw.decode().strip().upper()
                    if cmd.startswith(("EHLO", "HELO")):
                        self.wfile.write(b"250-fake\r\n250 AUTH PLAIN\r\n")
                        self.wfile.flush()
                    elif cmd.startswith("AUTH"):
                        self._w("235 ok")
                    elif cmd == "DATA":
                        self._w("354 go ahead")
                        lines = []
                        while True:
                            ln = self.rfile.readline()
                            if ln.strip() == b".":
                                break
                            lines.append(ln.decode())
                        outer.accepted.append("".join(lines))
                        self._w("250 queued")
                    elif cmd == "QUIT":
                        self._w(outer.quit_reply)
                        return
                    else:
                        self._w("250 ok")

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = Server(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=lambda: self._server.serve_forever(poll_interval=0.02), daemon=True)

    def __enter__(self) -> "FakeSMTPServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


# ---------------------------------------------------------------------------
# Worker-test doubles
# ---------------------------------------------------------------------------

class TrackingFactory:
    """Session factory that remembers every session it hands out, so a test
    can assert that NO session/transaction is open at the moment of SMTP."""

    def __init__(self, sessionmaker_):
        self.sm = sessionmaker_
        self.sessions: list = []

    def __call__(self):
        s = self.sm()
        self.sessions.append(s)
        return s

    def open_transactions(self) -> list:
        return [s for s in self.sessions if s.in_transaction()]


class FakeSender:
    """Stands in for send_email() at the worker's call site. Records every
    call (kwargs incl. message_id_header) and can run a hook mid-'SMTP'."""

    def __init__(self, outcome=None, error=None, raises=None, on_call=None):
        from mailer_agent.mail.sender import SendOutcome

        self.outcome = outcome or SendOutcome.SENT
        self.error = error
        self.raises = raises
        self.on_call = on_call
        self.calls: list[dict] = []

    def __call__(self, **kw):
        from mailer_agent.mail.sender import SendOutcome, SendResult

        self.calls.append(kw)
        if self.on_call:
            self.on_call(kw)
        if self.raises:
            raise self.raises
        return SendResult(
            self.outcome is SendOutcome.SENT, kw.get("message_id_header"), self.error, self.outcome
        )
