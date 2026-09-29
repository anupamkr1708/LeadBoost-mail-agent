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
from mailer_agent.models import (
    Campaign,
    Contact,
    ExternalDispatch,
    ExternalDispatchState,
)


def naive_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
    idem = idem or f"idem-{uuid.uuid4().hex[:10]}"
    dispatch = ExternalDispatch(
        organization_id=org,
        idempotency_key=idem,
        external_action_id="481",
        correlation_id="corr-1",
        campaign_id=campaign.id,
        contact_id=contact.id,
        message_id=message.id,
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
