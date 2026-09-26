"""
Local SMTP protocol integration test (spec section 33/58 -- "controlled
SMTP" / "SENT/FAILED/UNKNOWN semantics").

Honesty check, stated plainly, matching the convention in
tests/evaluation/test_live_llm_semantic_quality.py: this test does NOT
prove delivery to a real mailbox, and it is NOT the "one mailbox you
control" live test the spec's section 58 describes -- this sandbox has
no outbound SMTP/network access to a real provider, and no real mailbox
credentials. What it DOES prove, for real, over an actual TCP socket on
localhost (not a mock of smtplib): that mailer_agent.mail.sender.send_email
correctly speaks SMTP protocol end to end -- connects, transmits a
properly-headered MIME message, and correctly classifies the outcome as
SENT when the server accepts it and FAILED when the connection is
refused outright. That is real, meaningful coverage of the module's own
core claim (the SENT/FAILED/UNKNOWN distinction in sender.py's module
docstring) that a mocked-smtplib unit test cannot fully provide, even
though it still isn't the live-mailbox test the spec ultimately wants.

Auth is stubbed (smtplib.SMTP.login patched to a no-op) purely because
configuring aiosmtpd's AUTH mechanism is orthogonal to what this test is
actually proving; everything else -- TCP connect, EHLO, MAIL FROM,
RCPT TO, DATA, and the exact bytes received -- goes over a real local
socket.
"""

from __future__ import annotations

import asyncio
import smtplib
import threading

import pytest
from aiosmtpd.controller import Controller
from aiosmtpd.handlers import Message as MessageHandler

from mailer_agent.mail import sender

pytestmark = pytest.mark.integration
# Not because this needs a real external service (it's a fully local
# aiosmtpd server, no network egress required) -- but because it spins up
# real sockets and includes real tenacity backoff sleeps on the
# connection-refused path (a few seconds), which doesn't belong in the
# default fast unit-test loop. Run explicitly: pytest -o addopts=""
# tests/test_smtp_local_integration.py


class _CapturingHandler(MessageHandler):
    def __init__(self):
        super().__init__()
        self.received = []

    def handle_message(self, message):
        self.received.append(message)


@pytest.fixture
def local_smtp_server():
    handler = _CapturingHandler()
    # A fixed high port, not 0/ephemeral -- aiosmtpd's own startup
    # self-check connects using the port attribute as set *before*
    # start(), which an OS-assigned ephemeral port doesn't populate in
    # time for that check.
    controller = Controller(handler, hostname="127.0.0.1", port=8825)
    controller.start()
    try:
        yield controller, handler
    finally:
        controller.stop()


@pytest.fixture
def patched_settings(monkeypatch, local_smtp_server):
    controller, _ = local_smtp_server
    monkeypatch.setattr(sender.settings, "smtp_host", controller.hostname)
    monkeypatch.setattr(sender.settings, "smtp_port", controller.port)
    monkeypatch.setattr(sender.settings, "smtp_use_tls", False)
    monkeypatch.setattr(sender.settings, "smtp_username", "test")
    monkeypatch.setattr(sender.settings, "smtp_password", "test")
    monkeypatch.setattr(sender.settings, "live_sending_enabled", True)
    # See module docstring: AUTH is stubbed, everything else is real.
    monkeypatch.setattr(smtplib.SMTP, "login", lambda self, *a, **kw: (235, b"OK"))
    return controller


def test_real_smtp_round_trip_produces_sent_with_correct_headers(local_smtp_server, patched_settings):
    controller, handler = local_smtp_server

    result = sender.send_email(
        to_email="prospect@example.com",
        from_email="jordan@testcorp.example.com",
        from_name="Jordan",
        subject="Following up",
        body_text="Does Thursday at 2pm work for a quick call?",
        reply_to="jordan@testcorp.example.com",
        in_reply_to_header="<abc123@testcorp.example.com>",
    )

    assert result.success is True
    assert result.outcome == sender.SendOutcome.SENT
    assert result.message_id is not None

    assert len(handler.received) == 1
    received = handler.received[0]
    assert received["Subject"] == "Following up"
    assert received["To"] == "prospect@example.com"
    # The exact same Message-ID minted before the network call was what
    # actually went out on the wire -- proves the "generate before
    # attempting" invariant sender.py's docstring describes isn't just
    # true in the return value, it's true in what the server received.
    assert received["Message-ID"] == result.message_id
    assert received["In-Reply-To"] == "<abc123@testcorp.example.com>"
    assert received["References"] == "<abc123@testcorp.example.com>"
    body = received.get_payload()
    if isinstance(body, list):
        body = body[0].get_payload()
    assert "Thursday at 2pm" in body


def test_real_connection_refused_produces_failed_not_sent(monkeypatch):
    """
    Point at a port nothing is listening on (a real, unmocked connection
    failure) and confirm this is classified as FAILED -- unambiguous,
    safe to retry -- not SENT and not UNKNOWN. UNKNOWN is reserved for
    failures during/after the DATA phase (see
    test_mid_transmission_failure_is_unknown_not_failed below for that
    path via a mocked sendmail(), which a refused-connection scenario
    can't exercise since it never gets that far).
    """
    monkeypatch.setattr(sender.settings, "smtp_host", "127.0.0.1")
    monkeypatch.setattr(sender.settings, "smtp_port", 1)  # nothing listens on port 1
    monkeypatch.setattr(sender.settings, "smtp_use_tls", False)
    monkeypatch.setattr(sender.settings, "live_sending_enabled", True)

    result = sender.send_email(
        to_email="prospect@example.com",
        from_email="jordan@testcorp.example.com",
        from_name="Jordan",
        subject="Test",
        body_text="Test body.",
    )

    assert result.success is False
    assert result.outcome == sender.SendOutcome.FAILED
    assert result.error is not None


def test_mid_transmission_failure_is_unknown_not_failed(local_smtp_server, patched_settings, monkeypatch):
    """
    Spec sections 29-30: a connection drop DURING/AFTER sendmail() (the
    DATA phase) must be UNKNOWN, not FAILED -- the message might have
    actually been delivered, so it must never be silently treated as
    safe to auto-retry. This module had zero test coverage anywhere in
    the repo before this file (confirmed by searching for send_email/
    SendResult/AmbiguousSendError usage across tests/) -- so this is the
    first real check that _smtp_send_with_retry's own exception
    classification (the whole reason this module distinguishes three
    outcomes instead of two) actually does what its docstring claims.

    sendmail() itself is mocked here (a real socket-level DATA-phase
    interruption is possible to engineer but not worth the added
    complexity when the thing under test -- exception classification --
    doesn't care how the exception arrived); the connection/auth/EHLO
    path above it is still the real local server from local_smtp_server.
    """
    def _raise_mid_transmission(*args, **kwargs):
        raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed during DATA")

    monkeypatch.setattr(smtplib.SMTP, "sendmail", _raise_mid_transmission)

    result = sender.send_email(
        to_email="prospect@example.com",
        from_email="jordan@testcorp.example.com",
        from_name="Jordan",
        subject="Test",
        body_text="Test body.",
    )

    assert result.success is False
    assert result.outcome == sender.SendOutcome.UNKNOWN, (
        "A failure during sendmail() itself must be UNKNOWN, not FAILED -- "
        "the message may have actually been delivered."
    )


def test_explicit_recipient_refusal_is_failed_not_unknown(local_smtp_server, patched_settings, monkeypatch):
    """The other half of the sendmail()-time distinction: the server
    explicitly saying no (not a connection drop) IS unambiguous and
    stays FAILED even though it surfaces from inside sendmail()."""
    def _raise_refused(*args, **kwargs):
        raise smtplib.SMTPRecipientsRefused({"prospect@example.com": (550, b"No such user")})

    monkeypatch.setattr(smtplib.SMTP, "sendmail", _raise_refused)

    result = sender.send_email(
        to_email="prospect@example.com",
        from_email="jordan@testcorp.example.com",
        from_name="Jordan",
        subject="Test",
        body_text="Test body.",
    )

    assert result.success is False
    assert result.outcome == sender.SendOutcome.FAILED
