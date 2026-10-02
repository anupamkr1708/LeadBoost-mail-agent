"""
Primary-exception precedence in mail/sender.py.

Invariant: when the body of the SMTP `with` block fails AND smtplib's context
exit (QUIT/close) then also fails, the BODY's exception is the one that
decides the outcome. Python normally lets the exit exception replace it; the
replacement is an OSError/SMTPException (in the retry set), which used to
turn:

    ambiguous sendmail failure  (UNKNOWN, must not be retried)
        -> replaced by QUIT error -> retried 3x -> FAILED

Outcome table pinned here (each with a failing QUIT):

    connect/TLS/login failure ........ original failure preserved   -> FAILED (existing retry)
    sendmail explicit rejection ...... original rejection preserved -> FAILED (existing retry)
    sendmail ambiguous failure ....... AmbiguousSendError preserved -> UNKNOWN, NOT retried
    sendmail success ................. SENT, never retried          (see test_sender_quit_containment.py)

No retry policy is changed: attempt counts equal the counts with a healthy
QUIT. Deterministic in-process fake SMTP server, no network.
"""

from __future__ import annotations

import smtplib

import pytest
import tenacity

from mailer_agent.mail import sender
from mailer_agent.mail.sender import AmbiguousSendError, SendOutcome
from tests.dispatch_support import FakeSMTPServer

QUIT_MARKER = "QUITREPLYMARKER"
KW = dict(
    to_email="prospect@example.com",
    from_email="outreach@sender.example.org",
    from_name="Outreach",
    subject="Hello",
    body_text="Body\n",
)


@pytest.fixture()
def wire(monkeypatch):
    """Real sender -> fake server whose QUIT reply is a failure carrying a marker."""
    servers = []

    def _make(quit_reply=f"421 {QUIT_MARKER}", use_tls=False):
        srv = FakeSMTPServer(quit_reply=quit_reply)
        srv.__enter__()
        servers.append(srv)
        for k, v in dict(smtp_host="127.0.0.1", smtp_port=srv.port, smtp_use_tls=use_tls,
                         smtp_username="u", smtp_password="p", live_sending_enabled=True).items():
            monkeypatch.setattr(sender.settings, k, v)
        return srv

    monkeypatch.setattr(sender._smtp_send_with_retry.retry, "wait", tenacity.wait_none())
    yield _make
    for s in servers:
        s.__exit__(None, None, None)


def _counting(monkeypatch, name, exc_factory):
    calls = []

    def fn(self, *a, **k):
        calls.append(1)
        raise exc_factory()

    monkeypatch.setattr(smtplib.SMTP, name, fn)
    return calls


# ------------------------------------------------ ambiguous sendmail + bad QUIT

@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: smtplib.SMTPDataError(451, b"aborted after DATA"),
        lambda: smtplib.SMTPServerDisconnected("Connection unexpectedly closed during DATA"),
        lambda: TimeoutError("timed out while sending DATA"),
        lambda: RuntimeError("something unexpected mid-transmission"),
    ],
    ids=["data-error", "disconnected", "timeout", "unexpected"],
)
def test_ambiguous_sendmail_plus_failing_quit_stays_unknown_and_is_not_retried(wire, monkeypatch, exc_factory):
    srv = wire()
    calls = _counting(monkeypatch, "sendmail", exc_factory)

    result = sender.send_email(**KW)

    assert result.outcome == SendOutcome.UNKNOWN and result.success is False
    assert len(calls) == 1                      # exactly one attempt -- no retry
    assert srv.connections == 1
    assert QUIT_MARKER not in (result.error or "")   # the cleanup error did not replace the cause
    assert "delivery status unknown" in result.error


def test_ambiguous_failure_keeps_its_original_cause_chain(wire, monkeypatch):
    wire()
    original = smtplib.SMTPDataError(451, b"aborted after DATA")
    _counting(monkeypatch, "sendmail", lambda: original)

    with pytest.raises(AmbiguousSendError) as info:
        sender._smtp_send_with_retry.__wrapped__(_msg(), KW["from_email"], KW["to_email"])

    assert info.value.original is original
    assert info.value.__cause__ is original


# ------------------------------------------- explicit rejection + failing QUIT

def test_explicit_rejection_plus_failing_quit_preserves_the_rejection(wire, monkeypatch):
    srv = wire()
    calls = _counting(
        monkeypatch, "sendmail",
        lambda: smtplib.SMTPRecipientsRefused({"prospect@example.com": (550, b"no such user")}),
    )

    result = sender.send_email(**KW)

    assert result.outcome == SendOutcome.FAILED
    assert "no such user" in result.error and QUIT_MARKER not in result.error
    assert len(calls) == 1                       # explicit refusal: one attempt, not retried
    assert srv.accepted == []


# ---------------------------------------- connect/TLS/login failure + bad QUIT

def test_login_failure_plus_failing_quit_preserves_the_login_failure(wire, monkeypatch):
    srv = wire()
    calls = _counting(monkeypatch, "login", lambda: smtplib.SMTPAuthenticationError(535, b"bad credentials"))

    result = sender.send_email(**KW)

    assert result.outcome == SendOutcome.FAILED
    assert "bad credentials" in result.error and QUIT_MARKER not in result.error
    assert len(calls) == 1 and srv.accepted == []


def test_tls_failure_plus_failing_quit_preserves_the_tls_failure(wire, monkeypatch):
    srv = wire(use_tls=True)
    calls = _counting(monkeypatch, "starttls", lambda: smtplib.SMTPException("STARTTLS negotiation failed"))

    result = sender.send_email(**KW)

    assert result.outcome == SendOutcome.FAILED
    assert "STARTTLS negotiation failed" in result.error and QUIT_MARKER not in result.error
    assert len(calls) == 3 and srv.accepted == []


def test_connect_failure_is_unchanged(monkeypatch):
    """No connection => nothing to clean up; the original error is reported."""
    monkeypatch.setattr(sender._smtp_send_with_retry.retry, "wait", tenacity.wait_none())
    with FakeSMTPServer() as srv:
        port = srv.port
    for k, v in dict(smtp_host="127.0.0.1", smtp_port=port, smtp_use_tls=False,
                     smtp_username="u", smtp_password="p", live_sending_enabled=True).items():
        monkeypatch.setattr(sender.settings, k, v)   # server is gone: connection refused

    result = sender.send_email(**KW)
    assert result.outcome == SendOutcome.FAILED
    assert "refused" in result.error.lower() or "Errno" in result.error


# ------------------------------------------------------------- edge / control

def test_non_exception_baseexception_in_body_is_not_replaced_by_cleanup_error(wire, monkeypatch):
    wire()
    _counting(monkeypatch, "sendmail", lambda: KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        sender._smtp_send_with_retry.__wrapped__(_msg(), KW["from_email"], KW["to_email"])


def test_healthy_paths_are_untouched(wire):
    srv = wire(quit_reply="221 bye")
    assert sender.send_email(**KW).outcome == SendOutcome.SENT
    assert len(srv.accepted) == 1


def _msg():
    from email.mime.text import MIMEText

    m = MIMEText("Body", "plain")
    m["Subject"], m["From"], m["To"] = "s", KW["from_email"], KW["to_email"]
    return m
