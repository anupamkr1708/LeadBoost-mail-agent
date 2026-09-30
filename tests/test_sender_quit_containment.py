"""
F1: an exception raised while CLOSING the SMTP connection -- after the
server has already accepted the message -- must not trigger another send.

Before the fix, smtplib.SMTP.__exit__ raising on a non-221 reply to QUIT
propagated out of _smtp_send_with_retry; the exception is in the retry set,
so tenacity re-ran the whole send (the server accepted the message 3 times)
and send_email() finally reported FAILED for a message that was delivered.

Exceptions raised by sendmail() itself, and everything before it (connect,
TLS, login), keep their existing retry / outcome behaviour -- pinned below.
The fake server is deterministic, in-process, on an ephemeral loopback port.
"""

from __future__ import annotations

import smtplib

import pytest
import tenacity

from mailer_agent.mail import sender
from mailer_agent.mail.sender import SendOutcome
from tests.dispatch_support import FakeSMTPServer

KW = dict(
    to_email="prospect@example.com",
    from_email="outreach@sender.example.org",
    from_name="Outreach",
    subject="Hello",
    body_text="Body\n",
)


@pytest.fixture()
def wire(monkeypatch):
    """Point the real sender at a fake server; make retries instantaneous."""
    def _make(quit_reply="221 bye"):
        srv = FakeSMTPServer(quit_reply=quit_reply)
        srv.__enter__()
        s = sender.settings
        for k, v in dict(smtp_host="127.0.0.1", smtp_port=srv.port, smtp_use_tls=False,
                         smtp_username="u", smtp_password="p", live_sending_enabled=True).items():
            monkeypatch.setattr(s, k, v)
        return srv

    monkeypatch.setattr(sender._smtp_send_with_retry.retry, "wait", tenacity.wait_none())
    servers = []
    yield lambda **kw: servers.append(_make(**kw)) or servers[-1]
    for srv in servers:
        srv.__exit__(None, None, None)


# ------------------------------------------------------------- the F1 fix

@pytest.mark.parametrize("quit_reply", ["421 service shutting down", "500 unexpected", "554 nope"])
def test_accepted_message_with_failing_quit_is_sent_exactly_once(wire, quit_reply):
    srv = wire(quit_reply=quit_reply)
    result = sender.send_email(**KW)

    assert result.outcome == SendOutcome.SENT and result.success is True
    assert len(srv.accepted) == 1              # exactly one delivery
    assert srv.connections == 1                # and no retry connection was even opened
    assert result.message_id in srv.accepted[0]


def test_quit_timing_out_after_acceptance_is_also_contained(wire, monkeypatch):
    srv = wire()
    real_docmd = smtplib.SMTP.docmd

    def docmd(self, cmd, args=""):
        if cmd.upper() == "QUIT":
            raise TimeoutError("timed out waiting for QUIT reply")   # an OSError => in the retry set
        return real_docmd(self, cmd, args)

    monkeypatch.setattr(smtplib.SMTP, "docmd", docmd)
    result = sender.send_email(**KW)
    assert result.outcome == SendOutcome.SENT
    assert len(srv.accepted) == 1 and srv.connections == 1


def test_supplied_message_id_survives_a_contained_quit_failure(wire):
    srv = wire(quit_reply="421 bye")
    result = sender.send_email(**KW, message_id_header="<fixed.id@sender.example.org>")
    assert result.outcome == SendOutcome.SENT and result.message_id == "<fixed.id@sender.example.org>"
    assert "Message-ID: <fixed.id@sender.example.org>" in srv.accepted[0]


# ------------------------------------------- behaviour that must NOT change

def test_clean_send_is_unchanged(wire):
    srv = wire()
    assert sender.send_email(**KW).outcome == SendOutcome.SENT
    assert len(srv.accepted) == 1


def test_login_failure_is_still_retried_three_times_then_failed(wire, monkeypatch):
    srv = wire()
    calls = []

    def bad_login(self, *a, **k):
        calls.append(1)
        raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

    monkeypatch.setattr(smtplib.SMTP, "login", bad_login)
    result = sender.send_email(**KW)
    assert result.outcome == SendOutcome.FAILED
    assert len(calls) == 3 and srv.accepted == []


def test_explicit_refusal_from_sendmail_is_still_failed_and_retried(wire, monkeypatch):
    srv = wire()
    calls = []

    def refuse(self, *a, **k):
        calls.append(1)
        raise smtplib.SMTPRecipientsRefused({"prospect@example.com": (550, b"no such user")})

    monkeypatch.setattr(smtplib.SMTP, "sendmail", refuse)
    result = sender.send_email(**KW)
    assert result.outcome == SendOutcome.FAILED
    assert len(calls) == 3 and srv.accepted == []


def test_ambiguous_sendmail_failure_is_still_unknown_and_not_retried(wire, monkeypatch):
    srv = wire()
    calls = []

    def drop(self, *a, **k):
        calls.append(1)
        raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed during DATA")

    monkeypatch.setattr(smtplib.SMTP, "sendmail", drop)
    result = sender.send_email(**KW)
    assert result.outcome == SendOutcome.UNKNOWN
    assert len(calls) == 1
