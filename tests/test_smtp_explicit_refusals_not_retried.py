"""
Explicit SMTP refusals (auth failed, recipient/sender refused, ...) must fail
fast: one attempt, outcome FAILED. They inherit from OSError /
SMTPResponseException, which are in the transient retry set, so without the
explicit exclusion they were retried three times. Genuinely transient errors
must still be retried.
"""

import smtplib

import pytest
from tenacity import wait_none

from mailer_agent.mail import sender
from mailer_agent.mail.sender import SendOutcome


class _FakeSMTP:
    constructed = 0
    login_error: Exception | None = None

    def __init__(self, *a, **k):
        type(self).constructed += 1
        if type(self).connect_error is not None:
            raise type(self).connect_error

    connect_error: Exception | None = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, **k):
        pass

    def login(self, *a):
        if type(self).login_error is not None:
            raise type(self).login_error

    def sendmail(self, *a):
        pass


@pytest.fixture()
def fake_smtp(monkeypatch):
    _FakeSMTP.constructed, _FakeSMTP.login_error, _FakeSMTP.connect_error = 0, None, None
    monkeypatch.setattr(sender.smtplib, "SMTP", _FakeSMTP)
    monkeypatch.setattr(sender._smtp_send_with_retry.retry, "wait", wait_none())
    monkeypatch.setattr(sender.settings, "live_sending_enabled", True)
    return _FakeSMTP


def _send():
    return sender.send_email(
        to_email="a@example.com", from_email="s@example.com", from_name="S", subject="x", body_text="y"
    )


@pytest.mark.parametrize("error", [
    smtplib.SMTPAuthenticationError(535, b"bad credentials"),
    smtplib.SMTPHeloError(500, b"no"),
    smtplib.SMTPSenderRefused(550, b"no", "s@example.com"),
])
def test_explicit_refusal_is_one_attempt_and_failed(fake_smtp, error):
    fake_smtp.login_error = error
    result = _send()
    assert result.outcome is SendOutcome.FAILED
    assert fake_smtp.constructed == 1


def test_transient_connect_error_is_still_retried(fake_smtp):
    fake_smtp.connect_error = smtplib.SMTPConnectError(421, b"try later")
    result = _send()
    assert result.outcome is SendOutcome.FAILED
    assert fake_smtp.constructed == 3
