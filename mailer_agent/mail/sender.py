"""
Outbound sending over SMTP.

The one detail that matters most and is easiest to get wrong: proper
RFC 5322 threading headers. Without a stable Message-ID on the message
we send, and In-Reply-To/References on follow-ups, there is no reliable
way to match an inbound reply back to the right contact later except
fuzzy subject matching -- which breaks the moment someone edits the
subject line. So every send here generates and stores its own
Message-ID *before* the SMTP call is even attempted, and every
follow-up/reply references the most recent prior Message-ID in the
thread.

Ambiguous outcomes
-------------------
smtplib cannot always tell us whether the receiving server actually
accepted a message before a connection problem occurred. If the
process crashes or the socket drops *during or after* the ``DATA``
phase, the message may or may not have actually been delivered -- and
guessing wrong in either direction is unsafe:
  - Treating an ambiguous send as "sent" risks silently losing a
    message that never went out.
  - Treating it as "failed" (and therefore safe to retry) risks a
    duplicate send to the prospect.

So this module distinguishes three outcomes, not two:
  - ``SendOutcome.SENT``    -- the SMTP server accepted the message.
  - ``SendOutcome.FAILED``  -- the send definitely did not happen
    (connection/auth/HELO failed, or the server explicitly refused the
    recipient/sender before or without transmitting the message body).
    Safe to retry.
  - ``SendOutcome.UNKNOWN`` -- the failure happened during or after the
    ``sendmail()`` call itself, so we cannot tell whether the message
    reached the prospect. Callers MUST NOT blindly auto-resend an
    UNKNOWN outcome -- it requires a human to check (or a
    provider-side dedup signal) before anything is retried.
"""

from __future__ import annotations

import enum
import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import make_msgid

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mailer_agent.config import get_settings

logger = logging.getLogger("mailer_agent.mail.sender")

settings = get_settings()


class SendOutcome(str, enum.Enum):
    SENT = "sent"
    FAILED = "failed"       # definitely did not send -- safe to retry
    UNKNOWN = "unknown"     # ambiguous -- do NOT auto-resend


class AmbiguousSendError(Exception):
    """
    Raised when an error occurs during/after the SMTP DATA phase, where
    we can no longer be certain the message wasn't delivered.  Distinct
    from ordinary connection/auth failures, which are unambiguous.
    """
    def __init__(self, message: str, original: Exception):
        super().__init__(message)
        self.original = original


class SendResult:
    def __init__(
        self,
        success: bool,
        message_id: str | None = None,
        error: str | None = None,
        outcome: SendOutcome = SendOutcome.SENT,
    ):
        # `success` is kept for backward compatibility with existing
        # callers: True only for a confirmed send. Both FAILED and
        # UNKNOWN report success=False -- callers that need to tell
        # them apart (to decide whether a retry is safe) should read
        # `outcome` instead of branching on `success` alone.
        self.success = success
        self.message_id = message_id
        self.error = error
        self.outcome = outcome


def send_email(
    *,
    to_email: str,
    from_email: str,
    from_name: str,
    subject: str,
    body_text: str,
    reply_to: str | None = None,
    in_reply_to_header: str | None = None,
    references_header: str | None = None,
) -> SendResult:
    """
    Sends a plain-text email (deliberately plain text, not HTML -- plain
    text from a real-looking personal address reads as human and avoids
    the tracking-pixel/HTML-template signals spam filters key on for
    cold outreach).

    The Message-ID is generated and returned to the caller *before* any
    network call is attempted, so it can be persisted first: if the
    process crashes mid-send, the stable Message-ID we intended to use
    is already on record and a retry can be recognized as the same
    logical message rather than minted as a new one.
    """
    domain = from_email.split("@")[-1] if "@" in from_email else "localhost"
    msg_id = make_msgid(domain=domain)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{from_name} <{from_email}>"
    msg["To"] = to_email
    msg["Message-ID"] = msg_id
    if reply_to:
        msg["Reply-To"] = reply_to
    if in_reply_to_header:
        msg["In-Reply-To"] = in_reply_to_header
    if references_header:
        msg["References"] = references_header
    elif in_reply_to_header:
        msg["References"] = in_reply_to_header

    msg.attach(MIMEText(body_text, "plain"))

    if not settings.live_sending_enabled:
        logger.info("[DRY RUN] Would send to %s | subject=%r | msg_id=%s", to_email, subject, msg_id)
        return SendResult(success=True, message_id=msg_id, outcome=SendOutcome.SENT)

    try:
        _smtp_send_with_retry(msg, from_email, to_email)
        logger.info("Sent email to %s | msg_id=%s", to_email, msg_id)
        return SendResult(success=True, message_id=msg_id, outcome=SendOutcome.SENT)
    except AmbiguousSendError as e:
        # Error occurred during/after the SMTP DATA phase -- we cannot
        # tell whether the prospect actually received this message.
        # Explicit UNKNOWN, never silently treated as failed-safe-to-retry.
        logger.error(
            "AMBIGUOUS send outcome to %s (msg_id=%s): %s -- treating as UNKNOWN, "
            "not auto-retrying",
            to_email, msg_id, e,
        )
        return SendResult(success=False, message_id=msg_id, error=str(e), outcome=SendOutcome.UNKNOWN)
    except Exception as e:
        # Failure happened before/without an attempt to transmit the
        # message body (connection, TLS, auth, or an explicit
        # recipient/sender refusal) -- definitely did not send.
        logger.error("Failed to send email to %s: %s", to_email, e)
        return SendResult(success=False, message_id=msg_id, error=str(e), outcome=SendOutcome.FAILED)


# Only retried on transient/network-level failures that occur *before*
# the message body is transmitted. Authentication failures and
# recipient/sender-refused errors are permanent for this message and
# would just waste time (and risk looking like a retry storm to the
# SMTP server) if retried, so they're deliberately not in this set --
# they fail fast and get logged instead.
_TRANSIENT_SMTP_ERRORS = (
    smtplib.SMTPServerDisconnected,
    smtplib.SMTPConnectError,
    smtplib.SMTPResponseException,
    ConnectionError,
    TimeoutError,
    OSError,
)

# Exceptions sendmail() raises when the server has explicitly told us
# it will not accept the message. These are unambiguous failures (the
# server responded, and said no) even though they surface from inside
# the sendmail() call -- not ambiguity about whether it was delivered.
_EXPLICIT_REFUSAL_ERRORS = (
    smtplib.SMTPRecipientsRefused,
    smtplib.SMTPSenderRefused,
    smtplib.SMTPHeloError,
    smtplib.SMTPAuthenticationError,
    smtplib.SMTPNotSupportedError,
)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type(_TRANSIENT_SMTP_ERRORS),
)
def _smtp_send_with_retry(msg: MIMEMultipart, from_email: str, to_email: str) -> None:
    """
    Connect, authenticate, and send.

    Only the ``sendmail()`` call itself is treated as a potential
    ambiguous-outcome zone: a failure while connecting, negotiating
    TLS, or authenticating happens strictly before any message content
    is transmitted, so those are unambiguous failures and remain
    retryable transient errors. A failure raised *by* sendmail() that
    isn't an explicit server refusal (e.g. the connection drops mid
    DATA-phase) is re-raised as ``AmbiguousSendError`` so the caller
    records UNKNOWN instead of a plain retryable FAILED.
    """
    context = ssl.create_default_context()
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
        if settings.smtp_use_tls:
            server.starttls(context=context)
        server.login(settings.smtp_username, settings.smtp_password)

        try:
            server.sendmail(from_email, [to_email], msg.as_string())
        except _EXPLICIT_REFUSAL_ERRORS:
            # The server responded and explicitly rejected the message --
            # unambiguous failure, not a delivery-uncertain condition.
            raise
        except Exception as e:
            raise AmbiguousSendError(
                f"Error during SMTP transmission -- delivery status unknown: {e}", e
            ) from e
