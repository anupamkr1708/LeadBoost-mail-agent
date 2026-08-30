"""
Outbound sending over SMTP.

The one detail that matters most and is easiest to get wrong: proper
RFC 5322 threading headers. Without a stable Message-ID on the message
we send, and In-Reply-To/References on follow-ups, there is no reliable
way to match an inbound reply back to the right contact later except
fuzzy subject matching -- which breaks the moment someone edits the
subject line. So every send here generates and stores its own
Message-ID, and every follow-up/reply references the most recent prior
Message-ID in the thread.
"""

from __future__ import annotations

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


class SendResult:
    def __init__(self, success: bool, message_id: str | None = None, error: str | None = None):
        self.success = success
        self.message_id = message_id
        self.error = error


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
        return SendResult(success=True, message_id=msg_id)

    try:
        _smtp_send_with_retry(msg, from_email, to_email)
        logger.info("Sent email to %s | msg_id=%s", to_email, msg_id)
        return SendResult(success=True, message_id=msg_id)
    except Exception as e:
        logger.error("Failed to send email to %s: %s", to_email, e)
        return SendResult(success=False, message_id=msg_id, error=str(e))


# Only retried on transient/network-level failures. Authentication
# failures and recipient/sender-refused errors are permanent for this
# message and would just waste time (and risk looking like a retry
# storm to the SMTP server) if retried, so they're deliberately not in
# this set -- they fail fast and get logged instead.
_TRANSIENT_SMTP_ERRORS = (
    smtplib.SMTPServerDisconnected,
    smtplib.SMTPConnectError,
    smtplib.SMTPResponseException,
    ConnectionError,
    TimeoutError,
    OSError,
)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type(_TRANSIENT_SMTP_ERRORS),
)
def _smtp_send_with_retry(msg: MIMEMultipart, from_email: str, to_email: str) -> None:
    context = ssl.create_default_context()
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
        if settings.smtp_use_tls:
            server.starttls(context=context)
        server.login(settings.smtp_username, settings.smtp_password)
        server.sendmail(from_email, [to_email], msg.as_string())
