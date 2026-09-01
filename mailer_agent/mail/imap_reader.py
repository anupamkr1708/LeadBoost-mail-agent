"""
IMAP polling for reply detection.

Deliberately kept separate from anything DB-related: this module's only
job is "connect, fetch unseen messages since some point, parse them into
plain structures". Correlating a parsed email back to a Contact/thread
and deciding what to do about it lives in mail/reply_handler.py, which
keeps this module trivially testable without a real mailbox.
"""

from __future__ import annotations

import email
import imaplib
import logging
import re
from dataclasses import dataclass
from email.header import decode_header
from email.utils import parseaddr

from mailer_agent.config import get_settings

logger = logging.getLogger("mailer_agent.mail.imap")

settings = get_settings()

_RE_PREFIX = re.compile(r"^\s*re\s*:\s*", re.IGNORECASE)


def as_reply_subject(original_subject: str | None) -> str:
    """'following up' -> 'Re: following up'; 'Re: following up' -> unchanged
    (never stacks a second 'Re:' the way naive prepending does)."""
    if not original_subject:
        return "Re: our conversation"
    if _RE_PREFIX.match(original_subject):
        return original_subject.strip()
    return f"Re: {original_subject.strip()}"


@dataclass
class InboundEmail:
    from_email: str
    subject: str
    body_text: str
    message_id: str | None
    in_reply_to: str | None
    references: list[str]
    to_email: str | None = None  # The inbox this was sent to (for tenant resolution)


def _decode(value: str | None) -> str:
    if not value:
        return ""
    parts = decode_header(value)
    decoded = ""
    for text, enc in parts:
        if isinstance(text, bytes):
            decoded += text.decode(enc or "utf-8", errors="replace")
        else:
            decoded += text
    return decoded


def _extract_plain_text(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition") or "")
            if content_type == "text/plain" and "attachment" not in disposition:
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True) or b""
                return payload.decode(charset, errors="replace")
        return ""
    charset = msg.get_content_charset() or "utf-8"
    payload = msg.get_payload(decode=True) or b""
    return payload.decode(charset, errors="replace")


def fetch_unseen_replies() -> list[InboundEmail]:
    """
    Connects, fetches all UNSEEN messages, marks them Seen, and returns
    them parsed. UNSEEN (rather than tracking a last-checked timestamp)
    is used deliberately: it's IMAP-server-authoritative and survives
    this process restarting without losing or double-processing mail.
    """
    if not settings.imap_username or not settings.imap_password:
        logger.debug("IMAP not configured, skipping reply poll")
        return []

    results: list[InboundEmail] = []
    try:
        conn = imaplib.IMAP4_SSL(settings.imap_host, settings.imap_port)
        conn.login(settings.imap_username, settings.imap_password)
        conn.select("INBOX")

        status, data = conn.search(None, "UNSEEN")
        if status != "OK":
            logger.warning("IMAP search failed: %s", status)
            return []

        uids = data[0].split()
        for uid in uids:
            status, msg_data = conn.fetch(uid, "(RFC822)")
            if status != "OK" or not msg_data or not msg_data[0]:
                continue
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)

            _, from_addr = parseaddr(msg.get("From", ""))
            references_raw = msg.get("References", "")
            references = references_raw.split() if references_raw else []

            results.append(
                InboundEmail(
                    from_email=from_addr.lower(),
                    subject=_decode(msg.get("Subject")),
                    body_text=_extract_plain_text(msg).strip(),
                    message_id=msg.get("Message-ID"),
                    in_reply_to=msg.get("In-Reply-To"),
                    references=references,
                )
            )

        conn.close()
        conn.logout()
    except Exception as e:
        logger.error("IMAP poll failed: %s", e)

    return results
