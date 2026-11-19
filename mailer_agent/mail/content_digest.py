"""
Exact-content fingerprint for managed-reply approval.

A LeadBoost user approves the subject and body they were SHOWN. This module defines, in one place, the
digest that proves the message Mailer is about to transmit is that same text. The draft-read endpoint, the
release endpoint and the SMTP worker all call it; nothing else computes the digest.

CANONICAL SERIALIZATION (version 1 -- changing any of this changes every digest, so it is versioned):

    payload = {"body": <body str>, "subject": <subject str>}
    bytes   = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest  = sha256(bytes).hexdigest()           # 64 lowercase hex characters

* JSON gives unambiguous field boundaries: ("a", "bc") and ("ab", "c") can never collide, which plain
  concatenation would allow.
* Keys are sorted and separators are fixed, so the bytes do not depend on dict ordering or whitespace.
* ``ensure_ascii=False`` + UTF-8: the digest is over the real characters, not over a \\uXXXX spelling.
* A ``None`` subject is hashed as the empty string -- exactly what ``exact_message.build_exact_send_input``
  hands to the sender, so "the digest" and "what is transmitted" cannot disagree about None vs "".
* NO normalization of any kind: line endings, trailing whitespace and Unicode normal form are all preserved
  and all affect the digest. Approving "a\\n" does not approve "a\\r\\n".
* Text that cannot be encoded as UTF-8 (a lone surrogate) raises ``DigestError``; callers fail closed.

What the digest covers: the subject and body TEXT. It does not cover headers Mailer generates at send time
(From, Message-ID, Date) or MIME transfer encoding; the threading headers are derived from the stored parent
and are covered by tests, not by this digest.

This module imports only the standard library, so the API layer and the worker can use it without pulling in
the sender or the LLM stack.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Optional

DIGEST_VERSION = 1
# The draft that the preview shows is the draft that is hashed and sent, so a draft too large to preview in
# full must never be approvable. Callers refuse (never truncate) above these limits.
MAX_SUBJECT_CHARS = 998          # RFC 5322 line-length ceiling for a header line
MAX_BODY_CHARS = 20_000

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class DigestError(ValueError):
    """The content cannot be fingerprinted unambiguously. Fail closed."""


class DraftTooLarge(DigestError):
    """Subject or body exceeds the preview limit; the draft can be neither previewed nor approved."""


def canonical_bytes(subject: Optional[str], body: str) -> bytes:
    if subject is not None and not isinstance(subject, str):
        raise DigestError("subject must be a string or None")
    if not isinstance(body, str):
        raise DigestError("body must be a string")
    payload = {"body": body, "subject": subject if subject is not None else ""}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DigestError("content is not valid UTF-8 text") from exc


def content_sha256(subject: Optional[str], body: str) -> str:
    """64 lowercase hex characters; deterministic for identical (subject, body)."""
    return hashlib.sha256(canonical_bytes(subject, body)).hexdigest()


def is_valid_digest(value: object) -> bool:
    """True only for a string of exactly 64 lowercase hex characters (never raises)."""
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def digests_equal(a: Optional[str], b: Optional[str]) -> bool:
    """Constant-time comparison; False if either side is missing or malformed."""
    if not (is_valid_digest(a) and is_valid_digest(b)):
        return False
    return hmac.compare_digest(a.encode("ascii"), b.encode("ascii"))  # type: ignore[union-attr]


def enforce_preview_limits(subject: Optional[str], body: str) -> None:
    """Raise DraftTooLarge when the full content could not be shown (and so must not be approved)."""
    if len(subject or "") > MAX_SUBJECT_CHARS or len(body or "") > MAX_BODY_CHARS:
        raise DraftTooLarge("draft exceeds the preview size limit")
