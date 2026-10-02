"""
Mailbox credential encryption (M1). The only module that touches Fernet.

Passwords are stored as Fernet tokens; the key comes from
MAILBOX_ENCRYPTION_KEY (deployment configuration). There is no fallback key:
a missing or invalid key raises MailboxEncryptionUnavailable, which the API
maps to 503. Protects against database dumps/backups and direct DB reads; it
does NOT protect against compromise of a host that holds both the database
connection and the key.

Decryption is only available to trusted service code via
get_mailbox_credentials(); it is never exposed over HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cryptography.fernet import Fernet, InvalidToken

from mailer_agent.config import get_settings
from mailer_agent.models import Mailbox


class MailboxEncryptionUnavailable(RuntimeError):
    """MAILBOX_ENCRYPTION_KEY is missing or not a valid Fernet key."""


class MailboxDecryptionError(RuntimeError):
    """A stored token could not be decrypted with the configured key."""


def _fernet() -> Fernet:
    key = get_settings().mailbox_encryption_key.get_secret_value().strip()
    if not key:
        raise MailboxEncryptionUnavailable("MAILBOX_ENCRYPTION_KEY is not configured")
    try:
        return Fernet(key.encode())
    except (ValueError, TypeError):
        raise MailboxEncryptionUnavailable("MAILBOX_ENCRYPTION_KEY is not a valid Fernet key") from None


def encrypt_secret(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_secret(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode()).decode()
    except InvalidToken:
        raise MailboxDecryptionError("stored mailbox credential cannot be decrypted with the configured key") from None


@dataclass(frozen=True)
class MailboxCredentials:
    """Decrypted transport credentials. Passwords are excluded from repr()."""

    email_address: str
    smtp_host: str
    smtp_port: int
    smtp_use_tls: bool
    smtp_username: str
    smtp_password: str = field(repr=False)
    imap_host: str | None = None
    imap_port: int | None = None
    imap_username: str | None = None
    imap_password: str | None = field(default=None, repr=False)


def get_mailbox_credentials(mailbox: Mailbox) -> MailboxCredentials:
    """Decrypt a mailbox's credentials. Internal use only (future M2/M3 code)."""
    imap_password = (
        decrypt_secret(mailbox.imap_password_enc) if mailbox.imap_password_enc else None
    )
    return MailboxCredentials(
        email_address=mailbox.email_address,
        smtp_host=mailbox.smtp_host,
        smtp_port=mailbox.smtp_port,
        smtp_use_tls=mailbox.smtp_use_tls,
        smtp_username=mailbox.smtp_username,
        smtp_password=decrypt_secret(mailbox.smtp_password_enc),
        imap_host=mailbox.imap_host,
        imap_port=mailbox.imap_port,
        imap_username=mailbox.imap_username,
        imap_password=imap_password,
    )
