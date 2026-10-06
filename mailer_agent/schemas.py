from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    StringConstraints,
    field_validator,
    model_validator,
)

from mailer_agent.models import MailboxStatus


# ---- Campaigns -----------------------------------------------------------

class CampaignCreate(BaseModel):
    name: str
    sender_name: str
    sender_org: str
    sender_email: EmailStr
    reply_to_email: EmailStr | None = None
    sender_title: str | None = None
    sender_signature_extra: str | None = None
    value_prop: str = Field(..., description="What you're offering -- real, specific, written by you.")
    proof_points: str | None = Field(None, description="Facts/case studies the agent is allowed to cite.")
    tone: str = "professional, direct, concise"
    follow_up_days: list[int] = Field(
        default_factory=lambda: [3, 7, 14],
        description="Days to wait before each successive follow-up, e.g. [3,7,14] = 3 days after send, "
        "then 7 more, then 14 more. Fully configurable per campaign.",
    )
    timezone: str = Field(
        default="UTC",
        description=(
            "IANA timezone string for campaign scheduling and business-hours logic, "
            "e.g. 'America/New_York', 'Europe/London', 'Asia/Kolkata'. "
            "Defaults to UTC."
        ),
    )


class CampaignOut(BaseModel):
    id: int
    name: str
    organization_id: str | None
    sender_name: str
    sender_org: str
    sender_email: str
    is_active: bool
    follow_up_days: list[int]
    timezone: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---- Contacts --------------------------------------------------------------

class ContactCreate(BaseModel):
    name: str | None = None
    email: EmailStr
    title: str | None = None
    company: str | None = None
    context_notes: str | None = Field(
        None, description="Real verified facts about this contact/company the agent may reference."
    )


class ContactBulkCreate(BaseModel):
    contacts: list[ContactCreate]


class LeadIngestRequest(BaseModel):
    """
    Accepts leads in whatever shape the caller has them (e.g. LeadBoost's
    own Lead record, or any other CRM/export format) -- see
    mailer_agent/lead_ingestion.py for the field aliases understood.
    Each item is a free-form object; unrecognized keys are preserved
    into context_notes rather than dropped.
    """
    leads: list[dict[str, Any]]


class ContactOut(BaseModel):
    id: int
    campaign_id: int
    name: str | None
    email: str
    title: str | None
    company: str | None
    status: str
    follow_up_index: int
    next_action_at: datetime | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---- Messages ------------------------------------------------------------

class MessageOut(BaseModel):
    id: int
    direction: str
    message_type: str | None
    subject: str | None
    body: str
    status: str
    detected_intent: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ThreadOut(BaseModel):
    contact: ContactOut
    messages: list[MessageOut]


# ---- Manual actions --------------------------------------------------------

class ApproveDraftRequest(BaseModel):
    message_id: int


class SuppressRequest(BaseModel):
    email: EmailStr
    reason: str | None = "manual"


# ---- LeadBoost integration (Phase C) --------------------------------------
# See mailer_agent/api/integrations.py. Deliberately absent vs. any
# older/other integration shape: no organization_id (tenant comes only
# from the authenticated X-API-Key), no sender.* block, no SMTP
# credential -- recipient/message only.

class LeadBoostRecipientIn(BaseModel):
    email: EmailStr
    name: str | None = None


class LeadBoostMessageIn(BaseModel):
    # subject may legitimately be omitted (some transactional-style
    # sends are body-only / reply-in-thread), but body is always
    # required -- there is no draft_message() fallback on this path to
    # fill it in. See mailer_agent/mail/exact_message.py: it stores this
    # exact string as the one Message and defines the ExactSendInput a
    # later phase hands to send_email() unmodified.
    subject: str | None = None
    body: str = Field(..., min_length=1)


class LeadBoostOutreachActionIn(BaseModel):
    # Correlation only -- see ExternalDispatch.external_action_id's
    # docstring in models.py. Never used to resolve tenant, campaign, or
    # contact.
    external_action_id: str | None = None
    idempotency_key: str = Field(..., min_length=1, max_length=255)
    correlation_id: str | None = None
    recipient: LeadBoostRecipientIn
    message: LeadBoostMessageIn


class LeadBoostOutreachActionAccepted(BaseModel):
    """
    accepted=true means "durably accepted by the Mailer Agent for
    asynchronous dispatch" -- it does NOT mean sent, delivered, or even
    SMTP-attempted. See ExternalDispatchState in models.py and
    api/integrations.py's module docstring for the full outcome chain.
    """
    accepted: bool
    mailing_agent_reference: str | None = None


class LeadBoostOutreachActionStatus(BaseModel):
    """
    Read-only reconciliation view of one accepted LeadBoost dispatch
    (GET /integrations/leadboost/outreach-actions/{idempotency_key}).

    accepted is always true here: the row exists only because the Mailer
    Agent durably accepted the operation. It says nothing about delivery.

    state is the durable ExternalDispatchState, passed through verbatim:
      queued   accepted, not yet completed by a worker (for generated
               outreach this includes the message still being generated)
      sending  a worker owns it; SMTP may be executing
      sent     the sender reported success and that was recorded
      failed   the sender reported a definite failure and that was recorded
      unknown  the final outcome cannot be safely determined. This is NOT
               "failed" and NOT "not delivered"; it is never retried
               automatically.

    Deliberately absent: organization/campaign/contact/message/dispatch
    ids, claim fields, Message-ID, and error text (ExternalDispatch.
    error_message holds raw internal diagnostics).
    """
    accepted: bool = True
    state: Literal["queued", "sending", "sent", "failed", "unknown"]
    mailing_agent_reference: str
    updated_at: datetime | None = None


# ---- LeadBoost generated-outreach intake (C9.2) ---------------------------
# POST /integrations/leadboost/outreach-requests. LeadBoost has already
# authorized the action; Mailer generates the message from the context below.
#
# Deliberately narrow, and strict: extra="forbid" on EVERY model so an
# undeclared field is a 422, never silently dropped or honoured. In
# particular there is no organization/tenant, subject, body, sender identity,
# reply-to, SMTP/IMAP setting, credential, mailbox reference, proof_points,
# tone, action type, cadence or campaign setting -- sender identity is the
# deployment-level integration sender (config.leadboost_integration_sender_*)
# until the Mailer-owned Mailbox lands (M1/M2). Do not add fields here
# "because they might be useful"; every one is a new caller-controlled input.
#
# Length limits are this stage's choice (bounded prompt size, no
# prompt-stuffing), not a LeadBoost contract.

_STRICT = ConfigDict(extra="forbid")


class LeadBoostGeneratedRecipientIn(BaseModel):
    model_config = _STRICT

    email: EmailStr
    name: str | None = Field(default=None, min_length=1, max_length=200)
    title: str | None = Field(default=None, min_length=1, max_length=200)
    company: str | None = Field(default=None, min_length=1, max_length=200)


class LeadBoostGenerationContextIn(BaseModel):
    model_config = _STRICT

    # The offer for THIS action. Request-local: used to draft and ground this
    # one message, snapshotted on its ExternalDispatch, never stored on the
    # shared integration Campaign.
    value_proposition: str = Field(..., min_length=1, max_length=2000)
    # Verified facts about the recipient for THIS action (the only facts the
    # drafted message may cite about them). At most 8.
    recipient_facts: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        default_factory=list, max_length=8
    )


class LeadBoostOutreachRequestIn(BaseModel):
    model_config = _STRICT

    # Operation identity. external_action_id is required (unlike the
    # exact-message route) because the idempotency fingerprint is built from
    # it and the recipient: see api/integrations.py::_compute_generated_request_fingerprint.
    external_action_id: str = Field(..., min_length=1, max_length=255)
    idempotency_key: str = Field(..., min_length=1, max_length=255)
    correlation_id: str | None = Field(default=None, max_length=255)
    recipient: LeadBoostGeneratedRecipientIn
    context: LeadBoostGenerationContextIn


# ---- Mailboxes (M1) --------------------------------------------------------
# Strict on every model (extra="forbid"): organization, public_reference and
# the encrypted columns are never caller-settable. Passwords are SecretStr so
# a model repr can't leak them; the output model is an explicit allowlist.

_Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
_Password = Annotated[SecretStr, Field(min_length=1, max_length=1024)]


class MailboxCreate(BaseModel):
    model_config = _STRICT

    email_address: EmailStr
    smtp_host: _Text
    smtp_port: int = Field(..., ge=1, le=65535)
    smtp_use_tls: bool
    smtp_username: _Text
    smtp_password: _Password
    # IMAP is optional but all-or-none.
    imap_host: _Text | None = None
    imap_port: int | None = Field(default=None, ge=1, le=65535)
    imap_username: _Text | None = None
    imap_password: _Password | None = None

    @field_validator("email_address", mode="before")
    @classmethod
    def _strip_email(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @field_validator("email_address", mode="after")
    @classmethod
    def _lower_email(cls, v: str) -> str:
        return v.lower()

    @model_validator(mode="after")
    def _imap_all_or_none(self) -> "MailboxCreate":
        imap = (self.imap_host, self.imap_port, self.imap_username, self.imap_password)
        if any(v is not None for v in imap) and any(v is None for v in imap):
            raise ValueError(
                "imap_host, imap_port, imap_username and imap_password must be provided together or not at all"
            )
        return self


class MailboxUpdate(BaseModel):
    """
    Omitted field = unchanged. Passwords are write-only replacements.

    L1: SMTP transport metadata (host, port, TLS flag, username) is updatable so
    an external control plane whose own connection settings can change after
    provisioning (LeadBoost's EmailAccount) can keep this mailbox in sync. This
    is deliberately NOT identity: email_address, organization_id and
    public_reference are still not accepted here (extra="forbid" -> 422), and
    IMAP metadata stays out until M3 defines inbound ownership.
    """

    model_config = _STRICT

    status: MailboxStatus | None = None
    smtp_password: _Password | None = None
    imap_password: _Password | None = None
    smtp_host: _Text | None = None
    smtp_port: int | None = Field(default=None, ge=1, le=65535)
    smtp_use_tls: bool | None = None
    smtp_username: _Text | None = None

    @model_validator(mode="after")
    def _no_explicit_null(self) -> "MailboxUpdate":
        for name in self.model_fields_set:
            if getattr(self, name) is None:
                raise ValueError(f"{name} must not be null")
        return self


class MailboxOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    public_reference: str
    email_address: str
    smtp_host: str
    smtp_port: int
    smtp_use_tls: bool
    smtp_username: str
    imap_host: str | None = None
    imap_port: int | None = None
    imap_username: str | None = None
    status: MailboxStatus
    created_at: datetime | None = None
    updated_at: datetime | None = None
