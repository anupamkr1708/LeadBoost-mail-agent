from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field


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
