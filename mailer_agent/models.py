"""
Data model.

Design notes:
- A Campaign holds *sender identity + offer + cadence* -- the things a
  user configures once per outreach effort. Nothing industry-specific
  is hardcoded here; `value_prop` / `tone` / `follow_up_days` are all
  user-supplied data, not code branches.
- A Contact is one prospect inside a campaign. `follow_up_days` on the
  campaign is a JSON list of integers (e.g. [3, 7, 14]) -- "wait 3 days,
  then 7 more, then 14 more" -- read and applied dynamically by the
  scheduler; nothing about cadence is baked into logic.
- A Message is one email in either direction. The full ordered set of
  Messages for a Contact *is* the conversation memory -- there is no
  separate "memory blob" that can drift from what was actually sent/
  received. For long threads, `Contact.memory_summary` holds an
  LLM-generated rolling summary of everything older than the last few
  messages, so prompts stay bounded without ever discarding the
  underlying record.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class ContactStatus(str, enum.Enum):
    NEW = "new"                        # created, nothing sent yet
    ACTIVE = "active"                  # in sequence, awaiting next action or reply
    REPLIED = "replied"                # last inbound message needs a reply
    SEQUENCE_COMPLETE = "sequence_complete"  # ran out of follow-ups, no reply
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"
    SUPPRESSED = "suppressed"          # opted out / bounced -- never message again
    PAUSED = "paused"                  # human paused it manually


class MessageDirection(str, enum.Enum):
    OUTBOUND = "outbound"
    INBOUND = "inbound"


class MessageType(str, enum.Enum):
    INITIAL = "initial_outreach"
    FOLLOW_UP = "follow_up"
    REPLY = "reply"
    CLOSING = "closing"


class MessageStatus(str, enum.Enum):
    DRAFT = "draft"          # generated, not sent (live_sending_enabled=False, or awaiting approval)
    SENT = "sent"
    FAILED = "failed"
    RECEIVED = "received"    # inbound messages are always "received", never draft/sent


class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)

    # Sender identity -- who this campaign is "from"
    sender_name = Column(String, nullable=False)
    sender_org = Column(String, nullable=False)
    sender_email = Column(String, nullable=False)
    reply_to_email = Column(String, nullable=True)
    sender_title = Column(String, nullable=True)
    sender_signature_extra = Column(Text, nullable=True)  # e.g. phone / calendly link

    # The offer -- what this campaign is trying to sell / get agreement on.
    # This is the single most important field for avoiding generic output:
    # it's real, user-written business content, not a template category.
    value_prop = Column(Text, nullable=False)
    # Optional: specific proof points, case studies, pricing notes the
    # agent is allowed to reference. Kept separate from value_prop so the
    # prompt can clearly mark it as "verified facts you may cite".
    proof_points = Column(Text, nullable=True)
    tone = Column(String, default="professional, direct, concise")

    # Cadence -- dynamic, per campaign, not fixed in code.
    follow_up_days = Column(JSON, default=lambda: [3, 7, 14])
    max_follow_ups = Column(Integer, nullable=True)  # defaults to len(follow_up_days)

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    contacts = relationship("Contact", back_populates="campaign")


class Contact(Base):
    __tablename__ = "contacts"

    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), nullable=False)

    name = Column(String, nullable=True)
    email = Column(String, nullable=False, index=True)
    title = Column(String, nullable=True)
    company = Column(String, nullable=True)

    # Real, structured facts about this contact/company the agent may
    # use -- e.g. "raised Series A in March", "posted 4 open SDR roles".
    # Deliberately freeform text supplied by the caller (or a future
    # enrichment step) rather than a fixed schema, but always presented
    # to the LLM as "verified context" so it never has to invent facts
    # to sound personalized.
    context_notes = Column(Text, nullable=True)

    status = Column(String, default=ContactStatus.NEW.value)
    follow_up_index = Column(Integer, default=0)  # how many follow-ups sent so far
    next_action_at = Column(DateTime, nullable=True)  # when the scheduler should act next

    # Rolling LLM-generated summary of older messages, used to keep
    # long threads' prompts bounded. See memory/store.py.
    memory_summary = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    campaign = relationship("Campaign", back_populates="contacts")
    messages = relationship(
        "Message", back_populates="contact", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    contact_id = Column(Integer, ForeignKey("contacts.id"), nullable=False)

    direction = Column(String, nullable=False)  # MessageDirection
    message_type = Column(String, nullable=True)  # MessageType (null for inbound)
    subject = Column(String, nullable=True)
    body = Column(Text, nullable=False)

    status = Column(String, default=MessageStatus.DRAFT.value)

    # Threading headers -- required to correlate an inbound reply back
    # to the right contact/thread instead of guessing from subject text.
    message_id_header = Column(String, nullable=True, index=True)   # our own Message-ID when sent
    in_reply_to_header = Column(String, nullable=True, index=True)  # for outbound: prior Message-ID
    references_header = Column(Text, nullable=True)

    # Only set for inbound messages, by the reply classifier.
    detected_intent = Column(String, nullable=True)  # interested/objection/question/not_interested/oos/unsubscribe/neutral
    intent_confidence = Column(Float, nullable=True)

    error_message = Column(Text, nullable=True)  # populated when status=failed

    created_at = Column(DateTime, default=datetime.utcnow)

    contact = relationship("Contact", back_populates="messages")


class SuppressionEntry(Base):
    __tablename__ = "suppression_list"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False, index=True, unique=True)
    reason = Column(String, nullable=True)  # unsubscribed / bounced / manual
    created_at = Column(DateTime, default=datetime.utcnow)
