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
from datetime import datetime, timezone

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
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship

from mailer_agent.utils.datetime_utils import utcnow


class Base(DeclarativeBase):
    pass


class ContactStatus(str, enum.Enum):
    """
    Relationship state in the sales conversation.
    
    Enhanced to reflect B2B buying stages and conversation state.
    """
    NEW = "new"                        # created, nothing sent yet
    ACTIVE = "active"                  # in sequence, awaiting next action or reply
    REPLIED = "replied"                # last inbound message needs a reply
    ENGAGED = "engaged"                # positive interest shown, active conversation
    QUALIFYING = "qualifying"          # asking questions, gathering info
    EVALUATING = "evaluating"          # comparing solutions, commercial discussion
    MEETING_REQUESTED = "meeting_requested"  # prospect wants to meet
    MEETING_SCHEDULED = "meeting_scheduled"  # meeting confirmed
    NEGOTIATING = "negotiating"        # discussing terms, pricing, contract
    NURTURE = "nurture"                # interested but not ready (future opportunity)
    SEQUENCE_COMPLETE = "sequence_complete"  # ran out of follow-ups, no reply
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"
    SUPPRESSED = "suppressed"          # opted out / bounced -- never message again
    PAUSED = "paused"                  # human paused it manually
    NEEDS_REVIEW = "needs_review"      # requires human attention


class MessageDirection(str, enum.Enum):
    OUTBOUND = "outbound"
    INBOUND = "inbound"


class MessageType(str, enum.Enum):
    INITIAL = "initial_outreach"
    FOLLOW_UP = "follow_up"
    REPLY = "reply"
    CLOSING = "closing"


class MessageStatus(str, enum.Enum):
    """
    Message send/receive state with explicit lifecycle.
    
    Enhanced to track send lifecycle and ambiguous outcomes.

    Verified-in-use vs reserved: DRAFT, SENT, FAILED, UNKNOWN, and
    RECEIVED are actually set by the current send paths (api/messages.py,
    followup/engine_v2.py, mail/reply_handler_v2.py) -- send is
    synchronous within one request/job, so a message goes straight from
    DRAFT to a terminal outcome. APPROVED, PENDING, and SENDING are
    defined but currently unused by any code path: they describe an
    asynchronous approve -> queue -> in-flight pipeline this system
    doesn't currently have (approval triggers a synchronous send in the
    same call). Kept in the enum as forward-reserved states for if that
    changes, not because anything sets them today -- don't assume a
    message will ever be observed in one of these three states via the
    current API.
    """
    DRAFT = "draft"          # generated, not sent (live_sending_enabled=False, or awaiting approval)
    APPROVED = "approved"    # reserved, not currently set -- see class docstring
    PENDING = "pending"      # reserved, not currently set -- see class docstring
    SENDING = "sending"      # reserved, not currently set -- see class docstring
    SENT = "sent"            # successfully sent and confirmed
    FAILED = "failed"        # send failed permanently
    UNKNOWN = "unknown"      # provider may have sent, but we don't know (crash window)
    RECEIVED = "received"    # inbound messages are always "received", never draft/sent


class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    
    # Multi-tenancy: Organization ownership
    # Each campaign belongs to exactly one organization (LeadBoost customer)
    organization_id = Column(String, nullable=True, index=True)  # Nullable for migration compatibility

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
    
    # Timezone for campaign scheduling
    # Used for interpreting "business hours" and prospect local time
    # Format: IANA timezone string (e.g., "America/New_York", "UTC", "Asia/Kolkata")
    timezone = Column(String, default="UTC")

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    contacts = relationship("Contact", back_populates="campaign")


class Contact(Base):
    __tablename__ = "contacts"
    __table_args__ = (
        # Enforce idempotency: same email cannot appear twice in one campaign.
        # This is the DB-level backstop; the application checks first to give
        # a friendly error, but this constraint prevents races.
        UniqueConstraint("campaign_id", "email", name="uq_contacts_campaign_email"),
    )

    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)

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
    
    # Enhanced relationship tracking
    last_reply_at = Column(DateTime, nullable=True)  # When prospect last replied
    last_outbound_at = Column(DateTime, nullable=True)  # When we last sent
    buying_stage = Column(String, nullable=True)  # From semantic analysis
    engagement_score = Column(Float, default=0.0)  # 0-1 score based on interactions

    # Rolling LLM-generated summary of older messages, used to keep
    # long threads' prompts bounded. See memory/store.py.
    #
    # Data-lineage note: memory/store.py's summarization used to include
    # every Message regardless of status (DRAFT/FAILED/UNKNOWN outbound
    # rows, not just genuinely-sent/received ones) when folding older
    # messages into this field -- fixed to use only confirmed
    # conversational evidence (see _conversational_evidence in that
    # module). That fix is forward-looking only: a memory_summary value
    # persisted before the fix may have been generated from a window
    # that included a draft/failed message's content, and this text
    # column has no per-fact provenance to selectively correct. No
    # migration was written for this deliberately -- there's no schema
    # change involved (the column is unchanged), and automatically
    # re-summarizing every existing contact would mean an unreviewed LLM
    # call per contact with its own risk of introducing new errors,
    # which isn't obviously safer than a stale field. If a specific
    # contact's summary is suspected of being contaminated (e.g. it
    # mentions a figure that doesn't appear in any SENT/RECEIVED message
    # for that contact), the safe fix is to clear that one
    # memory_summary value -- it will regenerate correctly (now
    # provenance-filtered) once the thread crosses SUMMARIZE_THRESHOLD
    # again.
    memory_summary = Column(Text, nullable=True)

    # Distributed work claiming (Phase 8 — safe work claiming).
    # When a scheduler worker picks up a contact for outreach/follow-up
    # it writes its worker_id here and sets claimed_at to now().
    # Only the claiming worker should then process this contact.
    # Lease expiry: if claimed_at < now() - CLAIM_LEASE_SECONDS (default 5 min)
    # the record is considered abandoned and can be re-claimed by any worker.
    # These fields are NULL when the contact is not currently being processed.
    claimed_by = Column(String, nullable=True, index=True)    # worker identity string
    claimed_at = Column(DateTime, nullable=True)              # when the lease was taken

    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    campaign = relationship("Campaign", back_populates="contacts")
    messages = relationship(
        "Message", back_populates="contact", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    contact_id = Column(Integer, ForeignKey("contacts.id", ondelete="CASCADE"), nullable=False)

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
    
    # Enhanced semantic analysis (JSON stored)
    semantic_analysis = Column(JSON, nullable=True)  # Full SemanticIntent as JSON
    classification_success = Column(Boolean, nullable=True)  # Did classification succeed?
    classification_failure_reason = Column(String, nullable=True)  # If failed, why?

    error_message = Column(Text, nullable=True)  # populated when status=failed

    created_at = Column(DateTime(timezone=True), default=utcnow)

    contact = relationship("Contact", back_populates="messages")

    __table_args__ = (
        # Message-ID is meant to be globally unique by construction (ours
        # are generated via email.utils.make_msgid(); real inbound ones
        # are unique per email infrastructure convention), so this is a
        # plain global constraint, not organization-scoped.
        #
        # Without this, the dedup check in
        # mail/reply_handler_v2.py::process_inbound_email_v2 ("does a
        # Message with this message_id_header already exist? if not,
        # insert") is a classic check-then-insert race: under true
        # concurrency (the same email arriving via webhook and IMAP
        # nearly simultaneously, or a retried webhook delivery), two
        # transactions can both see "not found" before either commits,
        # and both insert -- producing two logical messages for what
        # should be one. NULL values remain unconstrained (multiple
        # DRAFT/unsent messages with no Message-ID yet are expected and
        # fine) -- only non-NULL collisions are rejected.
        #
        # The application-level check-then-insert is kept (it avoids an
        # exception on the common, non-racing path and gives a cleaner
        # log message), but this constraint is what actually makes
        # duplicate-prevention correct under concurrency: see
        # mail/reply_handler_v2.py's IntegrityError handling around the
        # inbound-message insert, and tests/test_postgresql_concurrency.py.
        UniqueConstraint("message_id_header", name="uq_messages_message_id_header"),
    )


class SuppressionEntry(Base):
    __tablename__ = "suppression_list"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False, index=True)
    # Org-scoped: an unsubscribe in org "acme" should not suppress the same
    # address in org "leadboost" unless they share mailboxes.
    # The unique constraint is therefore on (email, organization_id).
    organization_id = Column(String, nullable=True, index=True)
    reason = Column(String, nullable=True)  # unsubscribed / bounced / manual
    created_at = Column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        UniqueConstraint("email", "organization_id", name="uq_suppression_email_org"),
    )
