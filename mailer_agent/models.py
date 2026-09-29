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
    Index,
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
    __table_args__ = (
        # Phase C (LeadBoost integration): at most one campaign per
        # organization may carry a given integration_source value.
        # integration_source is NULL for every ordinary, human-created
        # campaign, and both PostgreSQL and SQLite treat NULL as
        # distinct-from-itself in a unique index -- so this constraint is
        # inert for all existing/ordinary campaigns and only bites once a
        # second campaign for the same org tries to claim the same
        # non-NULL integration_source (e.g. two concurrent "first use"
        # requests both trying to create the one leadboost campaign for
        # an org -- see api/integrations.py's get-or-create, which relies
        # on this exact constraint as its race backstop, the same
        # "app checks, DB enforces" pattern already used by
        # uq_contacts_campaign_email below). A plain unique INDEX (not a
        # named UNIQUE CONSTRAINT) so the equivalent migration DDL
        # (CREATE UNIQUE INDEX ...) works unchanged on both PostgreSQL and
        # SQLite -- SQLite has no ALTER TABLE ... ADD CONSTRAINT.
        Index(
            "uq_campaigns_org_integration_source",
            "organization_id",
            "integration_source",
            unique=True,
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    
    # Multi-tenancy: Organization ownership
    # Each campaign belongs to exactly one organization (LeadBoost customer)
    organization_id = Column(String, nullable=True, index=True)  # Nullable for migration compatibility

    # Phase C (LeadBoost integration): which external integration this
    # campaign is the fixed, deterministic home for -- e.g. "leadboost".
    # NULL for every ordinary, human-created campaign; never set or read
    # by any pre-existing code path. See uq_campaigns_org_integration_source
    # above and api/integrations.py's get-or-create for how exactly one
    # such campaign per organization is guaranteed.
    integration_source = Column(String, nullable=True, index=True)

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


class ExternalDispatchState(str, enum.Enum):
    """
    Dispatch-operation lifecycle for an externally-authorized outreach
    action (Phase C / LeadBoost integration).

    Deliberately NOT the same state machine as LeadBoost's own
    OutreachAction (PENDING_REVIEW -> APPROVED -> DISPATCHING ->
    SUBMITTED): that lifecycle is LeadBoost's authorization/handoff
    record; this one tracks what *this* service has done with the
    dispatch operation after accepting it. See models.py module docstring
    context and mailer_agent/api/integrations.py.

    QUEUED  -- durably accepted (Transaction A committed), not yet
      claimed by a worker. This is what accepted=true maps to over the
      wire -- see api/integrations.py's response contract.
    SENDING -- claimed by a worker, about to/currently calling
      send_email(). Short-lived. Introduced in a later phase (worker
      claiming); not set anywhere in this batch. See
      resolve_expired_sending_lease() below for the one, corrected rule
      governing what an *expired* SENDING lease recovers to -- read that
      docstring before implementing any lease-recovery code (C6+).
    SENT / FAILED / UNKNOWN -- terminal-ish outcomes mirroring
      mail/sender.py's SendOutcome exactly. Introduced in a later phase;
      not set anywhere in this batch.

    This batch (C2-C4) only ever creates rows in QUEUED. The other four
    values are defined now so the column's full vocabulary is fixed
    before any code writes to it, but nothing in this batch sets them.
    """
    QUEUED = "queued"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    UNKNOWN = "unknown"


def resolve_expired_sending_lease() -> "ExternalDispatchState":
    """
    DESIGN CORRECTION, made explicit before any worker/lease-recovery
    code exists (C6+ -- this function is not called by anything in
    Batch 1; it exists to pin the rule down before that code is written).

    An earlier design draft (the Phase B.2 reconciliation report's
    crash/failure matrix) treated two situations differently:

      (a) worker crashes after Transaction B commits but BEFORE
          send_email() is ever called -- described there as safe to
          requeue and retry, since "SMTP never happened";
      (b) worker crashes DURING or AFTER the SMTP call -- correctly
          described there (and in the governing project brief) as
          UNKNOWN, never auto-resent.

    That distinction is real in principle but is NOT SAFE to act on,
    because this schema records no signal that distinguishes them.
    ExternalDispatch.claimed_at records only when a worker claimed the
    row, not whether that worker had reached send_email() yet. A
    lease-expiry recovery sweep sees the identical row shape --
    state=SENDING, claimed_at older than CLAIM_LEASE_SECONDS -- whether
    the crash happened in case (a), mid-SMTP, or just after SMTP
    accepted but before Transaction C committed. Since the recovery code
    cannot tell these apart after the fact, it must not act as if it
    could -- and must not default to the most optimistic of the
    indistinguishable possibilities.

    CORRECTED RULE, and the only one any future worker/lease-recovery
    implementation may follow: an expired SENDING lease ALWAYS resolves
    to UNKNOWN. It is never automatically moved back to QUEUED and never
    automatically resent. This makes case (a) and case (b) above resolve
    identically, which is the only choice consistent with not being able
    to tell them apart.

    The only way an expired SENDING row could safely become QUEUED again
    is via a *separate*, positively persisted signal that specifically
    proves send_email() was never invoked for that claim -- e.g. a
    durably-recorded "SMTP call started" marker, written in its own
    small transaction distinct from the claim itself, checked before
    SMTP is attempted. No such signal exists in this schema today. This
    function takes no arguments and has no conditional branch because
    that signal does not exist yet; if a later phase deliberately adds
    one, this function's contract -- and this file's tests -- are the
    place to update, not a new ad hoc check inside the worker loop.

    This mirrors mail/sender.py's own AmbiguousSendError philosophy
    (never silently retried) and the "never blind auto-resend" language
    the project brief already applies to the mid-SMTP and
    post-SMTP-pre-Transaction-C cases -- this function removes the one
    case that had drifted from that principle, so all three now agree.
    """
    return ExternalDispatchState.UNKNOWN


class ExternalDispatch(Base):
    """
    One durable record of a single externally-authorized outreach
    dispatch operation (Phase C / LeadBoost integration, and designed to
    be reusable for any future external caller, not LeadBoost-specific
    at the schema level).

    This is the actual idempotency backstop for the integration endpoint
    (mailer_agent/api/integrations.py): UNIQUE(organization_id,
    idempotency_key) is enforced by the database, not only checked in
    application code, so two concurrent identical requests can only ever
    produce one row (the loser's IntegrityError is caught, and the
    winner's row is re-read and returned -- see
    api/integrations.py::create_leadboost_outreach_action).

    request_fingerprint lets a replay of the same idempotency_key be
    distinguished from caller misuse: same key + same fingerprint is a
    safe replay (return the existing operation, no new send); same key +
    different fingerprint is rejected (409, zero mutation) rather than
    silently overwriting what the key already refers to.

    organization_id is populated *only* from the authenticated
    X-API-Key -> org_id resolution (api/deps.py::get_current_org_id),
    never from the request body -- see api/integrations.py. Every query
    against this table must filter on organization_id in the query
    predicate itself (WHERE organization_id = ... AND ...), never
    fetch-then-check, so a key from one org can never read or mutate
    another org's dispatch even if it somehow guesses a valid
    idempotency_key or public_reference.

    external_action_id is the caller's own correlation id (e.g.
    LeadBoost's OutreachAction.id) -- stored for observability only and
    NEVER used to resolve a tenant, campaign, or contact. Confusing
    "correlation data" with "authorization data" is exactly the bug this
    column's docstring exists to prevent (see the inbound webhook's
    to_email-based tenant resolution in api/webhooks.py for what that
    mistake actually looks like in this codebase today -- a separate,
    already-tracked gap this table's design deliberately does not
    repeat).

    public_reference is the opaque identifier actually handed back to
    the caller (as mailing_agent_reference) and used for the
    reconciliation lookup -- never the internal sequential `id`, so the
    external API surface doesn't leak enumerable row counts across the
    tenant boundary.

    claimed_by / claimed_at follow the exact same lease shape as
    Contact's own work-claiming fields (see followup/work_claiming.py) --
    introduced now so the column exists, but not written to by anything
    in this batch; a later phase generalizes work_claiming.py's claim
    logic to this table.
    """

    __tablename__ = "external_dispatches"
    __table_args__ = (
        # THE idempotency guarantee (see class docstring) -- DB-enforced,
        # not an in-memory or best-effort application check.
        UniqueConstraint(
            "organization_id", "idempotency_key",
            name="uq_external_dispatches_org_idempotency_key",
        ),
        # public_reference is handed out externally as the sole
        # reconciliation handle -- it must never collide across
        # organizations either.
        UniqueConstraint(
            "public_reference", name="uq_external_dispatches_public_reference",
        ),
        # Supports the expired-lease sweep a later phase adds (same
        # reasoning as idx_contacts_claimed_by in
        # migrations/002_work_claiming.py).
        Index("ix_external_dispatches_claimed_by", "claimed_by"),
        # Covers the work-claim query a later phase adds: something like
        # WHERE state = 'queued' OR (state = 'sending' AND claimed_at < cutoff).
        Index("ix_external_dispatches_state_claimed_at", "state", "claimed_at"),
    )

    id = Column(Integer, primary_key=True, index=True)

    # Tenancy -- see class docstring. NOT NULL: every dispatch belongs to
    # exactly one authenticated organization from the moment it's created.
    organization_id = Column(String, nullable=False, index=True)

    idempotency_key = Column(String, nullable=False)

    # Correlation only -- see class docstring. Never used for tenant/
    # campaign/contact resolution.
    external_action_id = Column(String, nullable=True)
    correlation_id = Column(String, nullable=True)

    # RESTRICT, not CASCADE (Batch 1.1 correction -- see
    # tests/test_external_dispatch_fk_durability.py). Contact.campaign_id
    # and Message.contact_id above both cascade, and this file does not
    # touch that pre-existing behavior: no code path anywhere in this
    # repository currently deletes a Campaign, Contact, or Message (no
    # DELETE endpoint, no db.delete() call exists today), so changing
    # those established relationships isn't warranted by anything found
    # in this review. But ExternalDispatch is different in kind from an
    # ordinary child row: it is the durable idempotency/reconciliation
    # record itself (see class docstring), and this codebase's schema
    # cannot rule out a parent row being deleted by something outside
    # this application's own request handlers -- a future admin tool, a
    # GDPR/data-deletion process, or direct operator SQL. If that ever
    # happens to a Campaign/Contact/Message that still has an
    # ExternalDispatch pointing at it, CASCADE would silently delete the
    # dispatch record along with it. That specifically breaks the
    # UNIQUE(organization_id, idempotency_key) guarantee this whole
    # design depends on: with the row gone, a retried request bearing
    # the same idempotency_key would find nothing and create a brand new
    # dispatch (and could trigger a genuinely duplicate send), silently
    # violating "no duplicate send for a replayed request" -- and would
    # separately erase reconciliation history for an operation LeadBoost
    # may still be asking about. RESTRICT here makes any such deletion
    # attempt fail loudly (an integrity error) instead of silently
    # discarding dispatch history, for any of the three FKs below,
    # whether the deletion is attempted directly against that row or
    # arrives indirectly via the Contact->Campaign / Message->Contact
    # cascade chain above (a cascading delete is one atomic operation;
    # if any RESTRICT anywhere in that chain would be violated, the
    # whole delete fails, not just the one row this FK is on).
    campaign_id = Column(Integer, ForeignKey("campaigns.id", ondelete="RESTRICT"), nullable=False)
    contact_id = Column(Integer, ForeignKey("contacts.id", ondelete="RESTRICT"), nullable=False)
    # Set in the same final commit that creates this row (together with
    # the ExternalDispatch row itself -- see
    # api/integrations.py::create_leadboost_outreach_action's
    # transaction-boundary docstring) -- never nullable, unlike a
    # "resolved later" design would need.
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="RESTRICT"), nullable=False)

    # SHA-256 hex digest over a canonical JSON serialization of
    # {external_action_id, recipient_email, recipient_name, subject,
    # body} -- see api/integrations.py::_compute_request_fingerprint.
    request_fingerprint = Column(String, nullable=False)

    # Opaque external handle -- see class docstring. uuid4 hex, generated
    # at creation time, never the internal sequential `id`.
    public_reference = Column(String, nullable=False)

    state = Column(String, nullable=False, default=ExternalDispatchState.QUEUED.value, index=True)

    # Not index=True here -- ix_external_dispatches_claimed_by above
    # already covers this column; index=True would create a duplicate.
    claimed_by = Column(String, nullable=True)
    claimed_at = Column(DateTime, nullable=True)

    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    campaign = relationship("Campaign")
    contact = relationship("Contact")
    message = relationship("Message")


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
