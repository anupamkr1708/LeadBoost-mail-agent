"""
Regression tests for two defects found during the real controlled live
E2E (LIVE_SENDING_ENABLED=true, one real Gmail round trip):

1. Reply threading: a contextual reply's Responder-generated subject
   ("Pricing details for Mailer Agent") silently replaced the inbound
   thread's subject ("Re: Can AI-driven follow-ups free up your sales
   team?"), so Gmail displayed the reply as a new conversation even
   though In-Reply-To/References were technically correct at the
   protocol level. Subject for a contextual reply must be derived
   deterministically from the inbound subject, never from the LLM.

2. A pricing/information request with no approved pricing data produced
   an honest "no fabricated number" draft that nonetheless claimed
   unverified operational facts: "I've pulled together the pricing
   tiers..." and "will send you the exact numbers in a follow-up email
   shortly". Neither claim traces to approved campaign/contact data.
   The fix is two-part and deliberately not a keyword branch:
     a) the planner now actually sees approved campaign/contact content
        (previously it judged "could this plausibly be answered from
        approved materials?" with no visibility into what's approved at
        all);
     b) the responder's system prompt explicitly forbids claiming
        prepared-but-unsent information as a category, not just
        specific numbers.

No live LLM calls: the fake provider from conftest returns only what a
test explicitly queues, consumed in call order (planner, then
responder) exactly as mail/reply_handler_v2.py invokes them.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailer_agent.llm.grounding import validate_grounding
from mailer_agent.llm.prompts import AGENT_SYSTEM_PROMPT
from mailer_agent.mail.imap_reader import InboundEmail, as_reply_subject
from mailer_agent.mail.reply_handler_v2 import (
    _build_known_facts_summary,
    _build_references_header,
    _draft_and_maybe_send_reply,
)
from mailer_agent.models import Base, Campaign, Contact, Message, MessageDirection, MessageStatus, MessageType
from mailer_agent.semantic_models import BuyingStage, IntentType, SemanticIntent, SentimentType

INBOUND_SUBJECT = "Re: Can AI-driven follow-ups free up your sales team?"
PARENT_MESSAGE_ID = "<179032808514.84325.761774581440267479@gmail.com>"
GMAIL_REPLY_MESSAGE_ID = "<CAFwHQFdh4njiMF-9=MnfPE-OMZL1cEN1O9DLgYicfJA_1bF_2A@mail.gmail.com>"

FABRICATED_PRICE_BODY = (
    "Hi Sam,\n\nOur Team tier is $499/month with unlimited seats.\n\nBest,\nDeepak"
)
UNSUPPORTED_COMMITMENT_SUBJECT = "Pricing details for Mailer Agent"


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _intent(**overrides) -> SemanticIntent:
    base = dict(
        intents=[IntentType.QUESTION],
        sentiment=SentimentType.POSITIVE,
        buying_stage=BuyingStage.EVALUATING,
        confidence=0.9,
        requires_human_review=False,
        reasoning="Prospect asked a follow-up question.",
    )
    base.update(overrides)
    return SemanticIntent(**base)


def _campaign(db, *, proof_points=None) -> Campaign:
    c = Campaign(
        name="Controlled Live Mail E2E", organization_id="default", sender_name="Deepak",
        sender_org="Mailer Agent Controlled Test", sender_email="deepak@testcorp.example.com",
        value_prop="We help sales teams automate outreach with intelligent AI agents.",
        proof_points=proof_points,
    )
    db.add(c)
    db.flush()
    return c


def _contact(db, campaign, *, context_notes=None) -> Contact:
    c = Contact(
        campaign_id=campaign.id, name="Controlled Integration Recipient",
        email="prospect@example.com", company="ProspectCo",
        context_notes=context_notes, status="active",
    )
    db.add(c)
    db.flush()
    return c


def _email_in(*, references=()) -> InboundEmail:
    return InboundEmail(
        from_email="prospect@example.com",
        subject=INBOUND_SUBJECT,
        body_text="Thanks, this looks interesting. Can you share more details about pricing?",
        message_id=GMAIL_REPLY_MESSAGE_ID,
        in_reply_to=PARENT_MESSAGE_ID,
        references=list(references),
    )


# ---------------------------------------------------------------------------
# 1 & 2. Contextual reply keeps the inbound thread subject and parent
#        In-Reply-To, regardless of what the responder generated.
# ---------------------------------------------------------------------------

def test_contextual_reply_keeps_thread_subject_not_llm_subject(db_session, fake_llm):
    campaign = _campaign(db_session)
    contact = _contact(db_session, campaign)
    email_in = _email_in()

    fake_llm.queue_response({
        "action_type": "provide_requested_information",
        "objective": "Explain what pricing information is available.",
        "reason": "Prospect asked for pricing details.",
        "confidence": 0.9,
        "requires_human_review": False,
    })
    fake_llm.queue_response({
        "subject": UNSUPPORTED_COMMITMENT_SUBJECT,  # the exact live-bug subject
        "body": "Hi Sam,\n\nHappy to help -- what usage scenario did you have in mind?\n\nBest,\nDeepak",
        "reasoning": "test",
    })

    result: dict = {}
    intent = _intent(has_pricing_question=True, requested_information=["pricing"])
    _draft_and_maybe_send_reply(db_session, contact, campaign, email_in, intent, result)
    db_session.flush()

    reply_msg = db_session.query(Message).filter_by(
        contact_id=contact.id, in_reply_to_header=email_in.message_id
    ).one()

    assert reply_msg.subject == as_reply_subject(INBOUND_SUBJECT)
    assert reply_msg.subject != UNSUPPORTED_COMMITMENT_SUBJECT
    # The new outbound reply chains onto the prospect's inbound message
    # itself (its own Message-ID), not onto that message's own parent --
    # each hop's immediate parent is the message being replied to.
    assert reply_msg.in_reply_to_header == GMAIL_REPLY_MESSAGE_ID


def test_contextual_reply_keeps_thread_subject_when_auto_sent(db_session, fake_llm, monkeypatch):
    """Same guarantee on the auto-send path, which uses a separately
    computed subject variable at send time (mail/sender.py's
    send_email call), not just the persisted draft row."""
    from mailer_agent.mail import reply_handler_v2 as rh
    monkeypatch.setattr(rh.settings, "auto_reply_enabled", True)

    campaign = _campaign(db_session)
    contact = _contact(db_session, campaign)
    email_in = _email_in()

    fake_llm.queue_response({
        "action_type": "answer",
        "objective": "Answer their question directly.",
        "reason": "Simple factual question, safe to auto-send.",
        "confidence": 0.95,
        "requires_human_review": False,
    })
    fake_llm.queue_response({
        "subject": UNSUPPORTED_COMMITMENT_SUBJECT,
        "body": "Hi Sam,\n\nHappy to help -- what usage scenario did you have in mind?\n\nBest,\nDeepak",
        "reasoning": "test",
    })

    result: dict = {}
    # has_pricing_question=False here: guardrails hard-require review for
    # pricing questions regardless of subject, which would mask this
    # assertion behind "reply_drafted_awaiting_approval" instead of
    # exercising the auto-send subject path this test targets.
    intent = _intent(has_pricing_question=False, confidence=0.95)
    _draft_and_maybe_send_reply(db_session, contact, campaign, email_in, intent, result)
    db_session.flush()

    reply_msg = db_session.query(Message).filter_by(
        contact_id=contact.id, in_reply_to_header=email_in.message_id
    ).one()

    assert result["action"] == "auto_replied"
    assert reply_msg.status == MessageStatus.SENT.value
    assert reply_msg.subject == as_reply_subject(INBOUND_SUBJECT)
    assert reply_msg.subject != UNSUPPORTED_COMMITMENT_SUBJECT


# ---------------------------------------------------------------------------
# 3. References: parent's References chain + parent's own Message-ID,
#    normalized, deduped, persisted on the Message row.
# ---------------------------------------------------------------------------

def test_build_references_header_chains_and_dedupes():
    # References + In-Reply-To + own Message-ID, in order.
    assert (
        _build_references_header(["<root@x>", "<mid1@x>"], "<mid1@x>", "<mid2@x>")
        == "<root@x> <mid1@x> <mid2@x>"
    )
    # Real-world gap this fixes: a client (Gmail, on a thread's first
    # reply) sets In-Reply-To but leaves References empty -- the
    # immediate ancestor must still end up in the chain, not just the
    # bare parent id.
    assert (
        _build_references_header([], "<parent@x>", "<child@x>")
        == "<parent@x> <child@x>"
    )
    # Parent's own id already present in its References (some clients do
    # this) must not be duplicated.
    assert (
        _build_references_header(["<root@x>", "<mid2@x>"], "<mid2@x>", "<mid2@x>")
        == "<root@x> <mid2@x>"
    )
    # No ancestor chain, no in-reply-to -- just the immediate parent.
    assert _build_references_header([], None, "<mid2@x>") == "<mid2@x>"
    # Nothing at all.
    assert _build_references_header([], None, None) is None


def test_contextual_reply_recovers_in_reply_to_when_references_is_empty(db_session, fake_llm):
    """
    Reproduces the exact real live-E2E case: Gmail's reply had
    References EMPTY but In-Reply-To set to the original outreach's
    Message-ID. Building the outbound chain from References alone would
    silently drop that ancestor -- In-Reply-To must be folded in too.
    """
    campaign = _campaign(db_session)
    contact = _contact(db_session, campaign)
    # _email_in() default: references=() (empty, matching the live DB
    # evidence), in_reply_to=PARENT_MESSAGE_ID.
    email_in = _email_in()
    assert email_in.references == []
    assert email_in.in_reply_to == PARENT_MESSAGE_ID

    fake_llm.queue_response({
        "action_type": "provide_requested_information",
        "objective": "Explain what pricing information is available.",
        "reason": "Prospect asked for pricing details.",
        "confidence": 0.9,
        "requires_human_review": False,
    })
    fake_llm.queue_response({
        "subject": "irrelevant",
        "body": "Hi Sam,\n\nHappy to help -- what usage scenario did you have in mind?\n\nBest,\nDeepak",
        "reasoning": "test",
    })

    result: dict = {}
    intent = _intent(has_pricing_question=True, requested_information=["pricing"])
    _draft_and_maybe_send_reply(db_session, contact, campaign, email_in, intent, result)
    db_session.flush()

    reply_msg = db_session.query(Message).filter_by(
        contact_id=contact.id, in_reply_to_header=email_in.message_id
    ).one()

    # Without folding in_reply_to in, this would be just
    # GMAIL_REPLY_MESSAGE_ID -- the original outreach's id would be lost.
    assert reply_msg.references_header == f"{PARENT_MESSAGE_ID} {GMAIL_REPLY_MESSAGE_ID}"
    assert reply_msg.in_reply_to_header == GMAIL_REPLY_MESSAGE_ID


def test_contextual_reply_persists_full_references_chain(db_session, fake_llm):
    campaign = _campaign(db_session)
    contact = _contact(db_session, campaign)
    # The prospect's inbound reply (email_in) carries its own References
    # chain from the mail client -- here, the original outreach message
    # that started the thread.
    email_in = _email_in(references=[PARENT_MESSAGE_ID])

    fake_llm.queue_response({
        "action_type": "provide_requested_information",
        "objective": "Explain what pricing information is available.",
        "reason": "Prospect asked for pricing details.",
        "confidence": 0.9,
        "requires_human_review": False,
    })
    fake_llm.queue_response({
        "subject": "irrelevant",
        "body": "Hi Sam,\n\nHappy to help -- what usage scenario did you have in mind?\n\nBest,\nDeepak",
        "reasoning": "test",
    })

    result: dict = {}
    intent = _intent(has_pricing_question=True, requested_information=["pricing"])
    _draft_and_maybe_send_reply(db_session, contact, campaign, email_in, intent, result)
    db_session.flush()

    reply_msg = db_session.query(Message).filter_by(
        contact_id=contact.id, in_reply_to_header=email_in.message_id
    ).one()

    # References = the inbound message's own ancestor chain (the
    # original outreach) + the inbound message's own Message-ID (the
    # immediate parent of this new reply) -- the full chain is preserved
    # for this hop, not collapsed down to just the immediate parent.
    assert reply_msg.references_header == f"{PARENT_MESSAGE_ID} {GMAIL_REPLY_MESSAGE_ID}"
    assert reply_msg.in_reply_to_header == GMAIL_REPLY_MESSAGE_ID


# ---------------------------------------------------------------------------
# 4. A fabricated specific price still gets hard-blocked (existing
#    grounding gate, unchanged -- this is a non-regression check).
# ---------------------------------------------------------------------------

def test_fabricated_price_still_hard_blocked(db_session, fake_llm):
    campaign = _campaign(db_session)  # no proof_points -- no approved pricing
    contact = _contact(db_session, campaign)
    email_in = _email_in()

    fake_llm.queue_response({
        "action_type": "provide_requested_information",
        "objective": "Share pricing.",
        "reason": "Prospect asked for pricing details.",
        "confidence": 0.9,
        "requires_human_review": False,
    })
    fake_llm.queue_response({
        "subject": "irrelevant", "body": FABRICATED_PRICE_BODY, "reasoning": "test",
    })

    result: dict = {}
    intent = _intent(has_pricing_question=True, requested_information=["pricing"])
    _draft_and_maybe_send_reply(db_session, contact, campaign, email_in, intent, result)
    db_session.flush()

    reply_msg = db_session.query(Message).filter_by(
        contact_id=contact.id, in_reply_to_header=email_in.message_id
    ).one()
    assert reply_msg.status == MessageStatus.DRAFT.value
    assert "Grounding" in result["approval_reason"]


# ---------------------------------------------------------------------------
# 5. Missing-pricing scenario: the two prompt/contract-level changes that
#    reduce (not just catch after the fact) the "I've pulled together...
#    / will send shortly" fabrication.
#
# Limitation (same as the earlier grounding-hold regression suite): this
# cannot be a property of the fake LLM's output, since the fake only
# echoes what a test queues -- it can't prove a real model won't say
# this again. What IS deterministically testable, and is exactly what
# changed, is that (a) the responder's system prompt now explicitly
# forbids this category of claim, and (b) the planner's prompt now
# actually contains the approved campaign/contact content it needs to
# judge "is this plausibly answerable from approved materials?" instead
# of guessing blind. The real live E2E is the right place to observe
# whether generation behavior actually improved.
# ---------------------------------------------------------------------------

def test_responder_prompt_forbids_prepared_but_unsent_claims():
    lowered = AGENT_SYSTEM_PROMPT.lower()
    assert "pulled it together" in lowered or "already put it together" in lowered
    assert "shortly" in lowered
    assert "confirm" in lowered


def test_planner_known_facts_includes_approved_campaign_content_no_proof_points(db_session):
    campaign = _campaign(db_session, proof_points=None)
    contact = _contact(db_session, campaign, context_notes=None)
    intent = _intent(has_pricing_question=True)

    summary = _build_known_facts_summary(intent, contact, campaign)

    assert campaign.value_prop in summary
    assert "none approved" in summary.lower()


def test_planner_known_facts_includes_approved_campaign_content_with_proof_points(db_session):
    proof_points = "Starter tier: $99/month. Team tier: $299/month."
    campaign = _campaign(db_session, proof_points=proof_points)
    contact = _contact(db_session, campaign, context_notes="Evaluating for a 12-person team.")
    intent = _intent(has_pricing_question=True)

    summary = _build_known_facts_summary(intent, contact, campaign)

    assert proof_points in summary
    assert "12-person team" in summary


# ---------------------------------------------------------------------------
# 6. Approved pricing data IS present -> grounding continues to allow an
#    answer that uses it (the fix must not overcorrect into refusing
#    every pricing question).
# ---------------------------------------------------------------------------

def test_grounding_allows_answer_when_pricing_is_actually_approved(db_session):
    campaign = _campaign(db_session, proof_points="Starter tier: $99/month. Team tier: $299/month.")
    contact = _contact(db_session, campaign)

    draft_body = (
        "Hi Sam,\n\nHappy to share: our Starter tier is $99/month and the Team "
        "tier is $299/month. Let me know if either fits your team's size.\n\nBest,\nDeepak"
    )
    grounding = validate_grounding(
        draft_body,
        proof_points=campaign.proof_points,
        context_notes=contact.context_notes,
        conversation_transcript="",
        value_prop=campaign.value_prop,
    )

    assert grounding.hard_block is False


# ---------------------------------------------------------------------------
# 7. Approval propagates the persisted References chain into the actual
#    send_email() call, not just onto the draft row (api/messages.py).
# ---------------------------------------------------------------------------

def test_approval_passes_references_header_to_send_email(db_session, monkeypatch):
    import mailer_agent.api.messages as messages_api

    campaign = _campaign(db_session)
    contact = _contact(db_session, campaign)
    references = f"{PARENT_MESSAGE_ID} {GMAIL_REPLY_MESSAGE_ID}"
    draft = Message(
        contact_id=contact.id,
        direction=MessageDirection.OUTBOUND.value,
        message_type=MessageType.REPLY.value,
        subject=as_reply_subject(INBOUND_SUBJECT),
        body="Hi Sam,\n\nHappy to help -- what usage scenario did you have in mind?\n\nBest,\nDeepak",
        status=MessageStatus.DRAFT.value,
        in_reply_to_header=GMAIL_REPLY_MESSAGE_ID,
        references_header=references,
    )
    db_session.add(draft)
    db_session.flush()

    captured = {}

    def fake_send_email(**kwargs):
        captured.update(kwargs)
        from mailer_agent.mail.sender import SendOutcome, SendResult

        return SendResult(
            success=True, message_id="<new-reply@testcorp.example.com>",
            outcome=SendOutcome.SENT, error=None,
        )

    monkeypatch.setattr(messages_api, "send_email", fake_send_email)

    messages_api.approve_and_send_draft(draft.id, org_id="default", db=db_session)

    # This is the exact one-line plumbing bug that can silently
    # reappear: the value is persisted correctly on the draft row (see
    # the tests above) but never reaches the real SMTP call.
    assert captured.get("references_header") == references
    assert captured.get("in_reply_to_header") == GMAIL_REPLY_MESSAGE_ID
