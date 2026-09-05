"""
Regression coverage for the conversation-evidence provenance fix.

Root cause this file guards against (found via direct diagnostic, not
speculation): build_conversation_context() and
maybe_summarize_older_messages() used to include EVERY Message row
regardless of status, and unconditionally labeled every outbound row
"US (sent)" -- including DRAFT and FAILED/UNKNOWN-outcome ones that
never reached the prospect. That meant a draft's own unsupported claim
could appear as its own "conversation evidence" the next time grounding
checked that draft (or one like it) against "the conversation so far",
making the check circular: the draft would appear to support itself.

Fixed with a single shared filter, _conversational_evidence() in
memory/store.py, applied at both call sites. This file tests that
filter directly, plus its effect on the end-to-end grounding path.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailer_agent.llm.grounding import validate_grounding
from mailer_agent.memory.store import build_conversation_context, maybe_summarize_older_messages
from mailer_agent.models import (
    Base,
    Campaign,
    Contact,
    Message,
    MessageDirection,
    MessageStatus,
    MessageType,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _make_campaign_and_contact(db_session, proof_points="We've worked with 50 companies across the region."):
    campaign = Campaign(
        name="Provenance Test", organization_id="default",
        sender_name="Jordan", sender_org="TestCorp", sender_email="jordan@testcorp.example.com",
        value_prop="We help teams move faster.", proof_points=proof_points,
    )
    db_session.add(campaign)
    db_session.flush()
    contact = Contact(campaign_id=campaign.id, name="Sam", email="sam@prospect.example.com", status="active")
    db_session.add(contact)
    db_session.flush()
    return campaign, contact


def _msg(contact, *, direction, status, body, subject="Test", message_type=MessageType.INITIAL.value):
    return Message(
        contact_id=contact.id, direction=direction, message_type=message_type,
        subject=subject, body=body, status=status,
    )


# ---------------------------------------------------------------------------
# 1. DRAFT absent from conversation context
# ---------------------------------------------------------------------------

def test_draft_absent_from_conversation_context(db_session):
    campaign, contact = _make_campaign_and_contact(db_session)
    db_session.add(_msg(contact, direction=MessageDirection.OUTBOUND.value, status=MessageStatus.DRAFT.value,
                         body="We've helped 500 companies achieve amazing results."))
    db_session.commit()

    transcript = build_conversation_context(db_session, contact)

    assert transcript == "(no messages sent yet -- this is the first contact)"
    assert "500" not in transcript


# ---------------------------------------------------------------------------
# 2. A DRAFT cannot self-ground through conversation context
# ---------------------------------------------------------------------------

def test_draft_cannot_self_ground_through_conversation_context(db_session):
    """
    Direct test of the circular-evidence mechanism itself: a draft with
    an unsupported claim, checked against a transcript built from a
    conversation that (before the fix) would have included that same
    draft. Proves the draft's own text is never fed back into its own
    grounding check.
    """
    campaign, contact = _make_campaign_and_contact(db_session, proof_points="We've worked with 50 companies.")
    draft = _msg(contact, direction=MessageDirection.OUTBOUND.value, status=MessageStatus.DRAFT.value,
                 body="We've helped 500 companies achieve amazing results.")
    db_session.add(draft)
    db_session.commit()

    transcript = build_conversation_context(db_session, contact)
    result = validate_grounding(
        draft.body,
        proof_points=campaign.proof_points,
        context_notes=contact.context_notes,
        conversation_transcript=transcript,
        value_prop=campaign.value_prop,
    )

    assert not result.is_safe_to_send, (
        "A draft's own unsupported claim must not become 'grounded' by "
        "virtue of appearing in its own conversation-context evidence."
    )


# ---------------------------------------------------------------------------
# 3. A draft grounded at creation becomes ungrounded after proof_points change
# ---------------------------------------------------------------------------

def test_draft_grounded_at_creation_becomes_ungrounded_after_context_change(db_session):
    campaign, contact = _make_campaign_and_contact(db_session, proof_points="We've worked with 75 companies this year.")
    draft = _msg(contact, direction=MessageDirection.OUTBOUND.value, status=MessageStatus.DRAFT.value,
                 body="We've worked with 75 companies just like yours.")
    db_session.add(draft)
    db_session.commit()

    # At creation time, grounded: "75" is genuinely supported.
    transcript = build_conversation_context(db_session, contact)
    result_before = validate_grounding(
        draft.body, proof_points=campaign.proof_points, context_notes=contact.context_notes,
        conversation_transcript=transcript, value_prop=campaign.value_prop,
    )
    assert result_before.is_safe_to_send

    # Campaign proof_points edited afterward -- the old figure is gone.
    campaign.proof_points = "We've worked with 12 companies this year."
    db_session.commit()

    transcript_after = build_conversation_context(db_session, contact)
    result_after = validate_grounding(
        draft.body, proof_points=campaign.proof_points, context_notes=contact.context_notes,
        conversation_transcript=transcript_after, value_prop=campaign.value_prop,
    )
    assert not result_after.is_safe_to_send, (
        "The recheck must use current proof_points, not whatever was true "
        "when the draft was written -- and must not let the draft's own "
        "unchanged body re-ground itself via conversation evidence."
    )


# ---------------------------------------------------------------------------
# 4. A stale/old draft cannot enter the rolling summary
# ---------------------------------------------------------------------------

def test_stale_draft_never_enters_rolling_summary(db_session, monkeypatch):
    """
    Enough messages to push old ones out of the verbatim window and into
    summarization range -- one of the old ones is a DRAFT with an
    unsupported claim. It must never reach the LLM summarization prompt.
    """
    campaign, contact = _make_campaign_and_contact(db_session)

    captured_prompts = []

    def fake_call_llm_text(system_prompt, human_prompt, **kwargs):
        captured_prompts.append(human_prompt)
        return "Summary: prospect asked about pricing and features."

    import mailer_agent.memory.store as store_module
    monkeypatch.setattr(store_module, "call_llm_text", fake_call_llm_text)

    # 10 messages total: enough to exceed SUMMARIZE_THRESHOLD (8) and
    # push the earliest ones out of the RECENT_MESSAGES_VERBATIM (6)
    # window into summarization range. Message #2 (old, would be
    # summarized) is a DRAFT with an unsupported claim.
    db_session.add(_msg(contact, direction=MessageDirection.OUTBOUND.value, status=MessageStatus.SENT.value, body="Initial outreach."))
    db_session.add(_msg(contact, direction=MessageDirection.OUTBOUND.value, status=MessageStatus.DRAFT.value,
                         body="We've helped 9999 companies -- an unsupported, never-sent claim."))
    for i in range(8):
        direction = MessageDirection.INBOUND.value if i % 2 == 0 else MessageDirection.OUTBOUND.value
        status = MessageStatus.RECEIVED.value if direction == MessageDirection.INBOUND.value else MessageStatus.SENT.value
        db_session.add(_msg(contact, direction=direction, status=status, body=f"Message number {i}."))
    db_session.commit()

    maybe_summarize_older_messages(db_session, contact)

    assert captured_prompts, "Summarization should have run given 10 messages"
    for prompt in captured_prompts:
        assert "9999" not in prompt, (
            "A DRAFT message's unsupported claim must never be folded into "
            "the rolling summary, even once it's old enough to leave the "
            "verbatim window."
        )


# ---------------------------------------------------------------------------
# 5-6. SENT outbound and INBOUND messages remain valid evidence
# ---------------------------------------------------------------------------

def test_sent_outbound_and_inbound_remain_valid_evidence(db_session):
    campaign, contact = _make_campaign_and_contact(db_session)
    db_session.add(_msg(contact, direction=MessageDirection.OUTBOUND.value, status=MessageStatus.SENT.value,
                         body="We've worked with 50 companies across the region."))
    db_session.add(_msg(contact, direction=MessageDirection.INBOUND.value, status=MessageStatus.RECEIVED.value,
                         body="Thanks, tell me more about the 50 companies you mentioned."))
    db_session.commit()

    transcript = build_conversation_context(db_session, contact)

    assert "US (sent)" in transcript
    assert "THEM (received)" in transcript
    assert "50 companies" in transcript  # from the sent message
    assert "tell me more" in transcript  # from the inbound message


def test_failed_and_unknown_outcome_messages_excluded(db_session):
    """FAILED and UNKNOWN (ambiguous SMTP outcome) outbound messages are
    excluded the same as DRAFT -- neither is confirmed to have reached
    the prospect."""
    campaign, contact = _make_campaign_and_contact(db_session)
    db_session.add(_msg(contact, direction=MessageDirection.OUTBOUND.value, status=MessageStatus.FAILED.value,
                         body="This send definitely failed."))
    db_session.add(_msg(contact, direction=MessageDirection.OUTBOUND.value, status=MessageStatus.UNKNOWN.value,
                         body="This send outcome is ambiguous -- might have gone out, might not have."))
    db_session.commit()

    transcript = build_conversation_context(db_session, contact)

    assert transcript == "(no messages sent yet -- this is the first contact)"
    assert "definitely failed" not in transcript
    assert "ambiguous" not in transcript
