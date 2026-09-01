"""
Final outbound safety gate: approval-time grounding recheck (spec section 11).

Calls the real endpoint function (api.messages.approve_and_send_draft)
directly against a real, isolated SQLite session -- not a mock of the
grounding check. The FastAPI Depends(...) defaults on org_id/db are just
ordinary parameters when the function is called directly (not through
HTTP), so they're supplied explicitly here; this avoids needing to boot
the full app (and its APScheduler-backed lifespan) just to exercise one
route function.

The core scenario this file exists to prove (previously an admitted gap:
"Approval path skips revalidation" / "the biggest problem"): a draft that
was grounded when generated must be re-validated against *current*
campaign/contact state at approval time, and blocked if that state has
since changed such that the draft is no longer supported.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailer_agent.api.messages import approve_and_send_draft
from mailer_agent.models import Base, Campaign, Contact, Message, MessageDirection, MessageStatus, MessageType


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


def _make_campaign(db_session, proof_points: str) -> Campaign:
    campaign = Campaign(
        name="Test Campaign",
        organization_id="default",
        sender_name="Jordan",
        sender_org="TestCorp",
        sender_email="jordan@testcorp.example.com",
        value_prop="We help teams move faster.",
        proof_points=proof_points,
    )
    db_session.add(campaign)
    db_session.flush()
    return campaign


def _make_contact(db_session, campaign: Campaign) -> Contact:
    contact = Contact(
        campaign_id=campaign.id,
        name="Sam Prospect",
        email="sam@prospect.example.com",
        company="ProspectCo",
        status="active",
    )
    db_session.add(contact)
    db_session.flush()
    return contact


def _make_draft(db_session, contact: Contact, body: str, subject: str = "Quick question") -> Message:
    msg = Message(
        contact_id=contact.id,
        direction=MessageDirection.OUTBOUND.value,
        message_type=MessageType.INITIAL.value,
        subject=subject,
        body=body,
        status=MessageStatus.DRAFT.value,
    )
    db_session.add(msg)
    db_session.commit()
    return msg


def test_grounded_unchanged_draft_is_approved_and_sent(db_session):
    """Baseline: a draft whose claims are still supported by current
    campaign data sends normally through approval."""
    campaign = _make_campaign(db_session, proof_points="We've worked with 50 companies across the region.")
    contact = _make_contact(db_session, campaign)
    msg = _make_draft(db_session, contact, body="We work with 50 companies and see strong results.")

    result = approve_and_send_draft(message_id=msg.id, org_id="default", db=db_session)

    assert result.status == MessageStatus.SENT.value

    db_session.refresh(msg)
    assert msg.status == MessageStatus.SENT.value
    assert msg.message_id_header is not None


def test_never_grounded_claim_is_blocked_at_approval(db_session):
    """A draft with a number that was never in proof_points is blocked --
    even on first approval attempt, not just after a context change."""
    campaign = _make_campaign(db_session, proof_points="We've worked with 50 companies across the region.")
    contact = _make_contact(db_session, campaign)
    msg = _make_draft(db_session, contact, body="We've helped 500 companies achieve amazing results.")

    with pytest.raises(HTTPException) as exc_info:
        approve_and_send_draft(message_id=msg.id, org_id="default", db=db_session)

    assert exc_info.value.status_code == 409

    db_session.refresh(msg)
    assert msg.status == MessageStatus.DRAFT.value, "Blocked draft must not be marked sent"
    assert msg.message_id_header is None, "Blocked draft must not have been sent to SMTP at all"


def test_context_changed_since_draft_generation_blocks_send(db_session):
    """
    The literal scenario from spec section 11:
      1. Draft generated -- supported by proof_points at the time.
      2. Campaign proof_points edited by a human afterward.
      3. Draft (unchanged) now contains a claim unsupported by *current*
         proof_points.
      4. Approval is attempted.
    Expected: send blocked, because the recheck uses current state, not
    whatever was true when the draft was written.
    """
    campaign = _make_campaign(db_session, proof_points="We've worked with 75 companies this year.")
    contact = _make_contact(db_session, campaign)
    # At creation time, "75" is genuinely supported -- this draft would
    # have passed grounding validation when it was generated.
    msg = _make_draft(db_session, contact, body="We've worked with 75 companies just like yours.")

    # A human edits the campaign's proof points after the draft existed --
    # simulating exactly the "context changed" scenario. The old figure is
    # gone; the draft is now unsupported by the current source of truth.
    campaign.proof_points = "We've worked with 12 companies this year."
    db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        approve_and_send_draft(message_id=msg.id, org_id="default", db=db_session)

    assert exc_info.value.status_code == 409
    assert "grounding" in exc_info.value.detail.lower()

    db_session.refresh(msg)
    assert msg.status == MessageStatus.DRAFT.value


def test_suppressed_contact_blocks_send_even_if_grounded(db_session):
    """Suppression check still applies alongside the grounding recheck."""
    from mailer_agent.models import SuppressionEntry

    campaign = _make_campaign(db_session, proof_points="We've worked with 50 companies across the region.")
    contact = _make_contact(db_session, campaign)
    msg = _make_draft(db_session, contact, body="We work with 50 companies and see strong results.")

    db_session.add(SuppressionEntry(email=contact.email, organization_id="default", reason="unsubscribed"))
    db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        approve_and_send_draft(message_id=msg.id, org_id="default", db=db_session)

    assert exc_info.value.status_code == 400
    db_session.refresh(msg)
    assert msg.status == MessageStatus.DRAFT.value


def test_approval_is_org_scoped(db_session):
    """A caller from a different org cannot approve/see this draft (404, not leaked)."""
    campaign = _make_campaign(db_session, proof_points="We've worked with 50 companies across the region.")
    contact = _make_contact(db_session, campaign)
    msg = _make_draft(db_session, contact, body="We work with 50 companies and see strong results.")

    with pytest.raises(HTTPException) as exc_info:
        approve_and_send_draft(message_id=msg.id, org_id="some-other-org", db=db_session)

    assert exc_info.value.status_code == 404

    db_session.refresh(msg)
    assert msg.status == MessageStatus.DRAFT.value


def test_already_sent_message_cannot_be_approved_again(db_session):
    """Re-approving a non-draft message is rejected -- guards against a
    duplicate send via a repeated/racing approval call."""
    campaign = _make_campaign(db_session, proof_points="We've worked with 50 companies across the region.")
    contact = _make_contact(db_session, campaign)
    msg = _make_draft(db_session, contact, body="We work with 50 companies and see strong results.")

    # First approval sends it.
    approve_and_send_draft(message_id=msg.id, org_id="default", db=db_session)
    db_session.refresh(msg)
    assert msg.status == MessageStatus.SENT.value

    # A second approval attempt (e.g. a racing duplicate request) must be rejected.
    with pytest.raises(HTTPException) as exc_info:
        approve_and_send_draft(message_id=msg.id, org_id="default", db=db_session)
    assert exc_info.value.status_code == 400
