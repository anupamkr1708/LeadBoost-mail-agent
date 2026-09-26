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
def live_sending(monkeypatch):
    """
    Opt a test into the approval endpoint's live-sending gate (see
    api/messages.py's boundary check) while keeping actual dispatch
    fully fake. With live_sending_enabled=True, send_email() itself
    would attempt a REAL SMTP connection (mail/sender.py's dry-run
    branch only applies when the flag is False) -- so send_email is
    replaced here with the same kind of simulation that branch does: a
    real, correctly-shaped SendResult, no socket touched. This mirrors
    what tests/conftest.py's own force_test_dry_run fixture and
    test_smtp_local_integration.py already do for this one shared flag.
    """
    import mailer_agent.api.messages as messages_api
    from email.utils import make_msgid

    from mailer_agent.mail.sender import SendOutcome, SendResult

    monkeypatch.setattr(messages_api.settings, "live_sending_enabled", True)

    def fake_send_email(*, to_email, from_email, from_name, subject, body_text,
                         reply_to=None, in_reply_to_header=None, references_header=None):
        domain = from_email.split("@")[-1] if "@" in from_email else "localhost"
        return SendResult(success=True, message_id=make_msgid(domain=domain), outcome=SendOutcome.SENT)

    monkeypatch.setattr(messages_api, "send_email", fake_send_email)


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


def test_grounded_unchanged_draft_is_approved_and_sent(db_session, live_sending):
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


def test_review_required_pricing_mention_can_be_approved_and_sent(db_session, live_sending):
    """
    Spec sections 24-26: a draft that mentions pricing but invents no
    specific figure is 'review_required', not 'hard_block'. Approval IS
    the human review step, so this must be sendable through the normal
    approve endpoint -- unlike a genuinely fabricated numeric claim.
    """
    campaign = _make_campaign(db_session, proof_points="We help teams move faster.")
    contact = _make_contact(db_session, campaign)
    msg = _make_draft(
        db_session, contact,
        body="Thanks for your interest! I'll get pricing details together "
             "for your team and follow up shortly.",
    )

    result = approve_and_send_draft(message_id=msg.id, org_id="default", db=db_session)

    assert result.status == MessageStatus.SENT.value


def test_fabricated_price_still_hard_blocked_at_approval(db_session):
    """The literal spec section 25 example: approval must NOT be able to
    turn a fabricated, never-approved price into a sendable message."""
    campaign = _make_campaign(db_session, proof_points="We help teams move faster.")
    contact = _make_contact(db_session, campaign)
    msg = _make_draft(db_session, contact, body="Our Enterprise plan is $499/month.")

    with pytest.raises(HTTPException) as exc_info:
        approve_and_send_draft(message_id=msg.id, org_id="default", db=db_session)

    assert exc_info.value.status_code == 409
    db_session.refresh(msg)
    assert msg.status == MessageStatus.DRAFT.value, "Fabricated price must never be sent"


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


def test_already_sent_message_cannot_be_approved_again(db_session, live_sending):
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


def test_approval_rejected_when_live_sending_disabled(db_session, monkeypatch):
    """
    Regression test for a real live-run gap: with LIVE_SENDING_ENABLED
    unset/false, the server itself logs DRY RUN MODE and send_email()
    logs "[DRY RUN] Would send to ..." -- but the approval endpoint used
    to still persist status=sent, because it only ever read
    send_email()'s (deliberately faked) success=True/SendOutcome.SENT
    result, never the live-sending flag itself. That produced a
    database that claimed a real send happened when no SMTP
    transmission was ever attempted. Approval must now reject outright
    (draft stays exactly as it was) rather than silently recording a
    send that didn't happen.
    """
    import mailer_agent.api.messages as messages_api

    campaign = _make_campaign(db_session, proof_points="We've worked with 50 companies across the region.")
    contact = _make_contact(db_session, campaign)
    msg = _make_draft(db_session, contact, body="We work with 50 companies and see strong results.")

    # If the endpoint reached send_email at all despite live sending
    # being disabled, this fails the test loudly instead of silently
    # attempting a real SMTP connection.
    def _must_not_be_called(**kwargs):
        raise AssertionError("send_email must not be called when live sending is disabled")

    monkeypatch.setattr(messages_api, "send_email", _must_not_be_called)
    assert messages_api.settings.live_sending_enabled is False

    with pytest.raises(HTTPException) as exc_info:
        approve_and_send_draft(message_id=msg.id, org_id="default", db=db_session)

    assert exc_info.value.status_code == 409
    assert "live sending" in exc_info.value.detail.lower()

    db_session.refresh(msg)
    assert msg.status == MessageStatus.DRAFT.value, "Must not be silently marked sent"
    assert msg.message_id_header is None, "No Message-ID should be minted for a rejected approval"


def test_approval_rejected_then_succeeds_once_live_sending_enabled(db_session, monkeypatch):
    """The rejection above is a hard gate, not a permanently broken
    state -- the same draft can be approved normally once live sending
    is actually enabled."""
    import mailer_agent.api.messages as messages_api
    from email.utils import make_msgid

    from mailer_agent.mail.sender import SendOutcome, SendResult

    campaign = _make_campaign(db_session, proof_points="We've worked with 50 companies across the region.")
    contact = _make_contact(db_session, campaign)
    msg = _make_draft(db_session, contact, body="We work with 50 companies and see strong results.")

    with pytest.raises(HTTPException) as exc_info:
        approve_and_send_draft(message_id=msg.id, org_id="default", db=db_session)
    assert exc_info.value.status_code == 409
    db_session.refresh(msg)
    assert msg.status == MessageStatus.DRAFT.value

    def fake_send_email(*, to_email, from_email, from_name, subject, body_text,
                         reply_to=None, in_reply_to_header=None, references_header=None):
        domain = from_email.split("@")[-1] if "@" in from_email else "localhost"
        return SendResult(success=True, message_id=make_msgid(domain=domain), outcome=SendOutcome.SENT)

    monkeypatch.setattr(messages_api.settings, "live_sending_enabled", True)
    monkeypatch.setattr(messages_api, "send_email", fake_send_email)

    result = approve_and_send_draft(message_id=msg.id, org_id="default", db=db_session)
    assert result.status == MessageStatus.SENT.value
    db_session.refresh(msg)
    assert msg.status == MessageStatus.SENT.value
    assert msg.message_id_header is not None
