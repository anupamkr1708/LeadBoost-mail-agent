"""
Tenant-safe inbound correlation (spec section 17).

Exercises the real inbound-processing path (mail.reply_handler_v2.
process_inbound_email_v2) against a fully isolated, in-process SQLite
database seeded with two organizations that both have a contact using
the *same* email address -- the scenario where cross-tenant leakage
would actually show up. This is not a code-inspection assertion; it
persists real rows and calls the real function end to end.

Isolation note: this file does NOT import mailer_agent.db (whose engine
is a module-level singleton bound once, at first import, to whatever
DATABASE_URL happened to be set by whichever test file's import ran
first in the session -- a real fragility other test files here work
around with `os.environ.setdefault(...)` before their first
mailer_agent import). Instead it builds its own SQLAlchemy engine/session
directly and passes that Session into process_inbound_email_v2, which
takes a Session as a plain parameter. That sidesteps the shared-singleton
problem entirely rather than depending on import order.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailer_agent.mail.imap_reader import InboundEmail
from mailer_agent.mail.reply_handler_v2 import process_inbound_email_v2
from mailer_agent.models import Base, Campaign, Contact


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


@pytest.fixture(autouse=True)
def _force_deterministic_fallback(monkeypatch):
    """
    This test is about tenant correlation, not classification content --
    force the deterministic (non-LLM) classification path so the test
    doesn't depend on fake_llm fixtures at all.
    """
    import mailer_agent.llm.provider_v2 as provider_module
    monkeypatch.setattr(provider_module, "is_llm_available", lambda: False)


def _seed_two_orgs_with_colliding_contact_email(db_session):
    """
    Two different organizations, each running their own campaign, each
    with a contact using the identical email address -- the exact
    scenario where a naive "look up contact by email" fallback would
    misattribute an inbound reply to the wrong organization.
    """
    campaign_a = Campaign(
        name="Org A Campaign",
        organization_id="org-a",
        sender_name="Alice",
        sender_org="Org A Inc",
        sender_email="alice@org-a.example.com",
        value_prop="Org A's value prop",
    )
    campaign_b = Campaign(
        name="Org B Campaign",
        organization_id="org-b",
        sender_name="Bob",
        sender_org="Org B Inc",
        sender_email="bob@org-b.example.com",
        value_prop="Org B's value prop",
    )
    db_session.add_all([campaign_a, campaign_b])
    db_session.flush()

    # Same prospect email address, contacted by both organizations
    # independently (a realistic collision -- e.g. a shared inbox, or
    # simply the same person being prospected by two different sellers).
    contact_a = Contact(
        campaign_id=campaign_a.id,
        name="Prospect (Org A's contact)",
        email="prospect@shared-prospect.example.com",
        status="active",
    )
    contact_b = Contact(
        campaign_id=campaign_b.id,
        name="Prospect (Org B's contact)",
        email="prospect@shared-prospect.example.com",
        status="active",
    )
    db_session.add_all([contact_a, contact_b])
    db_session.commit()

    return campaign_a, campaign_b, contact_a, contact_b


def test_inbound_reply_with_known_inbox_resolves_to_correct_org(db_session):
    """
    When the webhook/IMAP source tells us which inbox (to_email) received
    the reply, that must be used to resolve the correct organization even
    when another organization has a contact with the identical email
    address.
    """
    campaign_a, campaign_b, contact_a, contact_b = _seed_two_orgs_with_colliding_contact_email(db_session)

    email_in = InboundEmail(
        from_email="prospect@shared-prospect.example.com",
        subject="Re: intro",
        body_text="Thanks, tell me more.",
        message_id=None,
        in_reply_to=None,
        references=[],
        to_email="alice@org-a.example.com",  # received at Org A's inbox
    )

    result = process_inbound_email_v2(db_session, email_in)

    assert result["matched"] is True
    assert result["contact_id"] == contact_a.id, (
        "Reply received at Org A's inbox must attribute to Org A's contact, "
        "never Org B's, even though both contacts share an email address."
    )

    # And the reverse: the same prospect replying to Org B's inbox
    # attributes to Org B's contact.
    email_in_b = InboundEmail(
        from_email="prospect@shared-prospect.example.com",
        subject="Re: intro",
        body_text="Thanks, tell me more.",
        message_id=None,
        in_reply_to=None,
        references=[],
        to_email="bob@org-b.example.com",
    )
    result_b = process_inbound_email_v2(db_session, email_in_b)
    assert result_b["contact_id"] == contact_b.id


def test_inbound_reply_correlates_by_thread_header_regardless_of_email_collision(db_session):
    """
    The threading-header correlation path (In-Reply-To / References
    matching a prior outbound Message-ID) is inherently tenant-safe: a
    Message-ID is unique to one persisted outbound message, which belongs
    to exactly one contact/campaign/org. It should win even without
    to_email, and even with a colliding contact email on another org.
    """
    campaign_a, campaign_b, contact_a, contact_b = _seed_two_orgs_with_colliding_contact_email(db_session)

    from mailer_agent.models import Message, MessageDirection, MessageStatus, MessageType

    prior_outbound = Message(
        contact_id=contact_a.id,
        direction=MessageDirection.OUTBOUND.value,
        message_type=MessageType.INITIAL.value,
        subject="Intro",
        body="Hi there",
        status=MessageStatus.SENT.value,
        message_id_header="<abc123@org-a.example.com>",
    )
    db_session.add(prior_outbound)
    db_session.commit()

    email_in = InboundEmail(
        from_email="prospect@shared-prospect.example.com",
        subject="Re: Intro",
        body_text="Thanks, tell me more.",
        message_id=None,
        in_reply_to="<abc123@org-a.example.com>",
        references=["<abc123@org-a.example.com>"],
        to_email=None,  # deliberately absent -- threading header must still resolve correctly
    )

    result = process_inbound_email_v2(db_session, email_in)

    assert result["matched"] is True
    assert result["contact_id"] == contact_a.id
