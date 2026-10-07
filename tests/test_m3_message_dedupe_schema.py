"""
M3-C schema: inbound dedupe identity is (mailbox_id, RFC Message-ID); rows with
no mailbox keep the original global Message-ID uniqueness.

Regression for the cross-tenant drop: before M3, uq_messages_message_id_header
was global, so the same RFC Message-ID arriving in two organizations' mailboxes
(shared CC, mailing list) made the second org's insert a "duplicate".
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from mailer_agent.models import (
    Base,
    Campaign,
    Contact,
    Mailbox,
    Message,
    MessageDirection,
    MessageStatus,
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _mailbox(db, org, email):
    mb = Mailbox(
        organization_id=org, email_address=email, smtp_host="s", smtp_port=25,
        smtp_use_tls=False, smtp_username=email, smtp_password_enc="x",
    )
    db.add(mb)
    db.flush()
    return mb


def _contact(db, org, email="p@example.com"):
    c = Campaign(
        name="c", organization_id=org, sender_name="n", sender_org="o",
        sender_email=f"s@{org}.example", value_prop="v",
    )
    db.add(c)
    db.flush()
    ct = Contact(campaign_id=c.id, email=email)
    db.add(ct)
    db.flush()
    return ct


def _msg(contact, header, mailbox_id=None, direction=MessageDirection.INBOUND.value):
    return Message(
        contact_id=contact.id, direction=direction, body="b",
        status=MessageStatus.RECEIVED.value, message_id_header=header, mailbox_id=mailbox_id,
    )


def test_same_message_id_in_two_mailboxes_is_two_rows(db):
    a, b = _mailbox(db, "org-a", "a@x.example"), _mailbox(db, "org-b", "b@x.example")
    ca, cb = _contact(db, "org-a"), _contact(db, "org-b")
    db.add_all([_msg(ca, "<same@list>", a.id), _msg(cb, "<same@list>", b.id)])
    db.flush()  # must not raise


def test_same_message_id_twice_in_one_mailbox_is_rejected(db):
    a = _mailbox(db, "org-a", "a@x.example")
    ca = _contact(db, "org-a")
    db.add(_msg(ca, "<dup@x>", a.id))
    db.flush()
    db.add(_msg(ca, "<dup@x>", a.id))
    with pytest.raises(IntegrityError):
        db.flush()


def test_rows_without_mailbox_keep_global_uniqueness(db):
    ca, cb = _contact(db, "org-a"), _contact(db, "org-b")
    db.add(_msg(ca, "<legacy@x>"))
    db.flush()
    db.add(_msg(cb, "<legacy@x>"))
    with pytest.raises(IntegrityError):
        db.flush()


def test_null_message_ids_stay_unconstrained(db):
    a = _mailbox(db, "org-a", "a@x.example")
    ca = _contact(db, "org-a")
    db.add_all([
        _msg(ca, None), _msg(ca, None), _msg(ca, None, a.id), _msg(ca, None, a.id),
    ])
    db.flush()


def test_outbound_message_id_still_globally_unique(db):
    ca, cb = _contact(db, "org-a"), _contact(db, "org-b")
    out = MessageDirection.OUTBOUND.value
    db.add(_msg(ca, "<out@x>", direction=out))
    db.flush()
    db.add(_msg(cb, "<out@x>", direction=out))
    with pytest.raises(IntegrityError):
        db.flush()
