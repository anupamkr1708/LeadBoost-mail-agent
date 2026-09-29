"""
Batch 1.1 correction: ExternalDispatch FK durability.

See mailer_agent/models.py::ExternalDispatch's campaign_id/contact_id/
message_id column comments for the full reasoning. Short version: no
code path in this repository currently deletes a Campaign, Contact, or
Message (verified by grep -- no DELETE endpoint, no db.delete() call
exists anywhere today), but the schema itself must not silently allow
it to destroy dispatch history if something outside this application
ever does (an admin tool, a GDPR/data-deletion process, direct operator
SQL). ExternalDispatch's own FKs were changed from CASCADE to RESTRICT
so any such deletion attempt fails loudly instead of silently discarding
the durable idempotency/reconciliation record.

IMPORTANT: SQLite does not enforce foreign key constraints by default
(mailer_agent/db.py sets no `PRAGMA foreign_keys=ON`, deliberately left
that way for the app/test suite generally -- see this file's engine
fixture, which enables it only for its own dedicated connection, not
globally). A test that inserted rows and asserted "delete failed"
against the default engine setup would be a false positive: it would
"pass" even if RESTRICT were never actually configured, because SQLite
would silently ignore the ON DELETE clause entirely. These tests
explicitly turn on FK enforcement for their own engine so they are
actually proving something about the schema, not about SQLAlchemy's
Python-level bookkeeping.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from mailer_agent.models import Base, Campaign, Contact, ContactStatus, ExternalDispatch, Message


@pytest.fixture()
def fk_enforced_session():
    """A dedicated in-memory SQLite engine with real FK enforcement
    turned on for every connection -- see module docstring for why this
    must be explicit and per-fixture, not global."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _seed_full_chain(db) -> tuple[Campaign, Contact, Message, ExternalDispatch]:
    campaign = Campaign(
        name="LB", organization_id="org-a", integration_source="leadboost",
        sender_name="s", sender_org="s", sender_email="s@example.com", value_prop="v",
    )
    db.add(campaign)
    db.commit()

    contact = Contact(campaign_id=campaign.id, email="lead@example.com", status=ContactStatus.NEW.value)
    db.add(contact)
    db.commit()

    message = Message(contact_id=contact.id, direction="outbound", body="hi", status="draft")
    db.add(message)
    db.commit()

    dispatch = ExternalDispatch(
        organization_id="org-a", idempotency_key="k1", campaign_id=campaign.id,
        contact_id=contact.id, message_id=message.id,
        request_fingerprint="f", public_reference="ref1",
    )
    db.add(dispatch)
    db.commit()
    return campaign, contact, message, dispatch


def test_deleting_campaign_with_live_dispatch_is_blocked(fk_enforced_session):
    """Direct delete of a Campaign that an ExternalDispatch (transitively,
    via campaign_id) still references must fail, not silently cascade
    through Contact -> Message -> ExternalDispatch."""
    db = fk_enforced_session
    campaign, contact, message, dispatch = _seed_full_chain(db)

    with pytest.raises(IntegrityError):
        db.execute(text("DELETE FROM campaigns WHERE id = :id"), {"id": campaign.id})
        db.commit()
    db.rollback()

    # Nothing was destroyed.
    assert db.query(Campaign).filter(Campaign.id == campaign.id).count() == 1
    assert db.query(ExternalDispatch).filter(ExternalDispatch.id == dispatch.id).count() == 1


def test_deleting_contact_with_live_dispatch_is_blocked(fk_enforced_session):
    """Same, one level down: deleting the Contact directly (not via the
    Campaign cascade) must also be blocked by ExternalDispatch.contact_id."""
    db = fk_enforced_session
    campaign, contact, message, dispatch = _seed_full_chain(db)

    with pytest.raises(IntegrityError):
        db.execute(text("DELETE FROM contacts WHERE id = :id"), {"id": contact.id})
        db.commit()
    db.rollback()

    assert db.query(Contact).filter(Contact.id == contact.id).count() == 1
    assert db.query(ExternalDispatch).filter(ExternalDispatch.id == dispatch.id).count() == 1


def test_deleting_message_with_live_dispatch_is_blocked(fk_enforced_session):
    db = fk_enforced_session
    campaign, contact, message, dispatch = _seed_full_chain(db)

    with pytest.raises(IntegrityError):
        db.execute(text("DELETE FROM messages WHERE id = :id"), {"id": message.id})
        db.commit()
    db.rollback()

    assert db.query(Message).filter(Message.id == message.id).count() == 1
    assert db.query(ExternalDispatch).filter(ExternalDispatch.id == dispatch.id).count() == 1


def test_external_dispatch_survives_idempotency_key_unaffected_by_blocked_delete(fk_enforced_session):
    """
    The concrete failure mode this correction prevents: if a parent
    delete had been allowed to silently cascade away the
    ExternalDispatch row, a retry with the same idempotency_key would
    find nothing and create a second, duplicate dispatch. Proves the
    positive side: after a (blocked) delete attempt, the original
    dispatch is still exactly the one a same-key lookup finds -- no
    duplicate was created and none is possible while it still exists.
    """
    db = fk_enforced_session
    campaign, contact, message, dispatch = _seed_full_chain(db)

    with pytest.raises(IntegrityError):
        db.execute(text("DELETE FROM campaigns WHERE id = :id"), {"id": campaign.id})
        db.commit()
    db.rollback()

    found = (
        db.query(ExternalDispatch)
        .filter(ExternalDispatch.organization_id == "org-a", ExternalDispatch.idempotency_key == "k1")
        .all()
    )
    assert len(found) == 1
    assert found[0].id == dispatch.id
