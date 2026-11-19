"""
C9.3 -- GET /integrations/leadboost/outreach-actions/{idempotency_key}/conversation.

The conversation read must be: tenant-scoped (cross-tenant == not found),
strictly side-effect free (SELECT only, no SMTP / IMAP / LLM, no mutation, no
claim or lease recovery), faithful to ExternalDispatch.state (never
Message.status), provenance-safe (only mailbox-bound inbound whose mailbox
belongs to the organization), bounded (window, has_more, body cap) and minimal
(explicit allowlist: no ids, Message-IDs, error text, intent or secrets).

Isolation follows tests/test_leadboost_reconciliation.py (C9.1): per-test
in-memory SQLite and dependency_overrides; the auth tests instead monkeypatch
api.deps._KEY_MAP and use the REAL dependencies.
"""

from __future__ import annotations

import copy
import json
import ast
import os
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, event, update
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from mailer_agent.api import deps
from mailer_agent.api.deps import get_current_org_id, get_integration_org_id, require_api_key
from mailer_agent.api.integrations_conversation import BODY_CAP_CHARS, DEFAULT_LIMIT, MAX_LIMIT
from mailer_agent.api.main import app
from mailer_agent.config import get_settings
from mailer_agent.db import get_db
from mailer_agent.mail.exact_message import create_authorized_message
from mailer_agent.models import (
    Base,
    Campaign,
    ExternalDispatch,
    LEGACY_PUBLIC_STATE,
    ExternalDispatchState,
    Mailbox,
    Message,
)
from mailer_agent.schemas import (
    LeadBoostConversation,
    LeadBoostConversationAction,
    LeadBoostConversationMessage,
)
from tests.dispatch_support import MAILBOX_SMTP_PASSWORD, age, seed_mailbox, seed_org_mailboxes
from tests.dispatch_support import seed_dispatch as _seed_dispatch

ORG_A = "org-a"
ORG_B = "org-b"
BASE = "/integrations/leadboost/outreach-actions"
C91 = BASE  # the frozen reconciliation GET lives at BASE/{key}
# The five PUBLIC delivery states, each of which must pass through unchanged. GENERATING and the managed-reply
# AWAITING_APPROVAL / CANCELLED are internal-only: they are mapped (LEGACY_PUBLIC_STATE), never emitted -- see
# test_internal_states_are_mapped_not_leaked below. This list is the same five values it always was.
ALL_STATES = [s.value for s in ExternalDispatchState if s.value not in LEGACY_PUBLIC_STATE]

TOP_KEYS = {"action", "messages", "has_more"}
ACTION_KEYS = {"accepted", "state", "mailing_agent_reference", "created_at", "updated_at", "mailbox_reference"}
MESSAGE_KEYS = {
    "direction", "message_type", "subject", "body", "body_truncated", "created_at",
    "delivery_state", "mailing_agent_reference", "mailbox_reference",
    "message_reference",   # inbound messages only: the reply-parent reference (C9.4)
}
T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def url(key: str, suffix: str = "/conversation") -> str:
    return f"{BASE}/{quote(key, safe='')}{suffix}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _sender_identity(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "leadboost_integration_sender_email", "outreach@mailer.example.com")
    monkeypatch.setattr(s, "leadboost_integration_sender_name", "Test Sender")
    monkeypatch.setattr(s, "leadboost_integration_sender_org", "Test Org")


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=eng)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture()
def db(engine):
    session = sessionmaker(bind=engine)()
    seed_org_mailboxes(session)
    try:
        yield session
    finally:
        session.close()


class _Org:
    def __init__(self, org_id):
        self.org_id = org_id


@pytest.fixture()
def org():
    return _Org(ORG_A)


@pytest.fixture()
def client(db, org):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_integration_org_id] = lambda: org.org_id
    app.dependency_overrides[get_current_org_id] = lambda: org.org_id  # C9.1 parity calls
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def real_auth_client(db, monkeypatch):
    """Only get_db is overridden; the real API-key dependencies run."""
    monkeypatch.setattr(deps, "_KEY_MAP", {"key-a": ORG_A, "key-b": ORG_B})
    app.dependency_overrides[get_db] = lambda: db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def side_effect_spies(monkeypatch, fake_llm):
    """Any SMTP / IMAP / sender / LLM use from the read path fails loudly."""
    import imaplib
    import smtplib

    import mailer_agent.mail.sender as sender

    calls: list[str] = []

    def _boom(name):
        def _f(*a, **kw):
            calls.append(name)
            raise AssertionError(f"{name} must never be used by the conversation read")
        return _f

    monkeypatch.setattr(sender, "send_email", _boom("send_email"))
    for cls in ("SMTP", "SMTP_SSL"):
        monkeypatch.setattr(smtplib, cls, _boom(f"smtplib.{cls}"))
    for cls in ("IMAP4", "IMAP4_SSL"):
        monkeypatch.setattr(imaplib, cls, _boom(f"imaplib.{cls}"))
    return calls


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

def seed_dispatch(db, **kw) -> ExternalDispatch:
    """dispatch_support.seed_dispatch, with the outbound message pinned to T0 so
    ordering against the explicit-timestamp inbound rows below is deterministic."""
    d = _seed_dispatch(db, **kw)
    db.get(Message, d.message_id).created_at = T0
    db.commit()
    return d


def add_message(
    db, contact_id, *, direction="inbound", body="Thanks, tell me more.", subject="Re: Quick question",
    mailbox_id=None, message_type=None, status="received", at=None, **extra,
) -> Message:
    m = Message(
        contact_id=contact_id, direction=direction, message_type=message_type, subject=subject,
        body=body, status=status, mailbox_id=mailbox_id, created_at=at or T0, **extra,
    )
    db.add(m)
    db.commit()
    return m


def add_dispatch_same_contact(db, first: ExternalDispatch, *, idem, state="queued", subject="Second", body="Second body") -> ExternalDispatch:
    """A second action to the SAME recipient: same Contact, new Message, new dispatch."""
    msg = create_authorized_message(contact_id=first.contact_id, subject=subject, body=body)
    db.add(msg)
    db.flush()
    d = ExternalDispatch(
        organization_id=first.organization_id, idempotency_key=idem, campaign_id=first.campaign_id,
        contact_id=first.contact_id, message_id=msg.id, mailbox_id=first.mailbox_id,
        request_fingerprint="e" * 64, public_reference=f"ref-{idem}", state=state,
    )
    db.add(d)
    db.commit()
    return d


def dump_database(db) -> dict:
    db.expire_all()
    out = {}
    for t in Base.metadata.sorted_tables:
        pk = list(t.primary_key.columns)
        out[t.name] = [tuple(r) for r in db.execute(t.select().order_by(*pk)).all()]
    return out


# ---------------------------------------------------------------------------
# Shape, own-organization read, state (authoritative = ExternalDispatch)
# ---------------------------------------------------------------------------

def test_own_org_read_has_exact_shape(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state="sent")
    r = client.get(url("k1"))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == TOP_KEYS
    assert set(body["action"]) == ACTION_KEYS
    assert body["action"]["accepted"] is True
    assert body["action"]["state"] == "sent"
    assert body["action"]["mailing_agent_reference"] == d.public_reference
    assert body["has_more"] is False
    assert len(body["messages"]) == 1
    m = body["messages"][0]
    assert set(m) == MESSAGE_KEYS
    assert m["direction"] == "outbound"
    assert m["message_type"] == "initial_outreach"
    assert m["subject"] == "Quick question"
    assert m["body"].startswith("Hello there")
    assert m["body_truncated"] is False
    assert m["delivery_state"] == "sent"
    assert m["mailing_agent_reference"] == d.public_reference


@pytest.mark.parametrize("state", ALL_STATES)
def test_every_dispatch_state_passes_through(client, db, state):
    seed_dispatch(db, org=ORG_A, idem="k1", state=state)
    body = client.get(url("k1")).json()
    assert body["action"]["state"] == state
    assert body["messages"][0]["delivery_state"] == state


def test_unknown_is_never_translated(client, db):
    seed_dispatch(db, org=ORG_A, idem="k1", state="unknown")
    body = client.get(url("k1")).json()
    assert body["action"]["state"] == "unknown"
    assert body["messages"][0]["delivery_state"] == "unknown"


@pytest.mark.parametrize(
    "internal, public",
    [("awaiting_approval", "queued"), ("cancelled", "failed"), ("generating", "queued")],
)
def test_internal_states_are_mapped_not_leaked(client, db, internal, public):
    """A held managed-reply draft has sent nothing (queued); a cancelled one never will (failed).
    LeadBoost validates this vocabulary strictly, so an internal value must never be emitted."""
    seed_dispatch(db, org=ORG_A, idem="k1", state=internal)
    body = client.get(url("k1")).json()
    assert body["action"]["state"] == public
    assert body["messages"][0]["delivery_state"] == public


def test_public_vocabulary_is_exactly_the_five_c91_states():
    assert sorted(ALL_STATES) == ["failed", "queued", "sending", "sent", "unknown"]


def test_generating_reads_as_queued_like_c91(client, db):
    seed_dispatch(db, org=ORG_A, idem="k1", state="generating", claimed_by="gen-w")
    body = client.get(url("k1")).json()
    assert body["action"]["state"] == "queued"
    assert body["messages"][0]["delivery_state"] == "queued"


@pytest.mark.parametrize("state", ALL_STATES + ["generating"])
def test_state_parity_with_c91(client, db, state):
    """The conversation's action.state is exactly what the frozen C9.1 GET reports."""
    d = seed_dispatch(db, org=ORG_A, idem="k1", state=state)
    c91 = client.get(f"{C91}/k1").json()
    conv = client.get(url("k1")).json()
    assert conv["action"]["state"] == c91["state"]
    assert conv["action"]["mailing_agent_reference"] == c91["mailing_agent_reference"] == d.public_reference
    assert conv["messages"][0]["delivery_state"] == c91["state"]


def test_delivery_state_comes_from_the_dispatch_never_from_message_status(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state="sent")
    db.get(Message, d.message_id).status = "failed"   # a divergent mirror must not be "corrected" into the answer
    db.commit()
    body = client.get(url("k1")).json()
    assert body["action"]["state"] == "sent"
    assert body["messages"][0]["delivery_state"] == "sent"

    d2 = seed_dispatch(db, org=ORG_A, idem="k2", state="queued", email="other@example.com")
    db.get(Message, d2.message_id).status = "sent"    # message says sent, dispatch says not
    db.commit()
    body2 = client.get(url("k2")).json()
    assert body2["action"]["state"] == "queued"
    assert body2["messages"][0]["delivery_state"] == "queued"


def test_deferred_message_dispatch_is_readable_with_no_messages(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state="queued")
    msg_id = d.message_id
    d.message_id = None                       # accepted, generation still pending
    db.commit()
    db.delete(db.get(Message, msg_id))
    db.commit()
    body = client.get(url("k1")).json()
    assert body["action"]["state"] == "queued"
    assert body["messages"] == [] and body["has_more"] is False


def test_unrecognised_stored_state_is_a_plain_404_not_a_guess(client, db):
    seed_dispatch(db, org=ORG_A, idem="k1", state="sent")
    db.execute(update(ExternalDispatch).values(state="bogus-state"))
    db.commit()
    r = client.get(url("k1"))
    assert r.status_code == 404
    assert "bogus" not in r.text


# ---------------------------------------------------------------------------
# Conversation identity: per recipient (contact), not per action
# ---------------------------------------------------------------------------

def test_two_actions_to_one_recipient_share_the_conversation(client, db):
    a = seed_dispatch(db, org=ORG_A, idem="k1", state="sent")
    db.get(Message, a.message_id).created_at = T0
    db.commit()
    b = add_dispatch_same_contact(db, a, idem="k2", state="queued", subject="Follow-up", body="Second body")
    db.get(Message, b.message_id).created_at = T0 + timedelta(hours=1)
    db.commit()

    for key in ("k1", "k2"):
        body = client.get(url(key)).json()
        assert [m["body"][:6] for m in body["messages"]] == ["Hello ", "Second"]
        # each outbound message carries ITS OWN dispatch's state and reference
        assert [m["delivery_state"] for m in body["messages"]] == ["sent", "queued"]
        assert [m["mailing_agent_reference"] for m in body["messages"]] == [a.public_reference, b.public_reference]
    # ...while action.* is the requested root
    assert client.get(url("k1")).json()["action"]["state"] == "sent"
    assert client.get(url("k2")).json()["action"]["state"] == "queued"


def test_other_recipients_messages_never_appear(client, db):
    a = seed_dispatch(db, org=ORG_A, idem="k1", email="a@example.com")
    b = seed_dispatch(db, org=ORG_A, idem="k2", email="b@example.com", body="Only for b")
    add_message(db, b.contact_id, mailbox_id=b.mailbox_id, body="Reply from b")
    body = client.get(url("k1")).json()
    assert [m["direction"] for m in body["messages"]] == ["outbound"]
    assert "Only for b" not in str(body) and "Reply from b" not in str(body)
    assert a.contact_id != b.contact_id


def test_outbound_row_without_an_org_dispatch_is_excluded(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1")
    add_message(db, d.contact_id, direction="outbound", message_type="follow_up", status="sent",
                body="orphan outbound, no dispatch", at=T0 + timedelta(minutes=5))
    body = client.get(url("k1")).json()
    assert [m["body"][:5] for m in body["messages"]] == ["Hello"]
    assert "orphan" not in str(body)


# ---------------------------------------------------------------------------
# Inbound provenance (D2)
# ---------------------------------------------------------------------------

def test_mailbox_bound_inbound_is_included_oldest_first(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state="sent")
    db.get(Message, d.message_id).created_at = T0
    db.commit()
    from mailer_agent.models import Mailbox
    box = db.get(Mailbox, d.mailbox_id)
    add_message(db, d.contact_id, mailbox_id=box.id, body="Sounds interesting, call me.", at=T0 + timedelta(hours=2))
    body = client.get(url("k1")).json()
    assert [m["direction"] for m in body["messages"]] == ["outbound", "inbound"]
    inbound = body["messages"][1]
    assert set(inbound) == MESSAGE_KEYS
    assert inbound["body"] == "Sounds interesting, call me."
    assert inbound["message_type"] is None
    assert inbound["delivery_state"] is None and inbound["mailing_agent_reference"] is None
    assert inbound["mailbox_reference"] == box.public_reference


def test_inbound_via_another_mailbox_of_the_same_org_is_included(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1")
    other = seed_mailbox(db, org=ORG_A, email="second-box@sender.example.org")
    db.commit()
    add_message(db, d.contact_id, mailbox_id=other.id, body="via the second mailbox", at=T0 + timedelta(hours=1))
    body = client.get(url("k1")).json()
    assert body["messages"][-1]["body"] == "via the second mailbox"
    assert body["messages"][-1]["mailbox_reference"] == other.public_reference


def test_null_mailbox_inbound_is_excluded(client, db):
    """Webhook and legacy-global-IMAP rows (mailbox_id NULL) have no provable owner."""
    d = seed_dispatch(db, org=ORG_A, idem="k1")
    add_message(db, d.contact_id, mailbox_id=None, body="WEBHOOK-OR-LEGACY-INBOUND-SENTINEL",
                message_id_header="<spoofed@attacker.example>", at=T0 + timedelta(hours=1))
    r = client.get(url("k1"))
    assert [m["direction"] for m in r.json()["messages"]] == ["outbound"]
    assert "SENTINEL" not in r.text


def test_inbound_from_a_foreign_orgs_mailbox_never_appears(client, db, org):
    """An inbound row on org-A's contact that is bound to ORG-B's mailbox is not org-A's data."""
    a = seed_dispatch(db, org=ORG_A, idem="k1")
    b = seed_dispatch(db, org=ORG_B, idem="k1", email="b@example.com")
    add_message(db, a.contact_id, mailbox_id=b.mailbox_id, body="FOREIGN-MAILBOX-SENTINEL", at=T0 + timedelta(hours=1))
    assert "FOREIGN-MAILBOX-SENTINEL" not in client.get(url("k1")).text          # org-A view
    org.org_id = ORG_B
    assert "FOREIGN-MAILBOX-SENTINEL" not in client.get(url("k1")).text          # org-B's own contact is unaffected


def test_org_a_mailbox_inbound_is_invisible_to_org_b(client, db, org):
    a = seed_dispatch(db, org=ORG_A, idem="k1", email="same@example.com")
    b = seed_dispatch(db, org=ORG_B, idem="k1", email="same@example.com")
    add_message(db, a.contact_id, mailbox_id=a.mailbox_id, body="ORG-A-REPLY-SENTINEL", at=T0 + timedelta(hours=1))
    org.org_id = ORG_B
    r = client.get(url("k1"))
    assert r.status_code == 200 and "ORG-A-REPLY-SENTINEL" not in r.text
    assert r.json()["action"]["mailing_agent_reference"] == b.public_reference


# ---------------------------------------------------------------------------
# Tenancy / not found / integrity
# ---------------------------------------------------------------------------

def test_missing_key_is_404(client):
    assert client.get(url("does-not-exist")).status_code == 404


@pytest.mark.parametrize("state", ALL_STATES)
def test_cross_tenant_is_404_and_identical_to_missing(client, db, org, state):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state=state)
    add_message(db, d.contact_id, mailbox_id=d.mailbox_id, body="private inbound text")
    org.org_id = ORG_B
    cross = client.get(url("k1"))
    missing = client.get(url("never-existed"))
    assert cross.status_code == missing.status_code == 404
    assert cross.json() == missing.json()
    for leak in (d.public_reference, state, "org-a", "lead@example.com", "Quick question", "private inbound text"):
        assert leak not in cross.text


def test_same_key_in_two_orgs_each_sees_only_own(client, db, org):
    a = seed_dispatch(db, org=ORG_A, idem="shared", state="sent", email="a@example.com", body="A's text")
    b = seed_dispatch(db, org=ORG_B, idem="shared", state="failed", email="b@example.com", body="B's text")
    org.org_id = ORG_A
    ra = client.get(url("shared")).json()
    org.org_id = ORG_B
    rb = client.get(url("shared")).json()
    assert (ra["action"]["state"], ra["action"]["mailing_agent_reference"], ra["messages"][0]["body"]) == ("sent", a.public_reference, "A's text")
    assert (rb["action"]["state"], rb["action"]["mailing_agent_reference"], rb["messages"][0]["body"]) == ("failed", b.public_reference, "B's text")


def test_organization_id_in_query_is_ignored(client, db, org):
    seed_dispatch(db, org=ORG_A, idem="k1")
    org.org_id = ORG_B
    r = client.get(url("k1"), params={"organization_id": ORG_A, "org_id": ORG_A, "tenant": ORG_A})
    assert r.status_code == 404


def test_organization_in_headers_or_body_is_ignored(client, db, org):
    seed_dispatch(db, org=ORG_A, idem="k1")
    org.org_id = ORG_B
    r = client.get(url("k1"), headers={"X-Organization-Id": ORG_A, "X-Org-Id": ORG_A})
    assert r.status_code == 404
    r = client.request("GET", url("k1"), json={"organization_id": ORG_A})
    assert r.status_code == 404


def test_integrity_failure_contact_in_another_orgs_campaign_is_the_same_404(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1")
    db.execute(update(Campaign).where(Campaign.id == d.campaign_id).values(organization_id=ORG_B))
    db.commit()
    broken = client.get(url("k1"))
    missing = client.get(url("never-existed"))
    assert broken.status_code == 404 and broken.json() == missing.json()


def _corrupt_foreign_dispatch(db, victim: ExternalDispatch, *, org, idem, state="sent", body="FOREIGN-DISPATCH-SENTINEL"):
    """A dispatch row that claims another organization but points at `victim`'s
    contact. Not reachable through the API -- it models data corruption, which
    the organization predicates must still contain."""
    msg = create_authorized_message(contact_id=victim.contact_id, subject="foreign", body=body)
    msg.created_at = T0 + timedelta(minutes=30)
    db.add(msg)
    db.flush()
    d = ExternalDispatch(
        organization_id=org, idempotency_key=idem, campaign_id=victim.campaign_id,
        contact_id=victim.contact_id, message_id=msg.id, mailbox_id=None,
        request_fingerprint="d" * 64, public_reference=f"foreign-ref-{idem}", state=state,
    )
    db.add(d)
    db.commit()
    return d


def test_root_lookup_filters_on_the_dispatch_organization_itself(client, db):
    """Defense in depth beneath the campaign join: a dispatch row owned by another
    organization is never the root, even if it points at the caller's contact."""
    a = seed_dispatch(db, org=ORG_A, idem="k1")
    foreign = _corrupt_foreign_dispatch(db, a, org=ORG_B, idem="kx")
    r = client.get(url("kx"))
    assert r.status_code == 404
    assert foreign.public_reference not in r.text and "FOREIGN-DISPATCH-SENTINEL" not in r.text


def test_message_ownership_requires_a_dispatch_of_the_callers_organization(client, db):
    """Defense in depth: an outbound row owned only by a foreign organization's
    dispatch is not shown, even on a contact the caller can otherwise read."""
    a = seed_dispatch(db, org=ORG_A, idem="k1")
    _corrupt_foreign_dispatch(db, a, org=ORG_B, idem="kx")
    r = client.get(url("k1"))
    assert r.status_code == 200
    assert "FOREIGN-DISPATCH-SENTINEL" not in r.text and "foreign-ref" not in r.text
    assert [m["body"][:5] for m in r.json()["messages"]] == ["Hello"]


def test_real_api_key_dependency_is_used_and_fails_closed(real_auth_client, db, monkeypatch):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state="sent")
    c = real_auth_client
    assert c.get(url("k1")).status_code == 401                                   # no key
    assert c.get(url("k1"), headers={"X-API-Key": "bogus"}).status_code == 401
    ok = c.get(url("k1"), headers={"X-API-Key": "key-a"})
    assert ok.status_code == 200
    assert ok.json()["action"]["mailing_agent_reference"] == d.public_reference
    cross = c.get(url("k1"), headers={"X-API-Key": "key-b"})
    assert cross.status_code == 404 and d.public_reference not in cross.text

    # No keys configured at all: 503, never an implicit "default" organization.
    monkeypatch.setattr(deps, "_KEY_MAP", {})
    default_org = seed_dispatch(db, org="default", idem="kd", email="d@example.com")
    r = c.get(url("kd"))
    assert r.status_code == 503
    assert default_org.public_reference not in r.text


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------

def test_read_is_select_only_mutates_nothing_and_touches_no_side_effect_path(
    client, db, engine, side_effect_spies, fake_llm,
):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state="sending", claimed_by="w1", claimed_at=age(10_000))
    add_message(db, d.contact_id, mailbox_id=d.mailbox_id, body="a reply", at=T0 + timedelta(hours=1))
    before = dump_database(db)

    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _spy(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    try:
        first = client.get(url("k1"))
        for _ in range(3):
            assert client.get(url("k1")).json() == first.json()      # repeated reads are stable
    finally:
        event.remove(engine, "before_cursor_execute", _spy)

    assert first.status_code == 200 and statements
    non_select = [s for s in statements if not s.lstrip().upper().startswith("SELECT")]
    assert non_select == []                                           # no INSERT / UPDATE / DELETE / DDL
    assert not any("FOR UPDATE" in s.upper() for s in statements)     # no locks, no claim
    assert dump_database(db) == before                                # every table byte-identical, incl. claim fields
    assert side_effect_spies == []                                    # no SMTP / IMAP / sender
    assert fake_llm.call_count == 0                                   # no LLM
    row = db.get(ExternalDispatch, d.id)
    assert row.state == "sending" and row.claimed_by == "w1"          # an expired lease is NOT recovered by a read
    assert first.json()["action"]["state"] == "sending"


@pytest.mark.parametrize("state", ALL_STATES)
def test_no_state_is_advanced_by_reading(client, db, state):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state=state)
    for _ in range(3):
        assert client.get(url("k1")).json()["action"]["state"] == state
    db.expire_all()
    assert db.get(ExternalDispatch, d.id).state == state


# ---------------------------------------------------------------------------
# Information exposure
# ---------------------------------------------------------------------------

# The forbidden-fragment list below is UNCHANGED from the original test; it only moved to module level so that the
# regression tests can prove, on synthetic payloads, what the scan accepts and rejects.
#
# What changed is WHAT it is run over. A conversation response also carries values that are random or wall-clock
# BY CONSTRUCTION and are legitimately public: the opaque uuid4-hex references and the ISO timestamps. A short
# fragment occurs in those by chance -- "535" inside a 32-hex reference or inside the microseconds of
# "2026-11-17T10:27:03.535862", "0.93" inside "...:40.930881" -- which made this test fail roughly 1 run in 50.
# Those leaves are now (1) required to have exactly the shape they may have, (2) compared by the test with the
# exact value in the database, and (3) blanked before the scan. EVERY other value -- all free text (subject, body)
# and every state -- is scanned with the full, unchanged list.
FORBIDDEN_FRAGMENTS = (
    "hunter2", "Traceback", "AUTH PLAIN", "SMTPAuthenticationError", "535", "worker-host-42", ORG_A,
    "corr-LEAK", "481-LEAK", "GROUNDING-SENTINEL", "SEMANTIC-SENTINEL", "CLASSIFIER-REASON-SENTINEL",
    "outbound-abc123", "IN-REPLY-TO", "REFS-ONE", "SYNTHETIC-ID", "interested", "0.93",
    "lead@example.com", "outreach@sender.example.org", "smtp.mailbox.example",
    MAILBOX_SMTP_PASSWORD, "password", "ciphertext", "gAAAA",          # Fernet ciphertexts start with gAAAA
    "campaign_id", "contact_id", "message_id", "organization_id", "claimed_by", "claimed_at",
    "error", "request_fingerprint", "external_action_id", "correlation_id", "detected_intent",
    "semantic_analysis", "in_reply_to", "references", "idempotency",
)

_HEX32 = re.compile(r"[0-9a-f]{32}")
_ISO_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?")
ACTION_REFERENCE_FIELDS = ("mailing_agent_reference", "mailbox_reference")
ACTION_TIMESTAMP_FIELDS = ("created_at", "updated_at")
MESSAGE_REFERENCE_FIELDS = ("mailing_agent_reference", "mailbox_reference", "message_reference")
MESSAGE_TIMESTAMP_FIELDS = ("created_at",)


def _blank_reference(owner: dict, field: str, where: str):
    """Assert an opaque-reference leaf is None or exactly 32 lowercase hex, blank it, return what it was."""
    value = owner[field]
    if value is not None:
        assert isinstance(value, str) and _HEX32.fullmatch(value), f"{where}.{field} is not an opaque 32-hex reference"
    owner[field] = None
    return value


def _blank_timestamp(owner: dict, field: str, where: str):
    """Assert a timestamp leaf is None or a bare ISO-8601 datetime that parses, blank it, return what it was."""
    value = owner[field]
    if value is not None:
        assert isinstance(value, str) and _ISO_DATETIME.fullmatch(value), f"{where}.{field} is not an ISO-8601 datetime"
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:                      # right shape, not a real datetime (month 13, hour 99, ...)
            raise AssertionError(f"{where}.{field} is not a valid datetime: {exc}") from exc
    owner[field] = None
    return value


def scrub_random_leaves(body: dict) -> tuple[dict, dict]:
    """Assert the exact key sets, then validate and blank every random-valued leaf.

    Returns (scrubbed deep copy, the leaf values that were blanked) so the caller can compare the leaves with
    the database. Keys stay in the copy (value None), so key NAMES are still scanned; only the values go."""
    scrubbed = copy.deepcopy(body)
    assert isinstance(scrubbed, dict) and set(scrubbed) == TOP_KEYS, f"top-level keys: {sorted(scrubbed)}"
    action = scrubbed["action"]
    assert isinstance(action, dict) and set(action) == ACTION_KEYS, f"action keys: {sorted(action)}"
    messages = scrubbed["messages"]
    assert isinstance(messages, list)
    leaves: dict = {"action": {}, "messages": []}
    for field in ACTION_REFERENCE_FIELDS:
        leaves["action"][field] = _blank_reference(action, field, "action")
    for field in ACTION_TIMESTAMP_FIELDS:
        leaves["action"][field] = _blank_timestamp(action, field, "action")
    for i, message in enumerate(messages):
        assert isinstance(message, dict) and set(message) == MESSAGE_KEYS, f"messages[{i}] keys: {sorted(message)}"
        row = {}
        for field in MESSAGE_REFERENCE_FIELDS:
            row[field] = _blank_reference(message, field, f"messages[{i}]")
        for field in MESSAGE_TIMESTAMP_FIELDS:
            row[field] = _blank_timestamp(message, field, f"messages[{i}]")
        leaves["messages"].append(row)
    return scrubbed, leaves


def assert_response_is_private(body: dict, forbidden=FORBIDDEN_FRAGMENTS) -> dict:
    """Key sets + leaf shapes, then the forbidden-fragment scan over everything that is not a random leaf."""
    scrubbed, leaves = scrub_random_leaves(body)
    text = json.dumps(scrubbed)
    for fragment in forbidden:
        assert fragment not in text, fragment
    return leaves


def _utc_naive(value) -> datetime:
    """A datetime (or ISO string) as naive UTC, so SQLite (naive) and PostgreSQL (aware) rows compare alike."""
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def test_response_exposes_no_ids_headers_diagnostics_intent_or_secrets(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state="unknown")
    secret = "Traceback (most recent call last): smtp password=hunter2 AUTH PLAIN c2VjcmV0"
    db.execute(update(ExternalDispatch).where(ExternalDispatch.id == d.id).values(
        error_message=secret, claimed_by="worker-host-42:123", correlation_id="corr-LEAK",
        external_action_id="481-LEAK", request_fingerprint="f" * 64,
        grounding_context={"proof": "GROUNDING-SENTINEL"},
    ))
    sent = db.get(Message, d.message_id)
    sent.message_id_header = "<outbound-abc123@mailer.example.com>"
    sent.in_reply_to_header = "<prior-IN-REPLY-TO@mailer.example.com>"
    sent.references_header = "<REFS-ONE@x> <REFS-TWO@x>"
    sent.error_message = "send_email raised SMTPAuthenticationError: 535 bad creds"
    inbound = add_message(
        db, d.contact_id, mailbox_id=d.mailbox_id, body="Please call me.", at=T0 + timedelta(hours=1),
        message_id_header="<inbound-SYNTHETIC-ID@imap.local>", in_reply_to_header="<outbound-abc123@mailer.example.com>",
        references_header="<outbound-abc123@mailer.example.com>", detected_intent="interested",
        intent_confidence=0.93, semantic_analysis={"sentinel": "SEMANTIC-SENTINEL"},
        classification_failure_reason="CLASSIFIER-REASON-SENTINEL",
    )
    db.commit()

    r = client.get(url("k1"))
    assert r.status_code == 200
    body = r.json()
    # Exact key sets, leaf shapes, and the unchanged forbidden-fragment scan over everything but the random leaves.
    leaves = assert_response_is_private(body)

    # The random / wall-clock leaves are not scanned, so each is pinned to the exact value in the database instead.
    db.expire_all()
    action_row = db.get(ExternalDispatch, d.id)
    box = db.get(Mailbox, d.mailbox_id)
    outbound_row = db.get(Message, d.message_id)
    assert [m["direction"] for m in body["messages"]] == ["outbound", "inbound"]
    assert {k: v for k, v in leaves["action"].items() if k in ACTION_REFERENCE_FIELDS} == {
        "mailing_agent_reference": action_row.public_reference,
        "mailbox_reference": box.public_reference,
    }
    assert _utc_naive(leaves["action"]["created_at"]) == _utc_naive(action_row.created_at)
    assert _utc_naive(leaves["action"]["updated_at"]) == _utc_naive(action_row.updated_at)
    out_leaves, in_leaves = leaves["messages"]
    assert {k: v for k, v in out_leaves.items() if k in MESSAGE_REFERENCE_FIELDS} == {
        "mailing_agent_reference": action_row.public_reference,
        "mailbox_reference": box.public_reference,
        "message_reference": None,                      # outbound messages carry no reply-parent reference
    }
    assert _utc_naive(out_leaves["created_at"]) == _utc_naive(outbound_row.created_at) == _utc_naive(T0)
    # C9.4: the ONE intentional reference on an inbound message -- exposed so LeadBoost can answer it. It is the
    # persisted OPAQUE token (32 lowercase hex), never the row id.
    assert {k: v for k, v in in_leaves.items() if k in MESSAGE_REFERENCE_FIELDS} == {
        "mailing_agent_reference": None,                # only outbound messages name their dispatch
        "mailbox_reference": box.public_reference,
        "message_reference": inbound.reply_reference,
    }
    assert re.fullmatch(r"[0-9a-f]{32}", in_leaves["message_reference"])
    assert in_leaves["message_reference"] != str(inbound.id)
    assert _utc_naive(in_leaves["created_at"]) == _utc_naive(inbound.created_at) == _utc_naive(T0 + timedelta(hours=1))

    assert "id" not in body and all("id" not in m for m in body["messages"])
    assert inbound.id not in {v for m in body["messages"] for v in m.values() if isinstance(v, int)}


# ---------------------------------------------------------------------------
# The scan itself, proven deterministically on synthetic payloads (no database, no clock, no seed)
# ---------------------------------------------------------------------------

# Exactly the two chance matches that made the original test flaky:
REF_WITH_535_PREFIX = "61a34535c31d" + "0" * 20            # 32 hex; a real "mailing_agent_reference" looked like this
REF_WITH_535_SUFFIX = "0" * 25 + "c88535b"                  # 32 hex; ends "...c88535b" like a real "mailbox_reference"
REF_PLAIN = "a" * 32
STAMP_WITH_093 = "2026-11-15T05:23:40.930881"                # a real wall-clock "updated_at"; contains "0.93"
STAMP_WITH_535 = "2026-11-17T10:27:03.535862"                # a real wall-clock "created_at"; contains "535"


def _synthetic_response() -> dict:
    return {
        "action": {
            "accepted": True, "state": "unknown", "mailing_agent_reference": REF_WITH_535_PREFIX,
            "created_at": STAMP_WITH_535, "updated_at": STAMP_WITH_093, "mailbox_reference": REF_WITH_535_SUFFIX,
        },
        "messages": [
            {
                "direction": "outbound", "message_type": "initial_outreach", "subject": "Quick question",
                "body": "Hello there,\n\nWould a call next week suit you?\n", "body_truncated": False,
                "created_at": STAMP_WITH_093, "delivery_state": "unknown",
                "mailing_agent_reference": REF_WITH_535_PREFIX, "mailbox_reference": REF_WITH_535_SUFFIX,
                "message_reference": None,
            },
            {
                "direction": "inbound", "message_type": None, "subject": "Re: Quick question",
                "body": "Please call me.", "body_truncated": False, "created_at": STAMP_WITH_535,
                "delivery_state": None, "mailing_agent_reference": None, "mailbox_reference": REF_WITH_535_SUFFIX,
                "message_reference": REF_PLAIN,
            },
        ],
        "has_more": False,
    }


def test_scan_accepts_random_references_and_wall_clock_timestamps_that_contain_forbidden_fragments():
    payload = _synthetic_response()
    text = json.dumps(payload)
    # Guard: this payload really contains the fragments that broke the old whole-response substring scan...
    assert "535" in text and "0.93" in text
    assert [f for f in FORBIDDEN_FRAGMENTS if f in text] == ["535", "0.93"]
    # ...and the scan accepts it, because they sit only in leaves that are shape-checked and blanked.
    leaves = assert_response_is_private(payload)
    assert leaves["action"]["mailing_agent_reference"] == REF_WITH_535_PREFIX
    assert leaves["action"]["updated_at"] == STAMP_WITH_093
    assert payload == _synthetic_response()          # the caller's payload is not mutated


def test_scan_blanks_leaves_in_a_copy_and_keeps_the_key_names_in_scope():
    scrubbed, _ = scrub_random_leaves(_synthetic_response())
    assert scrubbed["action"]["mailbox_reference"] is None and scrubbed["messages"][1]["message_reference"] is None
    assert set(scrubbed["action"]) == ACTION_KEYS and set(scrubbed["messages"][1]) == MESSAGE_KEYS
    assert "535" not in json.dumps(scrubbed) and "0.93" not in json.dumps(scrubbed)


def _mutate(path: tuple, value) -> dict:
    payload = _synthetic_response()
    node = payload
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value
    return payload


def _rejected(payload: dict) -> bool:
    try:
        assert_response_is_private(payload)
    except AssertionError:
        return True
    return False


@pytest.mark.parametrize("where", [
    ("messages", 0, "body"), ("messages", 0, "subject"), ("messages", 1, "body"),
    ("messages", 0, "message_type"), ("messages", 0, "delivery_state"), ("action", "state"),
])
def test_scan_rejects_every_forbidden_fragment_in_any_free_text_or_state_field(where):
    assert not _rejected(_mutate(where, "an ordinary value"))                       # control: only the fragment can cause a rejection
    accepted = [f for f in FORBIDDEN_FRAGMENTS if not _rejected(_mutate(where, f"before {f} after"))]
    assert accepted == [], f"fragments the scan failed to catch in {where}: {accepted}"


@pytest.mark.parametrize("path, value", [
    (("messages", 0, "body"), REF_WITH_535_PREFIX),          # a random-looking hex string is exempt ONLY as a reference leaf...
    (("messages", 0, "subject"), REF_WITH_535_SUFFIX),       # ...never in free text
    (("messages", 1, "body"), STAMP_WITH_093),               # and a timestamp is exempt only as a timestamp leaf
    (("messages", 1, "subject"), STAMP_WITH_535),
])
def test_scan_exempts_by_field_not_by_value(path, value):
    assert _rejected(_mutate(path, value))


BAD_REFERENCES = [
    "lead@example.com", "worker-host-42", "<outbound-abc123@mailer.example.com>", "hunter2", "",
    "A" * 32,                      # upper-case hex is not the format
    "a" * 31, "a" * 33, "a" * 31 + "g",
    "a" * 32 + "\n",
    12345, 535, {"x": "hunter2"}, ["a" * 32],
]
BAD_TIMESTAMPS = [
    "yesterday", "2026-11-15 05:23:40", "20261115", "2026-11-15",
    "2026-11-15T05:23:40.930881 password=hunter2", "Traceback 2026-11-15T05:23:40",
    "2026-13-45T99:99:99",         # right shape, not a real datetime
    1763184220, {"x": "hunter2"},
]


@pytest.mark.parametrize("path", [
    ("action", "mailing_agent_reference"), ("action", "mailbox_reference"),
    ("messages", 0, "mailing_agent_reference"), ("messages", 0, "mailbox_reference"),
    ("messages", 1, "mailbox_reference"), ("messages", 1, "message_reference"),
])
def test_scan_rejects_a_reference_that_is_not_an_opaque_32_hex_token(path):
    assert not _rejected(_mutate(path, REF_PLAIN))                                  # control: a well-formed token is fine here
    accepted = [bad for bad in BAD_REFERENCES if not _rejected(_mutate(path, bad))]
    assert accepted == [], f"values the scan failed to reject at {path}: {accepted}"


@pytest.mark.parametrize("path", [("action", "created_at"), ("action", "updated_at"), ("messages", 0, "created_at"), ("messages", 1, "created_at")])
def test_scan_rejects_a_timestamp_that_is_not_a_bare_iso_datetime(path):
    assert not _rejected(_mutate(path, STAMP_WITH_093))                             # control: a real timestamp is fine here
    accepted = [bad for bad in BAD_TIMESTAMPS if not _rejected(_mutate(path, bad))]
    assert accepted == [], f"values the scan failed to reject at {path}: {accepted}"


def test_scan_accepts_missing_references_and_timezone_suffixed_timestamps():
    payload = _synthetic_response()
    payload["action"]["mailbox_reference"] = None
    payload["action"]["created_at"] = "2026-11-15T05:23:40Z"
    payload["action"]["updated_at"] = "2026-11-15T05:23:40.5+05:30"
    leaves = assert_response_is_private(payload)
    assert leaves["action"]["mailbox_reference"] is None


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(id=1),                                              # a top-level id
    lambda p: p.update(organization_id="org-a"),
    lambda p: p.pop("has_more"),
    lambda p: p["action"].update(organization_id="org-a"),
    lambda p: p["action"].update(error_message="x"),
    lambda p: p["action"].pop("mailbox_reference"),
    lambda p: p["messages"][0].update(id=7),                               # a row id on a message
    lambda p: p["messages"][0].update(message_id_header="<x@y>"),
    lambda p: p["messages"][1].update(detected_intent="interested"),
    lambda p: p["messages"][1].pop("message_reference"),
])
def test_scan_rejects_any_key_outside_or_missing_from_the_exact_key_sets(mutation):
    payload = _synthetic_response()
    mutation(payload)
    assert _rejected(payload)


def test_mailbox_reference_is_opaque_never_an_address(client, db):
    from mailer_agent.models import Mailbox
    d = seed_dispatch(db, org=ORG_A, idem="k1")
    box = db.get(Mailbox, d.mailbox_id)
    body = client.get(url("k1")).json()
    assert body["action"]["mailbox_reference"] == box.public_reference
    assert body["messages"][0]["mailbox_reference"] == box.public_reference
    assert box.email_address not in str(body) and "@" not in box.public_reference


def test_unbound_dispatch_has_null_mailbox_reference(client, db):
    seed_dispatch(db, org=ORG_A, idem="k1", mailbox=False)
    body = client.get(url("k1")).json()
    assert body["action"]["mailbox_reference"] is None
    assert body["messages"][0]["mailbox_reference"] is None


def test_response_models_reject_undeclared_fields():
    ok_action = dict(state="sent", mailing_agent_reference="r")
    ok_msg = dict(direction="inbound", body="b")
    with pytest.raises(ValidationError):
        LeadBoostConversationAction(**ok_action, error_message="x")
    with pytest.raises(ValidationError):
        LeadBoostConversationMessage(**ok_msg, message_id_header="<x>")
    with pytest.raises(ValidationError):
        LeadBoostConversation(action=LeadBoostConversationAction(**ok_action), messages=[], has_more=False, organization_id="o")
    with pytest.raises(ValidationError):
        LeadBoostConversationAction(state="generating", mailing_agent_reference="r")   # internal state is not public vocabulary
    with pytest.raises(ValidationError):
        LeadBoostConversationMessage(direction="sideways", body="b")


# ---------------------------------------------------------------------------
# Window: limit, has_more, ordering
# ---------------------------------------------------------------------------

def _thread(db, d, n_inbound):
    """Outbound at T0 plus n_inbound replies one minute apart: n_inbound + 1 messages."""
    db.get(Message, d.message_id).created_at = T0
    db.commit()
    for i in range(n_inbound):
        add_message(db, d.contact_id, mailbox_id=d.mailbox_id, body=f"reply-{i:03d}", at=T0 + timedelta(minutes=i + 1))


@pytest.mark.parametrize("bad", ["0", "-1", "51", "1000", "abc", "", "1.5"])
def test_invalid_limit_is_422(client, db, bad):
    seed_dispatch(db, org=ORG_A, idem="k1")
    assert client.get(url("k1"), params={"limit": bad}).status_code == 422


def test_limit_bounds_are_accepted(client, db):
    seed_dispatch(db, org=ORG_A, idem="k1")
    assert client.get(url("k1"), params={"limit": 1}).status_code == 200
    assert client.get(url("k1"), params={"limit": MAX_LIMIT}).status_code == 200
    assert (DEFAULT_LIMIT, MAX_LIMIT) == (20, 50)


def test_default_window_is_the_most_recent_20_oldest_first_with_has_more(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1")
    _thread(db, d, 24)                                   # 25 messages in total
    body = client.get(url("k1")).json()
    assert len(body["messages"]) == 20 and body["has_more"] is True
    bodies = [m["body"] for m in body["messages"]]
    assert bodies == [f"reply-{i:03d}" for i in range(4, 24)]    # newest 20; the outbound + 4 oldest replies are cut
    assert [m["created_at"] for m in body["messages"]] == sorted(m["created_at"] for m in body["messages"])


def test_has_more_is_false_when_the_window_fits_exactly(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1")
    _thread(db, d, 19)                                   # exactly 20 messages
    body = client.get(url("k1")).json()
    assert len(body["messages"]) == 20 and body["has_more"] is False
    assert body["messages"][0]["direction"] == "outbound"


def test_limit_one_returns_only_the_newest(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1")
    _thread(db, d, 3)
    body = client.get(url("k1"), params={"limit": 1}).json()
    assert [m["body"] for m in body["messages"]] == ["reply-002"] and body["has_more"] is True


def test_max_limit_returns_everything_up_to_50(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1")
    _thread(db, d, 59)                                   # 60 messages
    body = client.get(url("k1"), params={"limit": 50}).json()
    assert len(body["messages"]) == 50 and body["has_more"] is True
    assert body["messages"][-1]["body"] == "reply-058"


def test_equal_timestamps_are_ordered_deterministically_by_id(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1")
    db.get(Message, d.message_id).created_at = T0
    db.commit()
    for i in range(3):
        add_message(db, d.contact_id, mailbox_id=d.mailbox_id, body=f"tie-{i}", at=T0 + timedelta(minutes=1))
    bodies = [m["body"] for m in client.get(url("k1")).json()["messages"]]
    assert bodies[1:] == ["tie-0", "tie-1", "tie-2"]


# ---------------------------------------------------------------------------
# Body cap
# ---------------------------------------------------------------------------

def test_body_exactly_at_the_cap_is_not_truncated(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1")
    add_message(db, d.contact_id, mailbox_id=d.mailbox_id, body="x" * BODY_CAP_CHARS, at=T0 + timedelta(hours=1))
    m = client.get(url("k1")).json()["messages"][-1]
    assert len(m["body"]) == BODY_CAP_CHARS and m["body_truncated"] is False


def test_body_over_the_cap_is_cut_and_flagged(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1")
    add_message(db, d.contact_id, mailbox_id=d.mailbox_id, body="y" * (BODY_CAP_CHARS + 5000), at=T0 + timedelta(hours=1))
    m = client.get(url("k1")).json()["messages"][-1]
    assert len(m["body"]) == BODY_CAP_CHARS and m["body_truncated"] is True
    assert set(m["body"]) == {"y"}


def test_body_cap_counts_characters_not_bytes(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1")
    add_message(db, d.contact_id, mailbox_id=d.mailbox_id, body="é" * (BODY_CAP_CHARS + 1), at=T0 + timedelta(hours=1))
    m = client.get(url("k1")).json()["messages"][-1]
    assert len(m["body"]) == BODY_CAP_CHARS and m["body_truncated"] is True


def test_inbound_markup_is_returned_verbatim_as_data(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1")
    evil = '<script>alert(1)</script><img src=x onerror=alert(2)>'
    add_message(db, d.contact_id, mailbox_id=d.mailbox_id, body=evil, subject="<b>hi</b>", at=T0 + timedelta(hours=1))
    r = client.get(url("k1"))
    assert r.headers["content-type"].startswith("application/json")
    m = r.json()["messages"][-1]
    assert m["body"] == evil and m["subject"] == "<b>hi</b>"       # data, not interpreted or sanitised away


# ---------------------------------------------------------------------------
# Keys, routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["plain", "kéy 1:ünï", "tenant/with/slashes", "a b/c?d#e%f", "x/conversation", "-" * 200])
def test_unusual_and_encoded_keys_round_trip(client, db, key):
    seed_dispatch(db, org=ORG_A, idem=key, state="queued")
    r = client.get(url(key))
    assert r.status_code == 200 and r.json()["action"]["state"] == "queued"


def test_long_unknown_key_is_a_plain_404(client):
    assert client.get(url("x" * 1000)).status_code == 404


def test_c91_is_unchanged_and_not_shadowed(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state="sent")
    r = client.get(f"{C91}/k1")
    assert r.status_code == 200
    assert set(r.json()) == {"accepted", "state", "mailing_agent_reference", "updated_at"}
    assert r.json()["mailing_agent_reference"] == d.public_reference
    # the conversation path is a distinct resource and has no write verbs
    for verb in ("post", "put", "patch", "delete"):
        assert getattr(client, verb)(url("k1")).status_code in (404, 405)


# ---------------------------------------------------------------------------
# Architecture guard
# ---------------------------------------------------------------------------

def test_module_has_no_llm_sender_worker_imap_or_write_stack_imports():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "mailer_agent", "api", "integrations_conversation.py")
    tree = ast.parse(open(path).read())
    found: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            found |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            found.add(n.module)
            found |= {f"{n.module}.{a.name}" for a in n.names}
    forbidden = (
        "mailer_agent.llm", "mailer_agent.memory", "mailer_agent.followup", "mailer_agent.semantic",
        "mailer_agent.policy", "mailer_agent.suppression", "mailer_agent.mail",
        "mailer_agent.api.integrations",            # the frozen C9.1 module is neither reused nor modified
        "mailer_agent.api.integrations_generated", "mailer_agent.api.webhooks",
        "mailer_agent.mailbox_secrets", "smtplib", "imaplib", "groq",
    )
    bad = sorted(i for i in found if any(i == f or i.startswith(f + ".") for f in forbidden))
    assert bad == []


def test_module_issues_no_write_or_lock_constructs():
    """Belt and braces for the runtime statement spy: the source contains no
    session mutation, lock or raw-SQL write construct."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "mailer_agent", "api", "integrations_conversation.py")
    tree = ast.parse(open(path).read())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    forbidden_calls = {"add", "add_all", "delete", "merge", "flush", "commit", "execute", "bulk_save_objects",
                       "with_for_update", "text", "insert", "update"}
    assert (attrs | names) & forbidden_calls == set()
