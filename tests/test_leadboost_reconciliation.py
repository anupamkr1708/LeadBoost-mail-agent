"""
C9.1 -- GET /integrations/leadboost/outreach-actions/{idempotency_key}.

The reconciliation read must be: tenant-scoped (cross-tenant == not found),
strictly side-effect free (no SMTP, no LLM, no mutation), faithful to the
durable ExternalDispatch state (UNKNOWN stays UNKNOWN), and minimal (no
internal ids, no raw error text).

Isolation follows tests/test_leadboost_integration.py: per-test in-memory
SQLite and dependency_overrides. The auth tests instead monkeypatch
api.deps._KEY_MAP and use the REAL dependencies.
"""

from __future__ import annotations

import ast
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from mailer_agent.api import deps
from mailer_agent.api.deps import get_current_org_id, require_api_key
from mailer_agent.api.main import app
from mailer_agent.config import get_settings
from mailer_agent.db import get_db
from mailer_agent.models import Base, ExternalDispatch, ExternalDispatchState, Message
from tests.dispatch_support import seed_dispatch

ORG_A = "org-a"
ORG_B = "org-b"
BASE = "/integrations/leadboost/outreach-actions"
ALL_STATES = [s.value for s in ExternalDispatchState]
EXPECTED_KEYS = {"accepted", "state", "mailing_agent_reference", "updated_at"}


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
    app.dependency_overrides[get_current_org_id] = lambda: org.org_id
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
def send_spy(monkeypatch):
    """Any SMTP attempt from any route this test touches fails loudly."""
    calls = []

    def _boom(*a, **kw):
        calls.append((a, kw))
        raise AssertionError("send_email must never be called by the reconciliation GET")

    import mailer_agent.mail.sender as sender
    monkeypatch.setattr(sender, "send_email", _boom)
    return calls


def _snapshot(db, model, pk):
    db.expire_all()
    row = db.get(model, pk)
    return {c.name: getattr(row, c.name) for c in model.__table__.columns}


def _post_payload(idem="idem-post", email="lead@example.com"):
    return {
        "external_action_id": "481",
        "idempotency_key": idem,
        "correlation_id": "corr-1",
        "recipient": {"email": email, "name": "Jane"},
        "message": {"subject": "Hi", "body": "Hello there"},
    }


# ---------------------------------------------------------------------------
# States (1-5) and response shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state", ALL_STATES)
def test_get_returns_200_with_exact_durable_state(client, db, state):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state=state)
    r = client.get(f"{BASE}/k1")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == EXPECTED_KEYS
    assert body["accepted"] is True
    assert body["state"] == state
    assert body["mailing_agent_reference"] == d.public_reference


def test_unknown_is_never_translated(client, db):
    seed_dispatch(db, org=ORG_A, idem="k1", state="unknown")
    body = client.get(f"{BASE}/k1").json()
    assert body["state"] == "unknown"
    assert body["state"] not in {"failed", "sent", "queued", "sending"}


# ---------------------------------------------------------------------------
# Not found / tenancy (6-8)
# ---------------------------------------------------------------------------

def test_missing_key_is_404(client):
    r = client.get(f"{BASE}/does-not-exist")
    assert r.status_code == 404


def test_own_tenant_200(client, db):
    seed_dispatch(db, org=ORG_A, idem="k1")
    assert client.get(f"{BASE}/k1").status_code == 200


@pytest.mark.parametrize("state", ALL_STATES)
def test_cross_tenant_is_404_and_identical_to_missing(client, db, org, state):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state=state)
    org.org_id = ORG_B
    cross = client.get(f"{BASE}/k1")
    missing = client.get(f"{BASE}/never-existed")
    assert cross.status_code == 404
    assert cross.status_code == missing.status_code
    assert cross.json() == missing.json()
    text = cross.text
    for leak in (d.public_reference, state, "org-a", "lead@example.com", "Quick question"):
        assert leak not in text


def test_same_idempotency_key_in_two_orgs_each_sees_only_own(client, db, org):
    a = seed_dispatch(db, org=ORG_A, idem="shared", state="sent", email="a@example.com")
    b = seed_dispatch(db, org=ORG_B, idem="shared", state="failed", email="b@example.com")
    org.org_id = ORG_A
    ra = client.get(f"{BASE}/shared").json()
    org.org_id = ORG_B
    rb = client.get(f"{BASE}/shared").json()
    assert (ra["state"], ra["mailing_agent_reference"]) == ("sent", a.public_reference)
    assert (rb["state"], rb["mailing_agent_reference"]) == ("failed", b.public_reference)


def test_organization_id_query_param_is_ignored(client, db, org):
    seed_dispatch(db, org=ORG_A, idem="k1")
    org.org_id = ORG_B
    r = client.get(f"{BASE}/k1", params={"organization_id": ORG_A, "org_id": ORG_A})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Reference consistency (9)
# ---------------------------------------------------------------------------

def test_post_reference_equals_get_reference(client):
    post = client.post(BASE, json=_post_payload("idem-rt"))
    assert post.status_code == 202
    get = client.get(f"{BASE}/idem-rt")
    assert get.status_code == 200
    assert get.json()["mailing_agent_reference"] == post.json()["mailing_agent_reference"]
    assert get.json()["state"] == "queued"
    assert get.json()["accepted"] is True


# ---------------------------------------------------------------------------
# Side-effect freedom (10-19)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state", ALL_STATES)
def test_get_mutates_nothing_and_never_sends(client, db, send_spy, state):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state=state)
    d_id, m_id = d.id, d.message_id
    before_d, before_m = _snapshot(db, ExternalDispatch, d_id), _snapshot(db, Message, m_id)
    counts = (db.query(ExternalDispatch).count(), db.query(Message).count())

    first = client.get(f"{BASE}/k1").json()
    for _ in range(4):
        assert client.get(f"{BASE}/k1").json() == first  # repeated GET is stable

    assert _snapshot(db, ExternalDispatch, d_id) == before_d  # incl. state/updated_at/claim
    assert _snapshot(db, Message, m_id) == before_m
    assert (db.query(ExternalDispatch).count(), db.query(Message).count()) == counts
    assert send_spy == []
    assert first["state"] == state  # terminal and non-terminal states are not advanced


def test_expired_sending_lease_is_not_recovered_by_get(client, db):
    from tests.dispatch_support import age
    seed_dispatch(db, org=ORG_A, idem="k1", state="sending", claimed_by="w1", claimed_at=age(10_000))
    assert client.get(f"{BASE}/k1").json()["state"] == "sending"  # no lazy recovery
    row = db.query(ExternalDispatch).one()
    assert row.state == "sending" and row.claimed_by == "w1"


def test_handler_module_has_no_llm_sender_or_worker_imports():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "mailer_agent", "api", "integrations.py")
    tree = ast.parse(open(path).read())
    found = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            found |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            found.add(n.module)
            found |= {f"{n.module}.{a.name}" for a in n.names}
    forbidden = (
        "mailer_agent.llm.agent", "mailer_agent.llm.provider", "mailer_agent.memory",
        "mailer_agent.followup", "mailer_agent.mail.sender",
        "mailer_agent.mail.external_dispatch_worker", "mailer_agent.suppression",
    )
    bad = sorted(i for i in found if any(i == f or i.startswith(f + ".") for f in forbidden))
    assert bad == []


# ---------------------------------------------------------------------------
# Auth (20)
# ---------------------------------------------------------------------------

def test_real_api_key_dependency_is_used(real_auth_client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state="sent")
    c = real_auth_client
    assert c.get(f"{BASE}/k1").status_code == 401                               # no key
    assert c.get(f"{BASE}/k1", headers={"X-API-Key": "bogus"}).status_code == 401
    ok = c.get(f"{BASE}/k1", headers={"X-API-Key": "key-a"})
    assert ok.status_code == 200
    assert ok.json()["mailing_agent_reference"] == d.public_reference
    cross = c.get(f"{BASE}/k1", headers={"X-API-Key": "key-b"})
    assert cross.status_code == 404
    assert d.public_reference not in cross.text


# ---------------------------------------------------------------------------
# Information exposure (21-23)
# ---------------------------------------------------------------------------

def test_response_exposes_no_internal_ids_tenant_or_diagnostics(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state="unknown")
    secret = "Traceback (most recent call last): smtp password=hunter2 AUTH PLAIN c2VjcmV0"
    row = db.get(ExternalDispatch, d.id)
    row.error_message = secret
    row.claimed_by = "worker-host-42:123"
    db.commit()

    r = client.get(f"{BASE}/k1")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == EXPECTED_KEYS
    text = r.text
    for forbidden in (
        "hunter2", "Traceback", "AUTH PLAIN", "worker-host-42", ORG_A,
        "campaign_id", "contact_id", "message_id", "organization_id",
        "claimed_by", "claimed_at", "error", "lead@example.com", "Quick question",
        "request_fingerprint", "external_action_id", "correlation_id",
    ):
        assert forbidden not in text
    assert "id" not in body


def test_message_id_header_not_exposed(client, db):
    d = seed_dispatch(db, org=ORG_A, idem="k1", state="sent")
    db.get(Message, d.message_id).message_id_header = "<abc123@mailer.example.com>"
    db.commit()
    assert "abc123" not in client.get(f"{BASE}/k1").text


# ---------------------------------------------------------------------------
# Unusual keys (24) -- no new validation rules invented
# ---------------------------------------------------------------------------

def test_long_key_is_plain_404(client):
    assert client.get(f"{BASE}/{'x' * 1000}").status_code == 404


def test_unicode_and_encoded_keys_round_trip(client, db):
    seed_dispatch(db, org=ORG_A, idem="kéy 1:ünï", state="queued")
    r = client.get(f"{BASE}/k%C3%A9y%201:%C3%BCn%C3%AF")
    assert r.status_code == 200
    assert r.json()["state"] == "queued"


def test_empty_key_does_not_match_the_route(client):
    # "/outreach-actions/" has no GET handler: FastAPI answers 404/405/307,
    # never a dispatch record.
    r = client.get(f"{BASE}/", follow_redirects=False)
    assert r.status_code in (404, 405, 307)
    assert "mailing_agent_reference" not in r.text
