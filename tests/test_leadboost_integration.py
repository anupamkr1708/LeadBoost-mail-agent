"""
Tests for the LeadBoost integration API boundary (Phase C, Batch 1: C2-C4).

Covers: database schema/constraints, tenancy, idempotency (including a
real concurrent-request race, not just repeated sequential calls),
campaign get-or-create race safety, the contact next_action_at
scheduler-isolation invariant (exercised against the real
claim_due_contacts query, not just a column-value assertion), and
acceptance semantics (durable persistence only -- explicitly no SMTP,
no LLM in this batch).

Isolation: every HTTP-level test uses a fresh, per-test, in-memory
SQLite database via FastAPI's dependency_overrides for get_db /
require_api_key / get_current_org_id -- the exact pattern
tests/test_flow.py's module docstring documents and explains (avoids
the mailer_agent.db module-level engine singleton and the
api/deps.py._KEY_MAP module-level singleton, both built once at first
import). The two auth tests that exercise the *real* require_api_key /
get_current_org_id functions do so by monkeypatching
mailer_agent.api.deps._KEY_MAP directly rather than via environment
variables, since that map is only ever read from the environment once,
at import time -- setting env vars in a test would silently do nothing.
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from mailer_agent.api.deps import get_current_org_id, require_api_key
from mailer_agent.api.main import app
from mailer_agent.config import get_settings
from mailer_agent.followup.work_claiming import claim_due_contacts, make_worker_id
from mailer_agent.db import get_db
from mailer_agent.models import (
    Base,
    Campaign,
    Contact,
    ContactStatus,
    ExternalDispatch,
    ExternalDispatchState,
    Message,
)

ORG_A = "org-a"
ORG_B = "org-b"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _configure_sender_identity(monkeypatch):
    """
    The integration Campaign get-or-create fails closed (503) unless a
    deployment-level sender identity is configured -- see
    config.py::leadboost_integration_sender_email's docstring and
    api/integrations.py::_get_or_create_integration_campaign. Every test
    below exercises the *configured* path by default; the fail-closed
    path gets its own explicit test.

    Mutates the EXISTING cached Settings singleton's attributes directly
    (via monkeypatch.setattr, which restores them automatically at
    teardown) rather than setting env vars + calling
    get_settings.cache_clear(): get_settings() is a process-wide
    @lru_cache singleton (config.py) read by many modules
    (api/deps.py, db.py, api/integrations.py, ...), and clearing that
    cache from an autouse fixture in this file was found to silently
    break an unrelated, already-passing test in
    tests/e2e/test_full_lifecycle.py, which relies on mutating that same
    singleton in place and expects it to still be the same object later
    in its own test body -- clearing the cache mid-suite handed it back
    a fresh instance with defaults instead. See this file's git history
    /commit message for the reproduction. monkeypatch.setattr on the
    object's own attributes changes nothing about which object
    get_settings() returns, so it can't cause that class of
    cross-test-file breakage.
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "leadboost_integration_sender_email", "outreach@mailer.example.com")
    monkeypatch.setattr(settings, "leadboost_integration_sender_name", "Test Sender")
    monkeypatch.setattr(settings, "leadboost_integration_sender_org", "Test Org")


class _OrgHolder:
    """Mutable holder so one dependency_overrides lambda can be pointed
    at a different org mid-test, for cross-tenant checks, without
    rebuilding the fixture."""

    def __init__(self, org_id):
        self.org_id = org_id


@pytest.fixture()
def org_holder():
    return _OrgHolder(ORG_A)


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
def db_session(engine):
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db_session, org_holder):
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_current_org_id] = lambda: org_holder.org_id
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _payload(
    *,
    idempotency_key="idem-1",
    external_action_id="481",
    body="Hello there",
    subject="Hi",
    email="lead@example.com",
    name="Jane Doe",
    correlation_id=None,
):
    return {
        "external_action_id": external_action_id,
        "idempotency_key": idempotency_key,
        "correlation_id": correlation_id,
        "recipient": {"email": email, "name": name},
        "message": {"subject": subject, "body": body},
    }


# ---------------------------------------------------------------------------
# DATABASE (1-9)
# ---------------------------------------------------------------------------

def test_campaign_integration_source_column_exists(engine):
    assert "integration_source" in Campaign.__table__.columns.keys()


def test_external_dispatch_model_exists_with_expected_columns():
    cols = set(ExternalDispatch.__table__.columns.keys())
    assert cols == {
        "id", "organization_id", "idempotency_key", "external_action_id",
        "correlation_id", "campaign_id", "contact_id", "message_id",
        "request_fingerprint", "public_reference", "state", "claimed_by",
        "claimed_at", "error_message", "created_at", "updated_at",
    }


def test_external_dispatch_organization_id_not_nullable(db_session):
    dispatch = ExternalDispatch(
        organization_id=None,  # violates NOT NULL
        idempotency_key="k",
        campaign_id=1,
        contact_id=1,
        message_id=1,
        request_fingerprint="f",
        public_reference="r",
    )
    db_session.add(dispatch)
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_unique_org_and_idempotency_key_enforced(db_session):
    campaign = Campaign(
        name="c", organization_id=ORG_A, integration_source="leadboost",
        sender_name="s", sender_org="s", sender_email="s@example.com", value_prop="v",
    )
    db_session.add(campaign)
    db_session.commit()
    contact = Contact(campaign_id=campaign.id, email="lead@example.com", status=ContactStatus.NEW.value)
    db_session.add(contact)
    db_session.commit()
    message = Message(contact_id=contact.id, direction="outbound", body="hi", status="draft")
    db_session.add(message)
    db_session.commit()

    d1 = ExternalDispatch(
        organization_id=ORG_A, idempotency_key="dup", campaign_id=campaign.id,
        contact_id=contact.id, message_id=message.id,
        request_fingerprint="f1", public_reference="ref1",
    )
    db_session.add(d1)
    db_session.commit()

    d2 = ExternalDispatch(
        organization_id=ORG_A, idempotency_key="dup", campaign_id=campaign.id,
        contact_id=contact.id, message_id=message.id,
        request_fingerprint="f2", public_reference="ref2",
    )
    db_session.add(d2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Same idempotency_key, DIFFERENT org -- must be allowed.
    d3 = ExternalDispatch(
        organization_id=ORG_B, idempotency_key="dup", campaign_id=campaign.id,
        contact_id=contact.id, message_id=message.id,
        request_fingerprint="f3", public_reference="ref3",
    )
    db_session.add(d3)
    db_session.commit()  # must not raise


def test_unique_org_and_integration_source_enforced(db_session):
    c1 = Campaign(
        name="LB", organization_id=ORG_A, integration_source="leadboost",
        sender_name="s", sender_org="s", sender_email="s@example.com", value_prop="v",
    )
    db_session.add(c1)
    db_session.commit()

    c2 = Campaign(
        name="LB dup", organization_id=ORG_A, integration_source="leadboost",
        sender_name="s2", sender_org="s2", sender_email="s2@example.com", value_prop="v2",
    )
    db_session.add(c2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Two ordinary (integration_source=None) campaigns for the SAME org
    # must NOT collide with each other -- NULL != NULL in a unique index.
    c3 = Campaign(
        name="Ordinary 1", organization_id=ORG_A, sender_name="s3",
        sender_org="s3", sender_email="s3@example.com", value_prop="v3",
    )
    c4 = Campaign(
        name="Ordinary 2", organization_id=ORG_A, sender_name="s4",
        sender_org="s4", sender_email="s4@example.com", value_prop="v4",
    )
    db_session.add_all([c3, c4])
    db_session.commit()  # must not raise


def test_public_reference_uniqueness_enforced(db_session):
    campaign = Campaign(
        name="c", organization_id=ORG_A, integration_source="leadboost",
        sender_name="s", sender_org="s", sender_email="s@example.com", value_prop="v",
    )
    db_session.add(campaign)
    db_session.commit()
    contact = Contact(campaign_id=campaign.id, email="lead@example.com", status=ContactStatus.NEW.value)
    db_session.add(contact)
    db_session.commit()
    message = Message(contact_id=contact.id, direction="outbound", body="hi", status="draft")
    db_session.add(message)
    db_session.commit()

    d1 = ExternalDispatch(
        organization_id=ORG_A, idempotency_key="k1", campaign_id=campaign.id,
        contact_id=contact.id, message_id=message.id,
        request_fingerprint="f1", public_reference="same-ref",
    )
    db_session.add(d1)
    db_session.commit()

    d2 = ExternalDispatch(
        organization_id=ORG_B, idempotency_key="k2", campaign_id=campaign.id,
        contact_id=contact.id, message_id=message.id,
        request_fingerprint="f2", public_reference="same-ref",  # collides even across orgs
    )
    db_session.add(d2)
    with pytest.raises(IntegrityError):
        db_session.commit()


def _load_migration_004():
    """Load the real migration module by file path (its filename starts
    with a digit, so it isn't importable as a normal dotted module)."""
    import importlib.util
    import pathlib
    import sys as _sys

    path = (
        pathlib.Path(__file__).parent.parent
        / "migrations"
        / "004_external_dispatch_and_campaign_integration_source.py"
    )
    spec = importlib.util.spec_from_file_location("migration_004", str(path))
    module = importlib.util.module_from_spec(spec)
    _sys.modules["migration_004"] = module
    spec.loader.exec_module(module)
    return module


def test_migration_applies_to_a_genuinely_old_schema(tmp_path):
    """
    Runs the real migration module's functions (not a re-implementation)
    against a database built from a Base that deliberately does NOT have
    integration_source/ExternalDispatch, simulating an actual
    pre-Phase-C production database -- not just a fresh current-schema
    DB where every check trivially already passes.
    """
    from sqlalchemy import Column, Integer, String
    from sqlalchemy.orm import declarative_base

    db_path = tmp_path / "old_schema.db"
    old_engine = create_engine(f"sqlite:///{db_path}")

    OldBase = declarative_base()

    class OldCampaign(OldBase):
        __tablename__ = "campaigns"
        id = Column(Integer, primary_key=True)
        name = Column(String, nullable=False)
        organization_id = Column(String)
        sender_name = Column(String)
        sender_org = Column(String)
        sender_email = Column(String)
        value_prop = Column(String)

    OldBase.metadata.create_all(bind=old_engine)

    migration = _load_migration_004()
    Session = sessionmaker(bind=old_engine)
    db = Session()
    migration.add_campaign_integration_source_column(db)
    migration.add_campaign_integration_source_index(db)
    migration.create_external_dispatches_table(db)
    db.commit()
    assert migration.verify(db) is True
    db.close()


def test_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "idempotent.db"
    eng = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=eng)  # already-current schema

    migration = _load_migration_004()
    Session = sessionmaker(bind=eng)
    for _ in range(2):
        db = Session()
        migration.add_campaign_integration_source_column(db)
        migration.add_campaign_integration_source_index(db)
        migration.create_external_dispatches_table(db)
        db.commit()
        assert migration.verify(db) is True
        db.close()


# ---------------------------------------------------------------------------
# TENANCY (10-14)
# ---------------------------------------------------------------------------

def test_valid_api_key_resolves_expected_org(monkeypatch, db_session):
    import mailer_agent.api.deps as deps_module

    monkeypatch.setattr(deps_module, "_KEY_MAP", {"correct-key": ORG_A})
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        tc = TestClient(app)
        r = tc.post(
            "/integrations/leadboost/outreach-actions",
            json=_payload(),
            headers={"X-API-Key": "correct-key"},
        )
        assert r.status_code == 202
        dispatch = db_session.query(ExternalDispatch).one()
        assert dispatch.organization_id == ORG_A
    finally:
        app.dependency_overrides.clear()


def test_invalid_api_key_rejected(monkeypatch, db_session):
    import mailer_agent.api.deps as deps_module

    monkeypatch.setattr(deps_module, "_KEY_MAP", {"correct-key": ORG_A})
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        tc = TestClient(app)
        r = tc.post(
            "/integrations/leadboost/outreach-actions",
            json=_payload(),
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 401

        r2 = tc.post("/integrations/leadboost/outreach-actions", json=_payload())  # no header
        assert r2.status_code == 401

        assert db_session.query(ExternalDispatch).count() == 0
    finally:
        app.dependency_overrides.clear()


def test_request_body_cannot_select_tenant(client, org_holder, db_session):
    org_holder.org_id = ORG_A
    payload = _payload()
    payload["organization_id"] = ORG_B  # not a real field on the schema
    r = client.post("/integrations/leadboost/outreach-actions", json=payload)
    assert r.status_code == 202
    dispatch = db_session.query(ExternalDispatch).one()
    assert dispatch.organization_id == ORG_A


def test_cross_tenant_record_inaccessible(client, org_holder, db_session):
    org_holder.org_id = ORG_A
    r1 = client.post("/integrations/leadboost/outreach-actions", json=_payload(idempotency_key="shared-key"))
    assert r1.status_code == 202

    org_holder.org_id = ORG_B
    r2 = client.post("/integrations/leadboost/outreach-actions", json=_payload(idempotency_key="shared-key"))
    assert r2.status_code == 202
    assert r2.json()["mailing_agent_reference"] != r1.json()["mailing_agent_reference"]

    dispatches = (
        db_session.query(ExternalDispatch)
        .filter(ExternalDispatch.idempotency_key == "shared-key")
        .all()
    )
    assert len(dispatches) == 2
    assert {d.organization_id for d in dispatches} == {ORG_A, ORG_B}


def test_external_action_id_cannot_select_tenant(client, org_holder, db_session):
    org_holder.org_id = ORG_A
    r1 = client.post(
        "/integrations/leadboost/outreach-actions",
        json=_payload(external_action_id="shared-ext-id", idempotency_key="k1"),
    )
    assert r1.status_code == 202

    org_holder.org_id = ORG_B
    r2 = client.post(
        "/integrations/leadboost/outreach-actions",
        json=_payload(external_action_id="shared-ext-id", idempotency_key="k2"),
    )
    assert r2.status_code == 202

    dispatches = (
        db_session.query(ExternalDispatch)
        .filter(ExternalDispatch.external_action_id == "shared-ext-id")
        .all()
    )
    assert len(dispatches) == 2
    assert {d.organization_id for d in dispatches} == {ORG_A, ORG_B}

    campaigns = db_session.query(Campaign).filter(Campaign.integration_source == "leadboost").all()
    assert {c.organization_id for c in campaigns} == {ORG_A, ORG_B}


# ---------------------------------------------------------------------------
# IDEMPOTENCY (15-20)
# ---------------------------------------------------------------------------

def test_first_request_creates_one_dispatch(client, db_session):
    r = client.post("/integrations/leadboost/outreach-actions", json=_payload())
    assert r.status_code == 202
    assert db_session.query(ExternalDispatch).count() == 1
    assert db_session.query(Message).count() == 1


def test_replay_same_key_same_fingerprint_returns_same_dispatch(client, db_session):
    r1 = client.post("/integrations/leadboost/outreach-actions", json=_payload(idempotency_key="k"))
    r2 = client.post("/integrations/leadboost/outreach-actions", json=_payload(idempotency_key="k"))
    assert r1.json() == r2.json()
    assert db_session.query(ExternalDispatch).count() == 1
    assert db_session.query(Message).count() == 1


def test_replay_same_key_modified_fingerprint_returns_409(client, db_session):
    r1 = client.post(
        "/integrations/leadboost/outreach-actions",
        json=_payload(idempotency_key="k", body="Original body"),
    )
    assert r1.status_code == 202
    r2 = client.post(
        "/integrations/leadboost/outreach-actions",
        json=_payload(idempotency_key="k", body="Modified body"),
    )
    assert r2.status_code == 409
    assert db_session.query(ExternalDispatch).count() == 1
    assert db_session.query(Message).count() == 1
    # The original row is untouched.
    dispatch = db_session.query(ExternalDispatch).one()
    message = db_session.query(Message).one()
    assert message.body == "Original body"
    assert dispatch.request_fingerprint  # unchanged, still the original fingerprint


def test_modified_replay_creates_no_additional_rows(client, db_session):
    client.post("/integrations/leadboost/outreach-actions", json=_payload(idempotency_key="k", subject="A"))
    for _ in range(3):
        client.post("/integrations/leadboost/outreach-actions", json=_payload(idempotency_key="k", subject="B"))
    assert db_session.query(ExternalDispatch).count() == 1
    assert db_session.query(Message).count() == 1
    assert db_session.query(Contact).count() == 1


def test_replay_creates_no_second_message(client, db_session):
    for _ in range(3):
        client.post("/integrations/leadboost/outreach-actions", json=_payload(idempotency_key="k"))
    assert db_session.query(Message).count() == 1


def test_concurrent_identical_requests_converge_on_one_dispatch(tmp_path, org_holder):
    """
    A true DB-level race: N threads hit the real endpoint at (as near as
    Python's GIL allows) the same moment with the SAME idempotency_key.
    Each thread gets its OWN SQLAlchemy Session with its OWN real
    sqlite3 connection (mirroring exactly what mailer_agent.db.get_db()
    does per-request in production), all pointed at one shared
    file-backed SQLite database -- not :memory:+StaticPool, which hands
    every session the literal same sqlite3.Connection object and hits a
    real driver-level bug under genuine multi-thread contention on that
    one connection (SystemError from sqlite3's C extension), which is a
    property of the test harness, not of the code under test. A
    temp-file database gives each thread its own real connection to the
    same data, which is what "concurrent" is actually supposed to mean
    here, and is what exercises
    uq_external_dispatches_org_idempotency_key the way it would really
    be hit in production.
    """
    db_path = tmp_path / "concurrency.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    def _get_db_override():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _get_db_override
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_current_org_id] = lambda: org_holder.org_id
    try:
        test_client = TestClient(app)
        results = []
        barrier = threading.Barrier(5)

        def _fire():
            barrier.wait()
            r = test_client.post(
                "/integrations/leadboost/outreach-actions",
                json=_payload(idempotency_key="race-key"),
            )
            results.append(r)

        threads = [threading.Thread(target=_fire) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        app.dependency_overrides.clear()

    assert all(r.status_code == 202 for r in results), [r.status_code for r in results]
    refs = {r.json()["mailing_agent_reference"] for r in results}
    assert len(refs) == 1, f"expected exactly one converged reference, got {refs}"

    verify_session = SessionLocal()
    try:
        dispatches = (
            verify_session.query(ExternalDispatch)
            .filter(ExternalDispatch.idempotency_key == "race-key")
            .all()
        )
        assert len(dispatches) == 1
        assert verify_session.query(Message).count() == 1
    finally:
        verify_session.close()


# ---------------------------------------------------------------------------
# CAMPAIGN (21-24)
# ---------------------------------------------------------------------------

def test_first_request_creates_integration_campaign(client, db_session):
    client.post("/integrations/leadboost/outreach-actions", json=_payload())
    campaigns = db_session.query(Campaign).filter(Campaign.integration_source == "leadboost").all()
    assert len(campaigns) == 1
    assert campaigns[0].organization_id == ORG_A


def test_subsequent_request_reuses_same_campaign(client, db_session):
    client.post("/integrations/leadboost/outreach-actions", json=_payload(idempotency_key="k1", email="a@example.com"))
    client.post("/integrations/leadboost/outreach-actions", json=_payload(idempotency_key="k2", email="b@example.com"))

    campaigns = db_session.query(Campaign).filter(Campaign.integration_source == "leadboost").all()
    assert len(campaigns) == 1
    dispatches = db_session.query(ExternalDispatch).all()
    assert len(dispatches) == 2
    assert dispatches[0].campaign_id == dispatches[1].campaign_id == campaigns[0].id


def test_concurrent_first_use_creates_exactly_one_campaign(tmp_path, org_holder):
    """Same concurrency shape (and same file-backed-SQLite reasoning --
    see test_concurrent_identical_requests_converge_on_one_dispatch's
    docstring) as the idempotency race test, but with DIFFERENT
    idempotency keys per thread (so each thread legitimately creates its
    own Message + ExternalDispatch) -- the thing under test is that they
    all converge on the SAME campaign despite racing on its creation,
    per uq_campaigns_org_integration_source."""
    db_path = tmp_path / "campaign_concurrency.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    def _get_db_override():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _get_db_override
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_current_org_id] = lambda: org_holder.org_id
    try:
        test_client = TestClient(app)
        results = []
        barrier = threading.Barrier(5)

        def _fire(i):
            barrier.wait()
            r = test_client.post(
                "/integrations/leadboost/outreach-actions",
                json=_payload(idempotency_key=f"key-{i}", email=f"lead{i}@example.com"),
            )
            results.append(r)

        threads = [threading.Thread(target=_fire, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        app.dependency_overrides.clear()

    assert all(r.status_code == 202 for r in results), [r.status_code for r in results]

    verify_session = SessionLocal()
    try:
        campaigns = (
            verify_session.query(Campaign)
            .filter(Campaign.organization_id == org_holder.org_id, Campaign.integration_source == "leadboost")
            .all()
        )
        assert len(campaigns) == 1
        assert verify_session.query(ExternalDispatch).count() == 5
    finally:
        verify_session.close()


def test_ordinary_campaigns_remain_unaffected(client, db_session, org_holder):
    ordinary = Campaign(
        name="Ordinary human campaign",
        organization_id=ORG_A,
        sender_name="Human",
        sender_org="Human Co",
        sender_email="human@example.com",
        value_prop="A real, human-written value prop.",
    )
    db_session.add(ordinary)
    db_session.commit()
    ordinary_id = ordinary.id

    r = client.post("/integrations/leadboost/outreach-actions", json=_payload())
    assert r.status_code == 202

    # The ordinary campaign is untouched, and is NOT reused as the
    # integration campaign.
    refetched = db_session.query(Campaign).filter(Campaign.id == ordinary_id).one()
    assert refetched.integration_source is None
    assert refetched.name == "Ordinary human campaign"

    integration_campaigns = db_session.query(Campaign).filter(Campaign.integration_source == "leadboost").all()
    assert len(integration_campaigns) == 1
    assert integration_campaigns[0].id != ordinary_id


def test_missing_sender_identity_config_fails_closed(client, monkeypatch, db_session):
    """
    See config.py::leadboost_integration_sender_email's docstring and
    the Batch 1 report's SENDER IDENTITY DECISION: rather than silently
    inserting a placeholder sender identity to satisfy Campaign's NOT
    NULL columns, the first dispatch for an org fails closed with 503
    when this deployment-level setting is unset.

    Mutates the cached settings singleton's attribute directly (see
    _configure_sender_identity's docstring for why -- not env vars +
    cache_clear()).
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "leadboost_integration_sender_email", "")
    r = client.post("/integrations/leadboost/outreach-actions", json=_payload())
    assert r.status_code == 503
    assert db_session.query(Campaign).count() == 0
    assert db_session.query(ExternalDispatch).count() == 0


# ---------------------------------------------------------------------------
# CONTACT (25-27)
# ---------------------------------------------------------------------------

def test_new_integration_contact_created(client, db_session):
    client.post("/integrations/leadboost/outreach-actions", json=_payload(email="new@example.com", name="New Lead"))
    contact = db_session.query(Contact).one()
    assert contact.email == "new@example.com"
    assert contact.name == "New Lead"
    assert contact.status == ContactStatus.NEW.value


def test_repeated_recipient_reuses_contact(client, db_session):
    client.post(
        "/integrations/leadboost/outreach-actions",
        json=_payload(idempotency_key="k1", email="same@example.com"),
    )
    client.post(
        "/integrations/leadboost/outreach-actions",
        json=_payload(idempotency_key="k2", email="same@example.com"),
    )
    contacts = db_session.query(Contact).filter(Contact.email == "same@example.com").all()
    assert len(contacts) == 1
    assert db_session.query(Message).count() == 2  # two distinct actions, same contact


def test_integration_contact_next_action_at_remains_null(client, db_session):
    client.post("/integrations/leadboost/outreach-actions", json=_payload())
    contact = db_session.query(Contact).one()
    assert contact.next_action_at is None


def test_existing_integration_contact_with_active_followup_rejects_new_dispatch(client, db_session):
    """
    Batch 1.1 regression test (see
    _get_or_create_integration_contact / _reject_if_contact_has_active_followup
    in api/integrations.py).

    Simulates the real scenario that can populate next_action_at on an
    already-existing integration contact behind this endpoint's back --
    e.g. mail/reply_handler_v2.py's reschedule_after_reply(), or a manual
    /contacts/{id}/force-followup call -- neither of which knows or
    cares that a contact is integration-managed. A second, genuinely new
    LeadBoost dispatch (a fresh idempotency_key) to that same recipient
    must be rejected (409), must create no Message/ExternalDispatch, and
    must NOT silently clear next_action_at to force itself through --
    doing so would destroy a real, live follow-up schedule as an
    invisible side effect.
    """
    from datetime import datetime, timedelta, timezone

    # First dispatch: creates the contact normally, next_action_at NULL.
    r1 = client.post(
        "/integrations/leadboost/outreach-actions",
        json=_payload(idempotency_key="k1", email="active-thread@example.com"),
    )
    assert r1.status_code == 202
    contact = db_session.query(Contact).filter(Contact.email == "active-thread@example.com").one()
    assert contact.next_action_at is None

    # Simulate the reply-handling pipeline (or an admin force-followup)
    # scheduling a real follow-up on this contact, entirely independent
    # of this integration -- exactly what a live, human-attended
    # conversation looks like at the DB level.
    scheduled_at = datetime.now(timezone.utc) + timedelta(days=3)
    contact.next_action_at = scheduled_at
    db_session.commit()

    # A second, genuinely new LeadBoost dispatch to the same recipient.
    r2 = client.post(
        "/integrations/leadboost/outreach-actions",
        json=_payload(idempotency_key="k2", email="active-thread@example.com"),
    )
    assert r2.status_code == 409

    # Zero mutation: no second Message, no second ExternalDispatch, and
    # the live follow-up schedule is completely untouched -- not cleared,
    # not silently overwritten.
    assert db_session.query(Message).count() == 1
    assert db_session.query(ExternalDispatch).count() == 1
    db_session.refresh(contact)
    assert contact.next_action_at is not None
    assert contact.next_action_at.replace(tzinfo=timezone.utc) == scheduled_at


def test_existing_integration_contact_replay_still_succeeds_despite_active_followup(client, db_session):
    """
    The 409-on-active-followup guard must only gate genuinely NEW
    dispatch attempts, never a replay of an already-accepted
    idempotency_key -- a replay returns the existing operation before
    ever reaching contact get-or-create at all.
    """
    from datetime import datetime, timedelta, timezone

    r1 = client.post(
        "/integrations/leadboost/outreach-actions",
        json=_payload(idempotency_key="k1", email="replay-thread@example.com"),
    )
    assert r1.status_code == 202

    contact = db_session.query(Contact).filter(Contact.email == "replay-thread@example.com").one()
    contact.next_action_at = datetime.now(timezone.utc) + timedelta(days=1)
    db_session.commit()

    # Replay of the SAME idempotency_key/payload must still succeed.
    r2 = client.post(
        "/integrations/leadboost/outreach-actions",
        json=_payload(idempotency_key="k1", email="replay-thread@example.com"),
    )
    assert r2.status_code == 202
    assert r2.json() == r1.json()


def test_scheduler_does_not_pick_up_integration_contact(client, db_session):
    """
    Real regression test against the actual scheduler query
    (claim_due_contacts), not just a column-value assertion: even when
    asked for NEW-status contacts (exactly what
    dispatch_new_contacts_job asks for -- see followup/scheduler.py),
    the integration-created contact must never be claimed, because its
    next_action_at is NULL and claim_due_contacts requires
    next_action_at IS NOT NULL AND next_action_at <= now.
    """
    client.post("/integrations/leadboost/outreach-actions", json=_payload())
    contact = db_session.query(Contact).one()
    assert contact.status == ContactStatus.NEW.value
    assert contact.next_action_at is None

    claimed = claim_due_contacts(
        db_session, make_worker_id("test"), status=ContactStatus.NEW.value
    )
    assert claimed == []
    db_session.refresh(contact)
    assert contact.claimed_by is None


# ---------------------------------------------------------------------------
# ACCEPTANCE (28-31)
# ---------------------------------------------------------------------------

def test_accepted_true_after_durable_commit(client, db_session):
    r = client.post("/integrations/leadboost/outreach-actions", json=_payload())
    assert r.status_code == 202
    body = r.json()
    assert body["accepted"] is True
    assert body["mailing_agent_reference"]

    dispatch = db_session.query(ExternalDispatch).one()
    assert dispatch.public_reference == body["mailing_agent_reference"]
    assert dispatch.state == ExternalDispatchState.QUEUED.value


def test_public_reference_is_opaque_not_sequential_id(client):
    r = client.post("/integrations/leadboost/outreach-actions", json=_payload())
    ref = r.json()["mailing_agent_reference"]
    assert not ref.isdigit()
    assert len(ref) == 32  # uuid4().hex length


def test_no_smtp_called(client, monkeypatch):
    calls = []
    import mailer_agent.mail.sender as sender_module

    monkeypatch.setattr(sender_module, "send_email", lambda *a, **kw: calls.append((a, kw)))
    r = client.post("/integrations/leadboost/outreach-actions", json=_payload())
    assert r.status_code == 202
    assert calls == []


def test_no_llm_called(client, monkeypatch):
    calls = []
    import mailer_agent.llm.agent as agent_module

    monkeypatch.setattr(agent_module, "draft_message", lambda *a, **kw: calls.append((a, kw)))
    r = client.post("/integrations/leadboost/outreach-actions", json=_payload())
    assert r.status_code == 202
    assert calls == []
