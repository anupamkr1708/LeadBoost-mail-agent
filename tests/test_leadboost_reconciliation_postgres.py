"""
Real-PostgreSQL checks for the C9.1 reconciliation GET.

Skipped (never faked with SQLite) unless POSTGRES_TEST_URL points at a
reachable PostgreSQL -- same convention as test_external_dispatch_postgres.py.
Dedicated test database only: it is TRUNCATEd.

Proves, on the real engine: tenant scoping and the 404 convention hold with
timezone-aware timestamps, the GET is read-only (row identical afterwards),
and the lookup is index-backed with the unique (organization_id,
idempotency_key) index present in the catalog (no new index / migration).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("psycopg2", reason="psycopg2 not installed -- cannot test real PostgreSQL")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

POSTGRES_TEST_URL = os.environ.get("POSTGRES_TEST_URL")


def _pg_available() -> bool:
    if not POSTGRES_TEST_URL:
        return False
    try:
        eng = create_engine(POSTGRES_TEST_URL)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


pytestmark.append(
    pytest.mark.skipif(not _pg_available(), reason="POSTGRES_TEST_URL not set or PostgreSQL unreachable")
)

from mailer_agent.api.deps import get_current_org_id, require_api_key  # noqa: E402
from mailer_agent.api.main import app  # noqa: E402
from mailer_agent.db import get_db  # noqa: E402
from mailer_agent.models import Base, ExternalDispatch  # noqa: E402
from tests.dispatch_support import seed_dispatch  # noqa: E402

BASE = "/integrations/leadboost/outreach-actions"
_TRUNC = (
    "TRUNCATE external_dispatches, mailboxes, messages, contacts, campaigns, "
    "suppression_list RESTART IDENTITY CASCADE"
)


@pytest.fixture()
def pg():
    eng = create_engine(POSTGRES_TEST_URL)
    Base.metadata.create_all(bind=eng)
    with eng.begin() as c:
        c.execute(text(_TRUNC))
    Session = sessionmaker(bind=eng)
    session = Session()
    holder = {"org": "org-a"}
    app.dependency_overrides[get_db] = lambda: session
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_current_org_id] = lambda: holder["org"]
    try:
        yield eng, session, holder, TestClient(app)
    finally:
        app.dependency_overrides.clear()
        session.close()
        with eng.begin() as c:
            c.execute(text(_TRUNC))
        eng.dispose()


@pytest.mark.parametrize("state", ["queued", "sending", "sent", "failed", "unknown"])
def test_states_tenancy_and_read_only_on_postgres(pg, state):
    eng, session, holder, client = pg
    d = seed_dispatch(session, org="org-a", idem="k1", state=state)
    ref, did = d.public_reference, d.id

    def raw():
        with eng.connect() as c:
            return tuple(c.execute(text("SELECT * FROM external_dispatches WHERE id = :i"), {"i": did}).one())

    before = raw()
    ok = client.get(f"{BASE}/k1")
    assert ok.status_code == 200
    assert ok.json()["state"] == state
    assert ok.json()["mailing_agent_reference"] == ref
    assert ok.json()["updated_at"] is not None

    holder["org"] = "org-b"
    cross = client.get(f"{BASE}/k1")
    assert cross.status_code == 404
    assert ref not in cross.text and state not in cross.text

    assert raw() == before  # no write, including updated_at


def test_lookup_is_index_backed_not_a_seq_scan(pg):
    eng, session, _holder, _client = pg
    for i in range(50):  # some rows so the planner has a choice
        seed_dispatch(session, org=f"org-{i % 5}", idem=f"k{i}", email=f"l{i}@example.com")
    with eng.begin() as c:
        c.execute(text("ANALYZE external_dispatches"))
        c.execute(text("SET LOCAL enable_seqscan = off"))
        plan = "\n".join(
            r[0] for r in c.execute(text(
                "EXPLAIN SELECT public_reference, state, updated_at FROM external_dispatches "
                "WHERE organization_id = 'org-1' AND idempotency_key = 'k1'"
            ))
        )
    # Which index the planner picks (the unique one or the single-column
    # organization_id one) depends on table statistics; what matters here
    # is that the predicate is index-served and that the unique index
    # exists -- see test_unique_constraint_present_in_catalog.
    assert "Seq Scan" not in plan and "Index" in plan, plan


def test_unique_constraint_present_in_catalog(pg):
    eng, *_ = pg
    with eng.connect() as c:
        defs = [r[0] for r in c.execute(text(
            "SELECT indexdef FROM pg_indexes WHERE tablename='external_dispatches' "
            "AND indexname='uq_external_dispatches_org_idempotency_key'"
        ))]
    assert len(defs) == 1
    assert "UNIQUE" in defs[0] and "organization_id, idempotency_key" in defs[0]
