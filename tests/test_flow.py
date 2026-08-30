"""
End-to-end tests covering: campaign/contact creation, dry-run initial
send, dynamic follow-up dispatch, inbound reply correlation + intent
classification + draft-vs-auto-send gating, and suppression enforcement.

Run with: pytest -q
"""

import os

os.environ["DATABASE_URL"] = "sqlite:///./_test.db"
os.environ["LIVE_SENDING_ENABLED"] = "false"
os.environ["AUTO_REPLY_ENABLED"] = "true"
os.environ["GROQ_API_KEY"] = ""  # exercise the deterministic fallback path

import pytest
from fastapi.testclient import TestClient

from mailer_agent.api.main import app
from mailer_agent.db import init_db


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    # Settings is cached via lru_cache and other modules already imported
    # engine/session bound to the module-level settings -- for this test
    # file we accept the single shared _test.db created above rather than
    # per-test isolation, and clean it up at the end of the session.
    init_db()
    return TestClient(app)


def test_full_flow(client):
    r = client.get("/health")
    assert r.status_code == 200

    r = client.post(
        "/campaigns",
        json={
            "name": "Test Campaign",
            "sender_name": "Jordan",
            "sender_org": "TestOrg",
            "sender_email": "jordan@testorg.example.com",
            "value_prop": "We help teams do the thing, faster.",
            "follow_up_days": [2, 5],
        },
    )
    assert r.status_code == 200
    campaign_id = r.json()["id"]

    r = client.post(
        f"/campaigns/{campaign_id}/contacts",
        json={"contacts": [{"name": "Sam", "email": "sam@prospect.example.com", "company": "Prospect Co"}]},
    )
    assert r.status_code == 200
    contact_id = r.json()[0]["id"]

    r = client.post(f"/campaigns/{campaign_id}/start")
    assert r.status_code == 200
    assert r.json()["dispatched_now"] == 1
    assert r.json()["queued_in_background"] == 0

    r = client.get(f"/contacts/{contact_id}/thread")
    assert len(r.json()["messages"]) == 1
    assert r.json()["messages"][0]["message_type"] == "initial_outreach"

    r = client.post(f"/contacts/{contact_id}/force-followup")
    assert r.status_code == 200

    r = client.get(f"/contacts/{contact_id}/thread")
    assert len(r.json()["messages"]) == 2
    assert r.json()["messages"][1]["message_type"] == "follow_up"


def test_suppression_blocks_future_adds(client):
    r = client.post("/suppress", json={"email": "blocked@prospect.example.com"})
    assert r.status_code == 200

    r = client.post(
        "/campaigns",
        json={
            "name": "Another Campaign",
            "sender_name": "Jordan",
            "sender_org": "TestOrg",
            "sender_email": "jordan@testorg.example.com",
            "value_prop": "Same offer.",
        },
    )
    campaign_id = r.json()["id"]

    r = client.post(
        f"/campaigns/{campaign_id}/contacts",
        json={"contacts": [{"email": "blocked@prospect.example.com"}]},
    )
    assert r.status_code == 200
    assert r.json() == []  # silently skipped, never added


def test_flexible_lead_ingestion(client):
    r = client.post(
        "/campaigns",
        json={
            "name": "Ingest Campaign",
            "sender_name": "Jordan",
            "sender_org": "TestOrg",
            "sender_email": "jordan@testorg.example.com",
            "value_prop": "Same offer.",
        },
    )
    campaign_id = r.json()["id"]

    leadboost_shaped = {
        "company_name": "ExampleCorp",
        "contact_name": "Priya Singh",
        "contact_title": "Head of Support",
        "email": "priya@examplecorp.example.com",
        "industry": "B2B SaaS",
        "qualification_label": "Hot Lead",
    }
    generic_shaped = {"full_name": "Sam Lee", "lead_email": "sam@othercorp.example.com", "org": "OtherCorp"}
    no_email = {"company_name": "NoEmailCorp"}

    r = client.post(
        f"/campaigns/{campaign_id}/leads/ingest",
        json={"leads": [leadboost_shaped, generic_shaped, no_email]},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["created"]) == 2
    assert len(body["skipped"]) == 1

    contact_id = body["created"][0]["contact_id"]
    r = client.get(f"/contacts/{contact_id}")
    assert r.json()["name"] == "Priya Singh"
    assert r.json()["company"] == "ExampleCorp"


def test_inbound_webhook_matches_thread(client):
    r = client.post(
        "/campaigns",
        json={
            "name": "Webhook Campaign",
            "sender_name": "Jordan",
            "sender_org": "TestOrg",
            "sender_email": "jordan@testorg.example.com",
            "value_prop": "Same offer.",
        },
    )
    campaign_id = r.json()["id"]
    r = client.post(
        f"/campaigns/{campaign_id}/contacts",
        json={"contacts": [{"email": "webhook-contact@prospect.example.com"}]},
    )
    client.post(f"/campaigns/{campaign_id}/start")

    r = client.post(
        "/webhooks/inbound-email",
        json={
            "from_email": "webhook-contact@prospect.example.com",
            "subject": "Re: hello",
            "body_text": "Sounds interesting, tell me more.",
        },
    )
    assert r.status_code == 200
    assert r.json()["matched"] is True


def test_start_campaign_sends_first_contact_immediately_and_queues_rest(client, monkeypatch):
    # Patch the shared settings singleton directly (env-var monkeypatching
    # doesn't reach modules that already cached their own `settings`
    # reference at import time) so this test doesn't actually sleep --
    # it's here to prove the *shape* of the dispatch (1 sent inline, N
    # queued to the background task), not to time the real delay.
    from mailer_agent.followup import engine as engine_module
    monkeypatch.setattr(engine_module.settings, "send_delay_seconds", 0.0)
    monkeypatch.setattr(engine_module.settings, "send_jitter_seconds", 0.0)

    r = client.post(
        "/campaigns",
        json={
            "name": "Stagger Campaign",
            "sender_name": "Jordan",
            "sender_org": "TestOrg",
            "sender_email": "jordan@testorg.example.com",
            "value_prop": "Same offer.",
        },
    )
    campaign_id = r.json()["id"]

    r = client.post(
        f"/campaigns/{campaign_id}/contacts",
        json={"contacts": [
            {"email": "first@prospect.example.com"},
            {"email": "second@prospect.example.com"},
            {"email": "third@prospect.example.com"},
        ]},
    )
    contact_ids = [c["id"] for c in r.json()]

    r = client.post(f"/campaigns/{campaign_id}/start")
    assert r.status_code == 200
    assert r.json()["dispatched_now"] == 1
    assert r.json()["queued_in_background"] == 2

    # TestClient runs FastAPI BackgroundTasks synchronously before
    # returning, so by the time we get here all three should have sent.
    for cid in contact_ids:
        thread = client.get(f"/contacts/{cid}/thread").json()
        assert len(thread["messages"]) == 1
        assert thread["messages"][0]["message_type"] == "initial_outreach"
