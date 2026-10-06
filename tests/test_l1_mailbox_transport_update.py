"""
L1 -- Mailbox PATCH accepts SMTP transport metadata (SQLite, deterministic).

Why: LeadBoost's EmailAccount can change smtp_host / smtp_port / security mode /
username after a mailbox has been provisioned. Before L1 the Mailer PATCH only
accepted status + passwords, so those changes could not be synchronized.

Hard limits asserted here: identity (email_address, organization_id,
public_reference) is still immutable, IMAP metadata is still not updatable
(M3), the organization still comes only from the API key, and no response or
log ever carries a secret.
"""

from __future__ import annotations

import logging

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailer_agent.api import deps
from mailer_agent.api.main import app
from mailer_agent.config import get_settings
from mailer_agent.db import get_db
from mailer_agent.mailbox_secrets import get_mailbox_credentials
from mailer_agent.models import Base, Mailbox

ORG_A, ORG_B = "org-a", "org-b"
HDR_A, HDR_B = {"X-API-Key": "key-a"}, {"X-API-Key": "key-b"}
KEY = Fernet.generate_key().decode()
SMTP_PW = "SmtpSecret-L1-aa11"
NEW_PW = "NewSmtpSecret-L1-bb22"


def _body(**over):
    base = {
        "email_address": "sales@example.com",
        "smtp_host": "smtp.example.com", "smtp_port": 587, "smtp_use_tls": True,
        "smtp_username": "sales@example.com", "smtp_password": SMTP_PW,
    }
    base.update(over)
    return base


@pytest.fixture()
def sm(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path/'l1.db'}", connect_args={"check_same_thread": False, "timeout": 15})
    Base.metadata.create_all(bind=eng)
    yield sessionmaker(bind=eng)
    eng.dispose()


@pytest.fixture()
def client(sm, monkeypatch):
    monkeypatch.setattr(deps, "_KEY_MAP", {"key-a": ORG_A, "key-b": ORG_B})
    monkeypatch.setattr(get_settings(), "mailbox_encryption_key", SecretStr(KEY))

    def _get_db():
        s = sm()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _create(client, headers=HDR_A, **over):
    r = client.post("/mailboxes", json=_body(**over), headers=headers)
    assert r.status_code == 201, r.text
    return r.json()["public_reference"]


def _row(sm, ref):
    s = sm()
    try:
        m = s.query(Mailbox).filter_by(public_reference=ref).one()
        s.expunge(m)
        return m
    finally:
        s.close()


def test_each_transport_field_is_updatable_alone_and_others_are_untouched(client, sm):
    ref = _create(client)
    before = _row(sm, ref)
    for patch, attr, expected in (
        ({"smtp_host": "smtp.new.example.com"}, "smtp_host", "smtp.new.example.com"),
        ({"smtp_port": 2525}, "smtp_port", 2525),
        ({"smtp_use_tls": False}, "smtp_use_tls", False),   # False is a real value, not "omitted"
        ({"smtp_username": "relay-user"}, "smtp_username", "relay-user"),
    ):
        r = client.patch(f"/mailboxes/{ref}", json=patch, headers=HDR_A)
        assert r.status_code == 200, r.text
        assert getattr(_row(sm, ref), attr) == expected
    after = _row(sm, ref)
    # ciphertext, identity and status never moved when no password / identity was sent
    assert after.smtp_password_enc == before.smtp_password_enc
    assert (after.email_address, after.organization_id, after.public_reference, after.status) == (
        before.email_address, before.organization_id, before.public_reference, before.status)


def test_activation_patch_is_one_atomic_update_of_status_metadata_and_password(client, sm):
    """The exact call LeadBoost makes on (re)verification success."""
    ref = _create(client)
    assert client.patch(f"/mailboxes/{ref}", json={"status": "disabled"}, headers=HDR_A).status_code == 200
    r = client.patch(
        f"/mailboxes/{ref}",
        json={"status": "active", "smtp_host": "smtp.rotated.example.com", "smtp_port": 465,
              "smtp_use_tls": True, "smtp_username": "rotated", "smtp_password": NEW_PW},
        headers=HDR_A,
    )
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "active" and out["smtp_host"] == "smtp.rotated.example.com"
    creds = get_mailbox_credentials(_row(sm, ref))     # what the dispatch worker will actually use
    assert (creds.smtp_host, creds.smtp_port, creds.smtp_username, creds.smtp_password) == (
        "smtp.rotated.example.com", 465, "rotated", NEW_PW)


def test_metadata_only_patch_does_not_need_the_encryption_key(client, sm, monkeypatch):
    ref = _create(client)
    monkeypatch.setattr(get_settings(), "mailbox_encryption_key", SecretStr(""))
    r = client.patch(f"/mailboxes/{ref}", json={"smtp_host": "smtp.keyless.example.com"}, headers=HDR_A)
    assert r.status_code == 200 and _row(sm, ref).smtp_host == "smtp.keyless.example.com"


def test_failed_password_encryption_leaves_metadata_untouched(client, sm, monkeypatch):
    ref = _create(client)
    before = _row(sm, ref)
    monkeypatch.setattr(get_settings(), "mailbox_encryption_key", SecretStr(""))
    r = client.patch(
        f"/mailboxes/{ref}",
        json={"status": "disabled", "smtp_host": "smtp.partial.example.com", "smtp_password": NEW_PW},
        headers=HDR_A,
    )
    assert r.status_code == 503
    after = _row(sm, ref)
    assert (after.smtp_host, after.status, after.smtp_password_enc) == (
        before.smtp_host, before.status, before.smtp_password_enc)


@pytest.mark.parametrize("body", [
    {"email_address": "other@example.com"}, {"organization_id": "org-b"}, {"public_reference": "x"},
    {"smtp_password_enc": "x"}, {"id": 1},
    {"imap_host": "imap.example.com"}, {"imap_port": 993}, {"imap_username": "u"},
])
def test_identity_and_imap_metadata_remain_immutable(client, sm, body):
    ref = _create(client)
    before = _row(sm, ref)
    assert client.patch(f"/mailboxes/{ref}", json=body, headers=HDR_A).status_code == 422
    after = _row(sm, ref)
    assert (after.email_address, after.organization_id, after.public_reference, after.imap_host) == (
        before.email_address, before.organization_id, before.public_reference, before.imap_host)


@pytest.mark.parametrize("body", [
    {"smtp_host": ""}, {"smtp_host": "   "}, {"smtp_host": None},
    {"smtp_port": 0}, {"smtp_port": 65536}, {"smtp_port": None}, {"smtp_port": "abc"},
    {"smtp_use_tls": None}, {"smtp_username": ""}, {"smtp_username": None},
])
def test_invalid_transport_values_are_rejected_and_change_nothing(client, sm, body):
    ref = _create(client)
    before = _row(sm, ref)
    assert client.patch(f"/mailboxes/{ref}", json=body, headers=HDR_A).status_code == 422
    after = _row(sm, ref)
    assert (after.smtp_host, after.smtp_port, after.smtp_use_tls, after.smtp_username) == (
        before.smtp_host, before.smtp_port, before.smtp_use_tls, before.smtp_username)


def test_other_organization_cannot_update_transport_and_gets_indistinguishable_404(client, sm):
    ref = _create(client, headers=HDR_A)
    before = _row(sm, ref)
    foreign = client.patch(f"/mailboxes/{ref}", json={"smtp_host": "evil.example.com"}, headers=HDR_B)
    unknown = client.patch("/mailboxes/does-not-exist", json={"smtp_host": "evil.example.com"}, headers=HDR_B)
    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json() == unknown.json()
    assert _row(sm, ref).smtp_host == before.smtp_host


def test_updated_responses_and_logs_never_contain_secrets(client, sm, caplog):
    ref = _create(client)
    caplog.set_level(logging.DEBUG)
    r = client.patch(f"/mailboxes/{ref}", json={"smtp_host": "smtp.x.example.com", "smtp_password": NEW_PW},
                     headers=HDR_A)
    assert r.status_code == 200
    blob = r.text + "\n".join(rec.getMessage() for rec in caplog.records)
    assert NEW_PW not in blob and SMTP_PW not in blob
    assert "smtp_password" not in r.json()


def test_422_on_transport_patch_does_not_echo_the_submitted_password(client):
    ref = _create(client)
    r = client.patch(f"/mailboxes/{ref}", json={"smtp_port": 0, "smtp_password": NEW_PW}, headers=HDR_A)
    assert r.status_code == 422 and NEW_PW not in r.text
