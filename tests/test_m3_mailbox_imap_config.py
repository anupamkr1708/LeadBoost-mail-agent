"""
M3 -- PATCH /mailboxes/{ref} accepts IMAP configuration as an all-or-none set.

Why: a mailbox provisioned by LeadBoost (L1) has no IMAP configuration, and
before M3 there was no way to add one, so it could never receive inbound mail.
Hard limits asserted here: the set is atomic, the password is write-only and
never returned/logged/stored in clear, organization comes only from the API
key, and a missing encryption key fails closed without touching the row.
"""

from __future__ import annotations

import logging

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, text
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
SMTP_PW = "SmtpSecret-M3-aa11"
IMAP_PW = "ImapSecret-M3-cc33"
NEW_IMAP_PW = "ImapSecret-M3-dd44"

IMAP_SET = {
    "imap_host": "imap.example.com", "imap_port": 993,
    "imap_username": "sales@example.com", "imap_password": IMAP_PW,
}


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
    eng = create_engine(f"sqlite:///{tmp_path/'m3.db'}", connect_args={"check_same_thread": False, "timeout": 15})
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


def test_full_set_gives_an_smtp_only_mailbox_an_imap_identity(client, sm, caplog):
    ref = _create(client)
    assert _row(sm, ref).imap_host is None
    with caplog.at_level(logging.DEBUG):
        r = client.patch(f"/mailboxes/{ref}", json=IMAP_SET, headers=HDR_A)
    assert r.status_code == 200, r.text
    out = r.json()
    assert (out["imap_host"], out["imap_port"], out["imap_username"]) == (
        "imap.example.com", 993, "sales@example.com")
    assert IMAP_PW not in r.text and "imap_password" not in out
    row = _row(sm, ref)
    assert row.imap_password_enc and IMAP_PW not in row.imap_password_enc
    assert get_mailbox_credentials(row).imap_password == IMAP_PW
    assert IMAP_PW not in caplog.text


def test_full_set_replaces_an_existing_imap_configuration_atomically(client, sm):
    ref = _create(client, **IMAP_SET)
    before = _row(sm, ref)
    r = client.patch(
        f"/mailboxes/{ref}",
        json={"imap_host": "imap.other.example", "imap_port": 143,
              "imap_username": "other-user", "imap_password": NEW_IMAP_PW},
        headers=HDR_A,
    )
    assert r.status_code == 200, r.text
    after = _row(sm, ref)
    assert (after.imap_host, after.imap_port, after.imap_username) == ("imap.other.example", 143, "other-user")
    assert after.imap_password_enc != before.imap_password_enc
    assert get_mailbox_credentials(after).imap_password == NEW_IMAP_PW
    # SMTP side untouched
    assert after.smtp_password_enc == before.smtp_password_enc


@pytest.mark.parametrize("missing", ["imap_host", "imap_port", "imap_username", "imap_password"])
def test_partial_set_is_rejected_and_changes_nothing(client, sm, missing):
    ref = _create(client)
    before = _row(sm, ref)
    body = {k: v for k, v in IMAP_SET.items() if k != missing}
    r = client.patch(f"/mailboxes/{ref}", json=body, headers=HDR_A)
    assert r.status_code == 422, r.text
    assert IMAP_PW not in r.text                      # 422 body never echoes the secret
    after = _row(sm, ref)
    assert (after.imap_host, after.imap_port, after.imap_username, after.imap_password_enc) == (
        before.imap_host, before.imap_port, before.imap_username, before.imap_password_enc)


def test_host_change_without_password_is_rejected_so_stored_secret_is_never_redirected(client, sm):
    ref = _create(client, **IMAP_SET)
    before = _row(sm, ref)
    r = client.patch(
        f"/mailboxes/{ref}",
        json={"imap_host": "attacker.example", "imap_port": 993, "imap_username": "sales@example.com"},
        headers=HDR_A,
    )
    assert r.status_code == 422
    assert _row(sm, ref).imap_host == before.imap_host


def test_password_alone_still_rotates_an_existing_imap_mailbox(client, sm):
    ref = _create(client, **IMAP_SET)
    before = _row(sm, ref)
    r = client.patch(f"/mailboxes/{ref}", json={"imap_password": NEW_IMAP_PW}, headers=HDR_A)
    assert r.status_code == 200, r.text
    after = _row(sm, ref)
    assert (after.imap_host, after.imap_port, after.imap_username) == (
        before.imap_host, before.imap_port, before.imap_username)
    assert get_mailbox_credentials(after).imap_password == NEW_IMAP_PW


def test_password_alone_on_a_mailbox_without_imap_is_still_409(client, sm):
    ref = _create(client)
    r = client.patch(f"/mailboxes/{ref}", json={"imap_password": IMAP_PW}, headers=HDR_A)
    assert r.status_code == 409
    assert _row(sm, ref).imap_password_enc is None


def test_other_organization_cannot_set_imap_on_my_mailbox(client, sm):
    ref = _create(client)
    r = client.patch(f"/mailboxes/{ref}", json=IMAP_SET, headers=HDR_B)
    assert r.status_code == 404
    assert _row(sm, ref).imap_host is None


def test_missing_encryption_key_fails_closed_and_leaves_the_row_untouched(client, sm, monkeypatch):
    ref = _create(client)
    monkeypatch.setattr(get_settings(), "mailbox_encryption_key", SecretStr(""))
    r = client.patch(f"/mailboxes/{ref}", json=IMAP_SET, headers=HDR_A)
    assert r.status_code == 503
    row = _row(sm, ref)
    assert row.imap_host is None and row.imap_password_enc is None
    with sm().bind.connect() as c:
        assert IMAP_PW not in str(c.execute(text("SELECT * FROM mailboxes")).all())
