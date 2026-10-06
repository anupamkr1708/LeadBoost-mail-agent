"""
M1 -- Mailer-owned Mailbox foundation (SQLite, deterministic).

Real API-key dependencies run (only get_db is overridden). Race behaviour
against real PostgreSQL is in test_mailbox_postgres.py.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from mailer_agent import mailbox_secrets
from mailer_agent.api import deps
from mailer_agent.api.main import app
from mailer_agent.config import get_settings
from mailer_agent.db import get_db
from mailer_agent.mailbox_secrets import (
    MailboxDecryptionError,
    MailboxEncryptionUnavailable,
    decrypt_secret,
    encrypt_secret,
    get_mailbox_credentials,
)
from mailer_agent.models import Base, ExternalDispatch, Mailbox, MailboxStatus
from mailer_agent.schemas import MailboxCreate

ORG_A, ORG_B = "org-a", "org-b"
HDR_A, HDR_B = {"X-API-Key": "key-a"}, {"X-API-Key": "key-b"}
KEY = Fernet.generate_key().decode()
SMTP_PW = "SmtpSecret-7c1e9d"
IMAP_PW = "ImapSecret-4b8a2f"
NEW_SMTP_PW = "NewSmtpSecret-31ab"
NEW_IMAP_PW = "NewImapSecret-77cd"
OUT_FIELDS = {
    "public_reference", "email_address", "smtp_host", "smtp_port", "smtp_use_tls", "smtp_username",
    "imap_host", "imap_port", "imap_username", "status", "created_at", "updated_at",
}


def _body(**over):
    base = {
        "email_address": "sales@example.com",
        "smtp_host": "smtp.example.com", "smtp_port": 587, "smtp_use_tls": True,
        "smtp_username": "sales@example.com", "smtp_password": SMTP_PW,
        "imap_host": "imap.example.com", "imap_port": 993,
        "imap_username": "sales@example.com", "imap_password": IMAP_PW,
    }
    base.update(over)
    return {k: v for k, v in base.items() if v is not None}


NO_IMAP = {"imap_host": None, "imap_port": None, "imap_username": None, "imap_password": None}


@pytest.fixture()
def sm(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path/'mbx.db'}", connect_args={"check_same_thread": False, "timeout": 15})
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


def _row(sm, ref):
    s = sm()
    try:
        m = s.query(Mailbox).filter_by(public_reference=ref).one()
        s.expunge(m)
        return m
    finally:
        s.close()


def _create(client, headers=HDR_A, **over):
    r = client.post("/mailboxes", json=_body(**over), headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


# --------------------------------------------------------------- persistence

def test_create_persists_and_survives_reload(client, sm):
    out = _create(client, email_address="  Sales@Example.COM ")
    assert set(out) == OUT_FIELDS
    assert out["email_address"] == "sales@example.com"
    assert out["status"] == "active"
    row = _row(sm, out["public_reference"])
    assert (row.organization_id, row.email_address, row.status) == (ORG_A, "sales@example.com", "active")
    assert (row.smtp_host, row.smtp_port, row.smtp_use_tls) == ("smtp.example.com", 587, True)
    assert len(out["public_reference"]) == 32 and row.created_at and row.updated_at
    assert client.get(f"/mailboxes/{out['public_reference']}", headers=HDR_A).json() == out


def test_smtp_only_mailbox_is_valid(client, sm):
    out = _create(client, **NO_IMAP)
    row = _row(sm, out["public_reference"])
    assert out["imap_host"] is None and row.imap_password_enc is None


# ----------------------------------------------------------------- ownership

def test_org_comes_from_key_only_and_reads_are_tenant_scoped(client, sm):
    a = _create(client, HDR_A, email_address="a@example.com")
    b = _create(client, HDR_B, email_address="b@example.com")
    assert _row(sm, a["public_reference"]).organization_id == ORG_A
    assert _row(sm, b["public_reference"]).organization_id == ORG_B
    assert [m["email_address"] for m in client.get("/mailboxes", headers=HDR_A).json()] == ["a@example.com"]
    assert [m["email_address"] for m in client.get("/mailboxes", headers=HDR_B).json()] == ["b@example.com"]
    assert client.get(f"/mailboxes/{a['public_reference']}", headers=HDR_A).status_code == 200


def test_foreign_and_unknown_reference_are_indistinguishable_404(client):
    a = _create(client, HDR_A)
    foreign = client.get(f"/mailboxes/{a['public_reference']}", headers=HDR_B)
    unknown = client.get("/mailboxes/" + "0" * 32, headers=HDR_B)
    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json() == unknown.json()
    pf = client.patch(f"/mailboxes/{a['public_reference']}", json={"status": "disabled"}, headers=HDR_B)
    pu = client.patch("/mailboxes/" + "0" * 32, json={"status": "disabled"}, headers=HDR_B)
    assert pf.status_code == pu.status_code == 404 and pf.json() == pu.json()
    assert client.get(f"/mailboxes/{a['public_reference']}", headers=HDR_A).json()["status"] == "active"


# ---------------------------------------------------------------- duplicates

def test_duplicate_in_same_org_is_409_and_leaves_one_row(client, sm):
    _create(client, email_address="dup@example.com")
    r = client.post("/mailboxes", json=_body(email_address=" DUP@example.com "), headers=HDR_A)
    assert r.status_code == 409
    s = sm()
    assert s.query(Mailbox).count() == 1
    s.close()


def test_same_address_in_different_orgs_is_allowed(client):
    _create(client, HDR_A, email_address="shared@example.com")
    _create(client, HDR_B, email_address="shared@example.com")   # documented M3 consideration


def test_db_unique_constraints_exist_and_enforce(sm):
    insp = inspect(sm.kw["bind"])
    uniques = {u["name"]: tuple(u["column_names"]) for u in insp.get_unique_constraints("mailboxes")}
    assert uniques["uq_mailboxes_org_email"] == ("organization_id", "email_address")
    assert uniques["uq_mailboxes_public_reference"] == ("public_reference",)

    def mk(ref="r1"):
        return Mailbox(public_reference=ref, organization_id=ORG_A, email_address="x@example.com",
                       smtp_host="h", smtp_port=25, smtp_use_tls=False, smtp_username="u", smtp_password_enc="t")

    s1, s2 = sm(), sm()
    s1.add(mk("r1")); s1.commit()
    s2.add(mk("r2"))
    with pytest.raises(IntegrityError):
        s2.commit()
    s2.rollback()
    assert s1.query(Mailbox).count() == 1
    s1.close(); s2.close()


# ---------------------------------------------------------------- encryption

def test_secrets_are_not_stored_in_plaintext(client, sm):
    out = _create(client)
    with sm.kw["bind"].connect() as c:
        row = c.execute(text("SELECT * FROM mailboxes")).mappings().one()
    stored = " ".join(str(v) for v in row.values())
    assert SMTP_PW not in stored and IMAP_PW not in stored
    assert row["smtp_password_enc"] != SMTP_PW and row["imap_password_enc"] != IMAP_PW
    m = _row(sm, out["public_reference"])
    creds = get_mailbox_credentials(m)
    assert (creds.smtp_password, creds.imap_password) == (SMTP_PW, IMAP_PW)
    assert creds.email_address == "sales@example.com" and creds.smtp_port == 587


def test_wrong_key_cannot_decrypt(client, sm, monkeypatch):
    m = _row(sm, _create(client)["public_reference"])
    monkeypatch.setattr(get_settings(), "mailbox_encryption_key", SecretStr(Fernet.generate_key().decode()))
    with pytest.raises(MailboxDecryptionError):
        get_mailbox_credentials(m)


@pytest.mark.parametrize("bad", ["", "   ", "not-a-fernet-key"])
def test_missing_or_invalid_key_blocks_create_with_503(client, sm, monkeypatch, bad):
    monkeypatch.setattr(get_settings(), "mailbox_encryption_key", SecretStr(bad))
    r = client.post("/mailboxes", json=_body(), headers=HDR_A)
    assert r.status_code == 503
    assert SMTP_PW not in r.text and "not-a-fernet-key" not in r.text
    s = sm()
    assert s.query(Mailbox).count() == 0
    s.close()
    assert get_settings().mailbox_encryption_key.get_secret_value() == bad   # nothing generated


def test_no_fallback_key_exists(monkeypatch):
    monkeypatch.setattr(get_settings(), "mailbox_encryption_key", SecretStr(""))
    with pytest.raises(MailboxEncryptionUnavailable):
        encrypt_secret("x")
    with pytest.raises(MailboxEncryptionUnavailable):
        decrypt_secret("x")
    assert Settings_default_key_is_empty()


def Settings_default_key_is_empty() -> bool:
    from mailer_agent.config import Settings
    return Settings.model_fields["mailbox_encryption_key"].default.get_secret_value() == ""


def test_reads_do_not_need_the_key(client, monkeypatch):
    ref = _create(client)["public_reference"]
    monkeypatch.setattr(get_settings(), "mailbox_encryption_key", SecretStr(""))
    assert client.get("/mailboxes", headers=HDR_A).status_code == 200
    assert client.get(f"/mailboxes/{ref}", headers=HDR_A).status_code == 200
    assert client.patch(f"/mailboxes/{ref}", json={"status": "disabled"}, headers=HDR_A).status_code == 200


# ------------------------------------------------------- secret non-disclosure

def test_secrets_never_appear_in_responses_errors_logs_or_reprs(client, sm, caplog):
    caplog.set_level(logging.DEBUG)
    created = client.post("/mailboxes", json=_body(), headers=HDR_A)
    ref = created.json()["public_reference"]
    with sm.kw["bind"].connect() as c:
        row = c.execute(text("SELECT smtp_password_enc, imap_password_enc FROM mailboxes")).one()
    secrets_ = [SMTP_PW, IMAP_PW, NEW_SMTP_PW, NEW_IMAP_PW, KEY, row[0], row[1]]

    responses = [
        created,
        client.get("/mailboxes", headers=HDR_A),
        client.get(f"/mailboxes/{ref}", headers=HDR_A),
        client.patch(f"/mailboxes/{ref}", json={"smtp_password": NEW_SMTP_PW, "imap_password": NEW_IMAP_PW}, headers=HDR_A),
        # 409
        client.post("/mailboxes", json=_body(), headers=HDR_A),
        # 404s carrying secrets
        client.patch("/mailboxes/" + "0" * 32, json={"smtp_password": NEW_SMTP_PW}, headers=HDR_A),
        client.patch(f"/mailboxes/{ref}", json={"smtp_password": NEW_SMTP_PW}, headers=HDR_B),
        # 422s carrying secrets
        client.post("/mailboxes", json=_body(smtp_port=0), headers=HDR_A),
        client.post("/mailboxes", json=_body(organization_id="x"), headers=HDR_A),
        client.post("/mailboxes", json=_body(smtp_password_enc="x"), headers=HDR_A),
        client.post("/mailboxes", json=_body(email_address="not-an-email"), headers=HDR_A),
        client.post("/mailboxes", json=_body(imap_host=None), headers=HDR_A),
        client.post("/mailboxes", json=_body(smtp_password=""), headers=HDR_A),
        client.post("/mailboxes", json=_body(smtp_use_tls="maybe"), headers=HDR_A),
        client.post("/mailboxes", json=_body(smtp_password=12345), headers=HDR_A),
        client.patch(f"/mailboxes/{ref}", json={"smtp_password": NEW_SMTP_PW, "status": "bogus"}, headers=HDR_A),
        client.patch(f"/mailboxes/{ref}", json={"smtp_password": NEW_SMTP_PW, "email_address": "z@example.com"}, headers=HDR_A),
        client.post("/mailboxes", content='{"smtp_password": "' + SMTP_PW + '", broken', headers={**HDR_A, "content-type": "application/json"}),
    ]
    assert [r.status_code for r in responses[:4]] == [201, 200, 200, 200]
    assert {r.status_code for r in responses[7:]} == {422}
    for r in responses:
        for s in secrets_:
            assert s not in r.text, (r.status_code, r.text)
    for r in responses[7:]:
        for err in r.json()["detail"]:
            assert set(err) == {"type", "loc", "msg"}      # no `input`, `ctx`, `url`

    log_text = "\n".join(rec.getMessage() + str(rec.__dict__.get("args", "")) for rec in caplog.records)
    for s in secrets_:
        assert s not in log_text

    m = _row(sm, ref)
    creds = get_mailbox_credentials(m)
    schema = MailboxCreate(**_body())
    for obj in (m, creds, schema):
        for rendered in (repr(obj), str(obj)):
            for s in secrets_:
                assert s not in rendered
    assert KEY not in repr(get_settings())


def test_default_fastapi_422_echoes_input_but_mailbox_422_does_not(client):
    # Control: proves the redaction is doing real work on this router only.
    r = client.post("/campaigns", json={"name": SMTP_PW}, headers=HDR_A)
    assert r.status_code == 422 and "input" in r.json()["detail"][0]
    m = client.post("/mailboxes", json=_body(smtp_port="abc"), headers=HDR_A)
    assert m.status_code == 422 and all("input" not in e for e in m.json()["detail"])


# ----------------------------------------------------------------- replacement

def test_patch_replaces_smtp_and_imap_passwords(client, sm):
    ref = _create(client)["public_reference"]
    before = _row(sm, ref)
    r = client.patch(f"/mailboxes/{ref}", json={"smtp_password": NEW_SMTP_PW, "imap_password": NEW_IMAP_PW}, headers=HDR_A)
    assert r.status_code == 200 and set(r.json()) == OUT_FIELDS
    after = _row(sm, ref)
    assert after.smtp_password_enc != before.smtp_password_enc
    assert after.imap_password_enc != before.imap_password_enc
    creds = get_mailbox_credentials(after)
    assert (creds.smtp_password, creds.imap_password) == (NEW_SMTP_PW, NEW_IMAP_PW)
    assert (after.id, after.public_reference, after.organization_id, after.email_address) == (
        before.id, before.public_reference, before.organization_id, before.email_address)


def test_patch_replaces_only_what_is_provided(client, sm):
    ref = _create(client)["public_reference"]
    before = _row(sm, ref)
    client.patch(f"/mailboxes/{ref}", json={"smtp_password": NEW_SMTP_PW}, headers=HDR_A)
    mid = _row(sm, ref)
    assert mid.imap_password_enc == before.imap_password_enc          # untouched, not re-encrypted
    assert get_mailbox_credentials(mid).imap_password == IMAP_PW
    client.patch(f"/mailboxes/{ref}", json={"status": "disabled"}, headers=HDR_A)
    after = _row(sm, ref)
    assert after.smtp_password_enc == mid.smtp_password_enc and after.imap_password_enc == mid.imap_password_enc


def test_invalid_key_prevents_replacement_and_changes_nothing(client, sm, monkeypatch):
    ref = _create(client)["public_reference"]
    before = _row(sm, ref)
    monkeypatch.setattr(get_settings(), "mailbox_encryption_key", SecretStr("garbage"))
    r = client.patch(f"/mailboxes/{ref}", json={"status": "disabled", "smtp_password": NEW_SMTP_PW}, headers=HDR_A)
    assert r.status_code == 503
    after = _row(sm, ref)
    assert (after.status, after.smtp_password_enc, after.imap_password_enc) == (
        "active", before.smtp_password_enc, before.imap_password_enc)


def test_imap_password_on_smtp_only_mailbox_is_rejected(client, sm):
    ref = _create(client, **NO_IMAP)["public_reference"]
    r = client.patch(f"/mailboxes/{ref}", json={"imap_password": NEW_IMAP_PW}, headers=HDR_A)
    assert r.status_code == 409
    assert _row(sm, ref).imap_password_enc is None


@pytest.mark.parametrize("body", [
    {"smtp_password": ""}, {"imap_password": ""}, {"status": None}, {"smtp_password": None},
    {"email_address": "z@example.com"}, {"organization_id": "x"}, {"public_reference": "x"},
    {"smtp_password_enc": "x"},
    # L1 revision: smtp_host/port/use_tls/username became updatable (see
    # tests/test_l1_mailbox_transport_update.py). IMAP metadata still is not.
    {"imap_host": "evil.example.com"}, {"imap_port": 993}, {"imap_username": "x"},
])
def test_patch_schema_rejects_empty_null_and_immutable_fields(client, body):
    ref = _create(client)["public_reference"]
    assert client.patch(f"/mailboxes/{ref}", json=body, headers=HDR_A).status_code == 422


# ---------------------------------------------------------------------- status

def test_status_values_and_identity(client, sm):
    out = _create(client)
    ref = out["public_reference"]
    for value in ("disabled", "active"):
        r = client.patch(f"/mailboxes/{ref}", json={"status": value}, headers=HDR_A)
        assert r.status_code == 200 and r.json()["status"] == value
    assert client.patch(f"/mailboxes/{ref}", json={"status": "paused"}, headers=HDR_A).status_code == 422
    final = client.get(f"/mailboxes/{ref}", headers=HDR_A).json()
    assert final["status"] == "active"
    assert {k: final[k] for k in ("public_reference", "email_address", "smtp_host")} == {
        k: out[k] for k in ("public_reference", "email_address", "smtp_host")}
    assert _row(sm, ref).organization_id == ORG_A
    assert {s.value for s in MailboxStatus} == {"active", "disabled"}


# ------------------------------------------------------------------------ auth

@pytest.mark.parametrize("method,path,kw", [
    ("post", "/mailboxes", {"json": _body()}),
    ("get", "/mailboxes", {}),
    ("get", "/mailboxes/" + "0" * 32, {}),
    ("patch", "/mailboxes/" + "0" * 32, {"json": {"status": "disabled"}}),
])
def test_auth_fails_closed(sm, monkeypatch, method, path, kw):
    monkeypatch.setattr(get_settings(), "mailbox_encryption_key", SecretStr(KEY))

    def _get_db():
        s = sm()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    try:
        c = TestClient(app)
        monkeypatch.setattr(deps, "_KEY_MAP", {})
        assert getattr(c, method)(path, **kw).status_code == 503                    # no keys configured
        assert getattr(c, method)(path, headers=HDR_A, **kw).status_code == 503
        monkeypatch.setattr(deps, "_KEY_MAP", {"key-a": ORG_A})
        assert getattr(c, method)(path, **kw).status_code == 401                    # missing
        assert getattr(c, method)(path, headers={"X-API-Key": "nope"}, **kw).status_code == 401   # invalid
    finally:
        app.dependency_overrides.clear()


def test_valid_key_is_accepted_and_auth_runs_before_body_validation(client):
    assert client.get("/mailboxes", headers=HDR_A).status_code == 200
    assert client.post("/mailboxes", json={"junk": 1}, headers={"X-API-Key": "nope"}).status_code == 401


# ----------------------------------------------------------------- schema rules

@pytest.mark.parametrize("extra", [
    {"organization_id": "org-b"}, {"public_reference": "abc"}, {"smtp_password_enc": "x"},
    {"imap_password_enc": "x"}, {"status": "active"}, {"id": 1}, {"provider": "gmail"},
    {"imap": {"host": "h", "unknown": 1}},     # nested object under an undeclared key
])
def test_create_rejects_unknown_and_forbidden_fields(client, sm, extra):
    assert client.post("/mailboxes", json=_body(**extra), headers=HDR_A).status_code == 422
    s = sm()
    assert s.query(Mailbox).count() == 0
    s.close()


@pytest.mark.parametrize("missing", [
    ("imap_host",), ("imap_port",), ("imap_username",), ("imap_password",),
    ("imap_host", "imap_port"), ("imap_host", "imap_port", "imap_username"),
])
def test_partial_imap_is_rejected(client, missing):
    body = _body()
    for k in missing:
        body.pop(k)
    assert client.post("/mailboxes", json=body, headers=HDR_A).status_code == 422


@pytest.mark.parametrize("field", ["email_address", "smtp_host", "smtp_port", "smtp_use_tls", "smtp_username", "smtp_password"])
def test_required_fields(client, field):
    body = _body()
    body.pop(field)
    assert client.post("/mailboxes", json=body, headers=HDR_A).status_code == 422


def test_schema_model_rejects_extra_directly():
    with pytest.raises(ValidationError):
        MailboxCreate(**_body(organization_id="x"))


# --------------------------------------------------------------- migration 006

def _load_migration():
    path = Path(__file__).parent.parent / "migrations" / "006_mailboxes.py"
    spec = importlib.util.spec_from_file_location("migration_006", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_006_upgrade_idempotent_downgrade_on_sqlite(tmp_path, monkeypatch):
    from contextlib import contextmanager

    eng = create_engine(f"sqlite:///{tmp_path/'mig.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng, tables=[t for t in Base.metadata.sorted_tables if t.name != "mailboxes"])
    Sess = sessionmaker(bind=eng)

    @contextmanager
    def scope():
        s = Sess()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    mig = _load_migration()
    monkeypatch.setattr(mig, "session_scope", scope)
    existing = set(inspect(eng).get_table_names())
    assert "mailboxes" not in existing

    assert mig.run() is True and mig.run() is True                       # idempotent
    assert existing <= set(inspect(eng).get_table_names()) and "mailboxes" in inspect(eng).get_table_names()
    assert {c["name"] for c in inspect(eng).get_columns("mailboxes")} >= {"public_reference", "smtp_password_enc", "imap_password_enc"}

    # coexists with create_all() having already created the table
    Base.metadata.create_all(bind=eng)
    assert mig.run() is True

    assert mig.downgrade() is True and "mailboxes" not in inspect(eng).get_table_names()
    assert mig.downgrade() is True                                       # no-op
    assert existing <= set(inspect(eng).get_table_names())               # nothing else touched
    assert mig.run() is True and "mailboxes" in inspect(eng).get_table_names()


# ----------------------------------------------------------------- compatibility

def test_existing_schema_and_campaign_flow_are_untouched(client, sm):
    # M2-A: dispatches reference their mailbox (nullable; credentials never copied onto the row).
    assert "mailbox_id" in ExternalDispatch.__table__.c
    assert ExternalDispatch.__table__.c.mailbox_id.nullable is True
    assert not {"smtp_password", "smtp_password_enc"} & set(ExternalDispatch.__table__.c.keys())
    cols = inspect(sm.kw["bind"]).get_columns("campaigns")
    assert {"sender_name", "sender_org", "sender_email", "reply_to_email"} <= {c["name"] for c in cols}
    _create(client)
    camp = client.post("/campaigns", headers=HDR_A, json={
        "name": "c1", "sender_name": "S", "sender_org": "O", "sender_email": "s@example.com",
        "value_prop": "We help.",
    })
    assert camp.status_code == 201 and camp.json()["name"] == "c1"
    assert client.get("/health").status_code == 200


def test_c92_route_still_registered_and_fail_closed(sm, monkeypatch):
    monkeypatch.setattr(deps, "_KEY_MAP", {})
    r = TestClient(app).post("/integrations/leadboost/outreach-requests", json={})
    assert r.status_code == 503
