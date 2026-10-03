"""
M2-A -- Mailer-owned mailbox in the durable execution path (SQLite, offline).

Intake (both LeadBoost routes): the organization must have exactly one ACTIVE
mailbox -> its id is stored on the dispatch; zero or several -> 409 with
nothing created.

Worker: the mailbox is re-validated at claim time (present, same org, ACTIVE),
its SMTP password is decrypted per operation, and send_email() is handed that
transport. Every failure here is a pre-SMTP FAILED, except an unusable
deployment key, which leaves the row QUEUED.
"""

from __future__ import annotations

import logging
import smtplib

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailer_agent.api import deps
from mailer_agent.api.deps import get_current_org_id, get_integration_org_id, require_api_key
from mailer_agent.api.main import app
from mailer_agent.config import get_settings
from mailer_agent.db import get_db
from mailer_agent.mail import external_dispatch_worker as w
from mailer_agent.mail import sender
from mailer_agent.mail.sender import SendOutcome, SmtpConfig
from mailer_agent.mailbox_secrets import encrypt_secret
from mailer_agent.models import (
    Base,
    Campaign,
    Contact,
    ExternalDispatch,
    ExternalDispatchState as S,
    Mailbox,
    MailboxStatus,
    Message,
    MessageStatus,
)
from tests.dispatch_support import (
    MAILBOX_SMTP_PASSWORD,
    FakeSender,
    TrackingFactory,
    seed_dispatch,
    seed_mailbox,
)

ORG_A, ORG_B = "org-a", "org-b"
EXACT_URL = "/integrations/leadboost/outreach-actions"
GEN_URL = "/integrations/leadboost/outreach-requests"
CLEAN_BODY = "Hi Jane,\n\nWorth a quick chat?\n\nBest"


# ------------------------------------------------------------------ fixtures

@pytest.fixture(autouse=True)
def _sender_identity(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "leadboost_integration_sender_email", "outreach@mailer.example.com")
    monkeypatch.setattr(s, "leadboost_integration_sender_name", "Test Sender")
    monkeypatch.setattr(s, "leadboost_integration_sender_org", "Test Org")


@pytest.fixture()
def sm(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path/'m2a.db'}", connect_args={"check_same_thread": False, "timeout": 15})
    Base.metadata.create_all(bind=eng)
    yield sessionmaker(bind=eng)
    eng.dispose()


@pytest.fixture()
def factory(sm):
    return TrackingFactory(sm)


@pytest.fixture()
def client(sm):
    def _get_db():
        s = sm()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_integration_org_id] = lambda: ORG_A
    app.dependency_overrides[get_current_org_id] = lambda: ORG_A
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def live(monkeypatch):
    monkeypatch.setattr(w.settings, "live_sending_enabled", True)


@pytest.fixture()
def spy(monkeypatch, live):
    f = FakeSender()
    monkeypatch.setattr(w, "send_email", f)
    return f


def _exact(key="idem-1", email="lead@example.com"):
    return {
        "external_action_id": "481",
        "idempotency_key": key,
        "recipient": {"email": email, "name": "Jane"},
        "message": {"subject": "Hi", "body": CLEAN_BODY},
    }


def _gen(key="idem-g1"):
    return {
        "external_action_id": "482",
        "idempotency_key": key,
        "recipient": {"email": "jane@acme.example.com", "name": "Jane Doe"},
        "context": {"value_proposition": "We help finance teams close faster.", "recipient_facts": []},
    }


def _mbx(sm, org=ORG_A, email="sales@acme-sender.example", **kw) -> int:
    with sm() as s:
        m = seed_mailbox(s, org=org, email=email, **kw)
        s.commit()
        return m.id


def _counts(sm):
    with sm() as s:
        return tuple(
            s.query(t).count() for t in (Campaign, Contact, Message, ExternalDispatch)
        )


def _run(factory, worker="w-m2a"):
    return w.process_next_external_dispatch(session_factory=factory, worker_id=worker, runtime=w.DispatchRuntime())


def _seed(factory, **kw):
    with factory.sm() as s:
        return seed_dispatch(s, **kw).id


def _row(factory, did):
    with factory.sm() as s:
        d = s.get(ExternalDispatch, did)
        m = s.get(Message, d.message_id)
        return d.state, d.error_message, d.claimed_by, m.status, m.message_id_header


# ------------------------------------------------- intake: mailbox resolution

@pytest.mark.parametrize("url,body", [(EXACT_URL, _exact()), (GEN_URL, _gen())], ids=["exact", "generated"])
def test_no_active_mailbox_is_409_and_creates_nothing(client, sm, fake_llm, url, body):
    assert client.post(url, json=body).status_code == 409
    assert _counts(sm) == (0, 0, 0, 0)
    assert fake_llm.call_count == 0                       # generated route: no LLM call either


@pytest.mark.parametrize("url,body", [(EXACT_URL, _exact()), (GEN_URL, _gen())], ids=["exact", "generated"])
def test_only_a_disabled_mailbox_counts_as_none(client, sm, fake_llm, url, body):
    _mbx(sm, status=MailboxStatus.DISABLED.value)
    assert client.post(url, json=body).status_code == 409
    assert _counts(sm) == (0, 0, 0, 0) and fake_llm.call_count == 0


@pytest.mark.parametrize("url,body", [(EXACT_URL, _exact()), (GEN_URL, _gen())], ids=["exact", "generated"])
def test_more_than_one_active_mailbox_is_409_and_creates_nothing(client, sm, fake_llm, url, body):
    _mbx(sm, email="a@acme-sender.example")
    _mbx(sm, email="b@acme-sender.example")
    assert client.post(url, json=body).status_code == 409
    assert _counts(sm) == (0, 0, 0, 0) and fake_llm.call_count == 0


@pytest.mark.parametrize("url,body", [(EXACT_URL, _exact()), (GEN_URL, _gen())], ids=["exact", "generated"])
def test_another_orgs_mailbox_is_never_used(client, sm, fake_llm, url, body):
    _mbx(sm, org=ORG_B)                                   # caller is org-a
    assert client.post(url, json=body).status_code == 409
    assert _counts(sm) == (0, 0, 0, 0) and fake_llm.call_count == 0


def test_exact_route_stores_the_orgs_sole_active_mailbox(client, sm):
    _mbx(sm, status=MailboxStatus.DISABLED.value, email="old@acme-sender.example")   # ignored
    active = _mbx(sm, email="live@acme-sender.example")
    _mbx(sm, org=ORG_B, email="other@b.example")
    assert client.post(EXACT_URL, json=_exact()).status_code == 202
    with sm() as s:
        assert s.query(ExternalDispatch).one().mailbox_id == active


def test_generated_route_stores_the_orgs_sole_active_mailbox(client, sm, fake_llm):
    active = _mbx(sm)
    fake_llm.queue_response({"subject": "Quick question", "body": CLEAN_BODY, "reasoning": "t"})
    assert client.post(GEN_URL, json=_gen()).status_code == 202
    with sm() as s:
        assert s.query(ExternalDispatch).one().mailbox_id == active


def test_replay_of_an_accepted_key_does_not_need_a_mailbox(client, sm):
    mid = _mbx(sm)
    first = client.post(EXACT_URL, json=_exact())
    assert first.status_code == 202
    with sm() as s:                                       # mailbox disabled after acceptance
        s.get(Mailbox, mid).status = MailboxStatus.DISABLED.value
        s.commit()
    again = client.post(EXACT_URL, json=_exact())
    assert again.status_code == 202 and again.json() == first.json()
    assert _counts(sm) == (1, 1, 1, 1)


def test_request_contract_stays_closed_to_mailbox_selection(client, sm, fake_llm):
    _mbx(sm)
    for field in ("mailbox_id", "mailbox_reference"):
        assert client.post(GEN_URL, json={**_gen(), field: "x"}).status_code == 422
    assert fake_llm.call_count == 0 and _counts(sm) == (0, 0, 0, 0)


# ------------------------------------------------- worker: mailbox execution

def test_send_uses_the_mailbox_transport_and_identity(factory, spy, caplog):
    caplog.set_level(logging.DEBUG)
    did = _seed(factory, smtp_host="smtp.mbx.example", smtp_port=2587)
    res = _run(factory)
    assert res.outcome == "sent" and len(spy.calls) == 1
    call = spy.calls[0]
    cfg = call["smtp_config"]
    assert cfg == SmtpConfig(
        host="smtp.mbx.example", port=2587, use_tls=False,
        username="outreach@sender.example.org", password=MAILBOX_SMTP_PASSWORD,
    )
    assert call["from_email"] == "outreach@sender.example.org"      # the mailbox address
    assert call["message_id_header"].endswith("@sender.example.org>")
    # the secret exists only in memory: not in repr, logs, or any stored row
    assert MAILBOX_SMTP_PASSWORD not in repr(cfg)
    assert MAILBOX_SMTP_PASSWORD not in caplog.text
    state, error, *_ = _row(factory, did)
    assert state == S.SENT.value and error is None


def test_from_address_is_the_mailbox_not_the_deployment_campaign_sender(factory, spy):
    _seed(factory)
    with factory.sm() as s:
        c = s.query(Campaign).one()
        assert c.sender_email == "outreach@sender.example.org"
        c.sender_email = "legacy-global@deploy.example.com"
        s.commit()
    _run(factory)
    assert spy.calls[0]["from_email"] == "outreach@sender.example.org"


def test_null_mailbox_id_fails_before_smtp_as_no_mailbox(factory, spy):
    did = _seed(factory, mailbox=False)
    res = _run(factory)
    state, error, claimed_by, msg_status, msg_id = _row(factory, did)
    assert res.outcome == "failed" and res.smtp_attempted is False and spy.calls == []
    assert state == S.FAILED.value and error.startswith("no_mailbox")
    assert (claimed_by, msg_status, msg_id) == (None, MessageStatus.FAILED.value, None)


def test_mailbox_disabled_after_acceptance_fails_before_smtp(factory, spy):
    did = _seed(factory)
    with factory.sm() as s:
        s.query(Mailbox).one().status = MailboxStatus.DISABLED.value
        s.commit()
    res = _run(factory)
    state, error, *_ = _row(factory, did)
    assert res.outcome == "failed" and spy.calls == []
    assert state == S.FAILED.value and error.startswith("mailbox_disabled")


def test_mailbox_of_another_org_fails_as_tenant_mismatch(factory, spy):
    did = _seed(factory)
    with factory.sm() as s:
        foreign = seed_mailbox(s, org=ORG_B, email="b@other.example")
        s.get(ExternalDispatch, did).mailbox_id = foreign.id
        s.commit()
    res = _run(factory)
    state, error, *_ = _row(factory, did)
    assert res.outcome == "failed" and spy.calls == []
    assert state == S.FAILED.value and error.startswith("tenant_mismatch")


def test_corrupt_ciphertext_fails_before_smtp_without_leaking_it(factory, spy):
    did = _seed(factory)
    with factory.sm() as s:
        s.query(Mailbox).one().smtp_password_enc = "not-a-fernet-token"
        s.commit()
    res = _run(factory)
    state, error, _, msg_status, msg_id = _row(factory, did)
    assert res.outcome == "failed" and spy.calls == []
    assert state == S.FAILED.value and error.startswith("mailbox_credentials_undecryptable")
    assert "not-a-fernet-token" not in error and (msg_status, msg_id) == (MessageStatus.FAILED.value, None)


def test_credential_encrypted_under_a_different_key_fails_before_smtp(factory, spy):
    did = _seed(factory)
    with factory.sm() as s:
        s.query(Mailbox).one().smtp_password_enc = Fernet(Fernet.generate_key()).encrypt(b"x").decode()
        s.commit()
    assert _run(factory).outcome == "failed" and spy.calls == []
    assert _row(factory, did)[1].startswith("mailbox_credentials_undecryptable")


@pytest.mark.parametrize("bad_key", ["", "not-a-valid-fernet-key"], ids=["unset", "invalid"])
def test_unusable_deployment_key_leaves_the_row_queued_not_failed(factory, spy, monkeypatch, bad_key):
    did = _seed(factory)
    good = get_settings().mailbox_encryption_key
    monkeypatch.setattr(get_settings(), "mailbox_encryption_key", SecretStr(bad_key))
    assert _run(factory) is None and spy.calls == []
    state, error, claimed_by, msg_status, msg_id = _row(factory, did)
    assert (state, error, claimed_by, msg_status, msg_id) == (S.QUEUED.value, None, None, MessageStatus.DRAFT.value, None)
    monkeypatch.setattr(get_settings(), "mailbox_encryption_key", good)           # deployment fixed
    assert _run(factory).outcome == "sent" and _row(factory, did)[0] == S.SENT.value


def test_rotated_credential_is_used_by_the_next_send(factory, spy):
    _seed(factory, idem="d1")
    _run(factory)
    with factory.sm() as s:
        s.query(Mailbox).one().smtp_password_enc = encrypt_secret("rotated-secret-1b2c")
        s.commit()
    _seed(factory, idem="d2", email="second@example.com")
    _run(factory)
    assert [c["smtp_config"].password for c in spy.calls] == [MAILBOX_SMTP_PASSWORD, "rotated-secret-1b2c"]


def test_disabling_the_mailbox_does_not_interrupt_an_in_flight_send(factory, spy):
    """ACTIVE is evaluated at claim time. Once SENDING is committed, the
    outcome is recorded honestly: the message went out."""
    did = _seed(factory)

    def disable_during_smtp(_kw):
        with factory.sm() as s:
            s.query(Mailbox).one().status = MailboxStatus.DISABLED.value
            s.commit()

    spy.on_call = disable_during_smtp
    assert _run(factory).outcome == "sent" and _row(factory, did)[0] == S.SENT.value


@pytest.mark.parametrize("outcome,state", [
    (SendOutcome.FAILED, S.FAILED), (SendOutcome.UNKNOWN, S.UNKNOWN),
])
def test_smtp_outcomes_map_exactly_as_before_with_a_mailbox(factory, monkeypatch, live, outcome, state):
    f = FakeSender(outcome=outcome, error="smtp said no")
    monkeypatch.setattr(w, "send_email", f)
    did = _seed(factory)
    _run(factory)
    assert _row(factory, did)[0] == state.value and len(f.calls) == 1


def test_no_session_or_transaction_is_open_while_smtp_runs(factory, spy):
    seen = []
    spy.on_call = lambda _kw: seen.append(len(factory.open_transactions()))
    _seed(factory)
    _run(factory)
    assert seen == [0]


# ------------------------------------------------- sender: transport override

class _FakeSMTP:
    instances: list = []
    login_error: Exception | None = None

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.logins, self.sent = host, port, [], []
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        self.tls = True

    def login(self, user, password):
        self.logins.append((user, password))
        if _FakeSMTP.login_error:
            raise _FakeSMTP.login_error

    def sendmail(self, frm, to, msg):
        self.sent.append((frm, to))


@pytest.fixture()
def fake_smtp(monkeypatch):
    _FakeSMTP.instances, _FakeSMTP.login_error = [], None
    monkeypatch.setattr(sender.smtplib, "SMTP", _FakeSMTP)
    monkeypatch.setattr(sender.settings, "live_sending_enabled", True)
    monkeypatch.setattr(sender.settings, "smtp_host", "global.example")
    monkeypatch.setattr(sender.settings, "smtp_port", 587)
    monkeypatch.setattr(sender.settings, "smtp_username", "global-user")
    monkeypatch.setattr(sender.settings, "smtp_password", "global-pass")
    monkeypatch.setattr(sender.settings, "smtp_use_tls", False)
    return _FakeSMTP


def _send(**over):
    return sender.send_email(
        to_email="lead@example.com", from_email="sales@m.example", from_name="S",
        subject="s", body_text="b", **over,
    )


def test_send_email_uses_the_given_smtp_config_not_global_settings(fake_smtp):
    cfg = SmtpConfig(host="mbx.example", port=2525, use_tls=True, username="mbx-user", password="mbx-pass")
    assert _send(smtp_config=cfg).outcome is SendOutcome.SENT
    (conn,) = fake_smtp.instances
    assert (conn.host, conn.port, conn.logins, conn.tls) == ("mbx.example", 2525, [("mbx-user", "mbx-pass")], True)


def test_send_email_without_smtp_config_still_uses_global_settings(fake_smtp):
    assert _send().outcome is SendOutcome.SENT
    (conn,) = fake_smtp.instances
    assert (conn.host, conn.port, conn.logins) == ("global.example", 587, [("global-user", "global-pass")])


def test_smtp_authentication_failure_is_failed_and_not_retried(fake_smtp):
    fake_smtp.login_error = smtplib.SMTPAuthenticationError(535, b"bad credentials")
    cfg = SmtpConfig(host="mbx.example", port=587, use_tls=False, username="u", password="auth-fail-secret-77")
    res = _send(smtp_config=cfg)
    assert res.outcome is SendOutcome.FAILED and len(fake_smtp.instances) == 1
    assert "auth-fail-secret-77" not in (res.error or "")


# ------------------------------------------------- migration 007

def test_migration_007_adds_a_nullable_fk_column_idempotently(tmp_path, monkeypatch):
    import importlib.util
    from contextlib import contextmanager
    from pathlib import Path

    from sqlalchemy import inspect, text

    eng = create_engine(f"sqlite:///{tmp_path/'mig7.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng, tables=[t for t in Base.metadata.sorted_tables if t.name != "external_dispatches"])
    with eng.begin() as c:   # pre-007 shape: only what the migration needs to exist
        c.execute(text("CREATE TABLE external_dispatches (id INTEGER PRIMARY KEY, organization_id VARCHAR NOT NULL)"))
        c.execute(text("INSERT INTO external_dispatches (id, organization_id) VALUES (1, 'org-a')"))
    Sess = sessionmaker(bind=eng)

    @contextmanager
    def scope():
        s_ = Sess()
        try:
            yield s_
            s_.commit()
        except Exception:
            s_.rollback()
            raise
        finally:
            s_.close()

    path = Path(__file__).parent.parent / "migrations" / "007_external_dispatch_mailbox.py"
    spec = importlib.util.spec_from_file_location("migration_007", path)
    mig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mig)
    monkeypatch.setattr(mig, "session_scope", scope)

    assert mig.run() is True and mig.run() is True                          # idempotent
    cols = {c["name"]: c for c in inspect(eng).get_columns("external_dispatches")}
    assert cols["mailbox_id"]["nullable"] is True
    fks = inspect(eng).get_foreign_keys("external_dispatches")
    assert any(fk["referred_table"] == "mailboxes" and fk["constrained_columns"] == ["mailbox_id"] for fk in fks)
    with eng.connect() as c:                                                 # existing rows untouched, NULL
        assert c.execute(text("SELECT mailbox_id FROM external_dispatches WHERE id = 1")).scalar() is None
