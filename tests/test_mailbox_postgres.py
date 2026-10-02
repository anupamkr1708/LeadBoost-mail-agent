"""
Real-PostgreSQL race tests for the Mailbox uniqueness guarantee (M1).

Skipped (never faked with SQLite) unless POSTGRES_TEST_URL points at a
reachable PostgreSQL -- same contract as the other *_postgres.py suites.
Dedicated test database only: the mailboxes table is TRUNCATEd.
"""

from __future__ import annotations

import os
import threading

import pytest

pytest.importorskip("psycopg2", reason="psycopg2 not installed -- cannot test real PostgreSQL")

from cryptography.fernet import Fernet
from fastapi import HTTPException
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

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

from mailer_agent.api import mailboxes as api  # noqa: E402
from mailer_agent.config import get_settings  # noqa: E402
from mailer_agent.mailbox_secrets import get_mailbox_credentials  # noqa: E402
from mailer_agent.models import Base, Mailbox  # noqa: E402
from mailer_agent.schemas import MailboxCreate  # noqa: E402

ORG = "org-race"


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setattr(get_settings(), "mailbox_encryption_key", SecretStr(Fernet.generate_key().decode()))


@pytest.fixture()
def factory():
    eng = create_engine(POSTGRES_TEST_URL, pool_size=20)
    Base.metadata.create_all(bind=eng)
    with eng.begin() as c:
        c.execute(text("TRUNCATE mailboxes RESTART IDENTITY"))
    yield sessionmaker(bind=eng)
    with eng.begin() as c:
        c.execute(text("TRUNCATE mailboxes RESTART IDENTITY"))
    eng.dispose()


def _payload(email="race@example.com"):
    return MailboxCreate(
        email_address=email, smtp_host="smtp.example.com", smtp_port=587, smtp_use_tls=True,
        smtp_username="u", smtp_password="pw-secret",
    )


def _create(factory, payload, org=ORG):
    s = factory()
    try:
        return api.create_mailbox(payload=payload, org_id=org, db=s)
    finally:
        s.close()


def _count(factory, org=ORG, email=None):
    s = factory()
    try:
        q = s.query(Mailbox).filter(Mailbox.organization_id == org)
        return (q.filter(Mailbox.email_address == email) if email else q).count()
    finally:
        s.close()


def test_concurrent_same_org_same_email_creates_exactly_one(factory, monkeypatch):
    racers = 10
    barrier = threading.Barrier(racers)
    real_encrypt = api.encrypt_secret

    def gated_encrypt(secret):          # releases every racer into the duplicate check together
        barrier.wait(timeout=30)
        return real_encrypt(secret)

    monkeypatch.setattr(api, "encrypt_secret", gated_encrypt)
    results: list = []
    lock = threading.Lock()

    def racer():
        try:
            out = _create(factory, _payload())
            res = ("ok", out.public_reference)
        except HTTPException as exc:
            res = ("http", exc.status_code)
        except Exception as exc:        # anything else is a failure of the guarantee
            res = ("error", repr(exc))
        with lock:
            results.append(res)

    threads = [threading.Thread(target=racer) for _ in range(racers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert sorted(r[0] for r in results).count("ok") == 1, results
    assert [r for r in results if r[0] != "ok"] == [("http", 409)] * (racers - 1), results
    assert _count(factory, email="race@example.com") == 1


def test_integrity_error_backstop_returns_409_with_no_partial_row(factory):
    """Deterministic: the loser passes the app-level check (the winner is still uncommitted),
    blocks on the unique index, then gets 409 from the IntegrityError path."""
    winner = factory()
    winner.add(Mailbox(
        organization_id=ORG, email_address="race@example.com", smtp_host="h", smtp_port=25,
        smtp_use_tls=False, smtp_username="u", smtp_password_enc="tok",
    ))
    winner.flush()                      # row exists in an open transaction only

    outcome: list = []

    def loser():
        try:
            _create(factory, _payload())
            outcome.append("created")
        except HTTPException as exc:
            outcome.append(exc.status_code)

    t = threading.Thread(target=loser)
    t.start()
    t.join(timeout=2)
    assert t.is_alive(), "loser should be blocked on the unique index while the winner is uncommitted"
    winner.commit()
    winner.close()
    t.join(timeout=30)
    assert outcome == [409]
    assert _count(factory, email="race@example.com") == 1


def test_same_email_in_different_orgs_both_succeed(factory):
    a = _create(factory, _payload(), org="org-1")
    b = _create(factory, _payload(), org="org-2")
    assert a.public_reference != b.public_reference
    assert _count(factory, "org-1") == _count(factory, "org-2") == 1


def test_postgres_roundtrip_encrypts_and_decrypts(factory):
    out = _create(factory, _payload("rt@example.com"))
    s = factory()
    try:
        row = s.query(Mailbox).filter_by(public_reference=out.public_reference).one()
        assert "pw-secret" not in row.smtp_password_enc
        assert get_mailbox_credentials(row).smtp_password == "pw-secret"
    finally:
        s.close()
