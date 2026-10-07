"""
Real-PostgreSQL tests for M3 (mailbox-scoped inbound). Skipped -- never faked
with SQLite -- unless POSTGRES_TEST_URL points at a reachable PostgreSQL.
Dedicated test database only: tables are TRUNCATEd.

What SQLite cannot show and these do: the partial unique indexes' real
semantics, genuinely concurrent inserts of the same inbound message, and
transactional rollback (a failed commit leaves nothing behind).
"""

from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("psycopg2", reason="psycopg2 not installed -- cannot test real PostgreSQL")

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.exc import IntegrityError, OperationalError  # noqa: E402
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


pytestmark.append(pytest.mark.skipif(not _pg_available(), reason="POSTGRES_TEST_URL not set or PostgreSQL unreachable"))

from mailer_agent.mail import imap_reader, mailbox_inbound  # noqa: E402
from mailer_agent.mail import reply_handler_v2 as rh  # noqa: E402
from mailer_agent.models import Base, Message  # noqa: E402
from tests.m3_support import (  # noqa: E402
    FakeImapWorld, inbound_rows, raw_email, seed_org_with_imap_mailbox,
)

HOST_A, USER_A, PW_A = "imap.a.example", "user-a@a.example", "PwA-Imap-Secret-1111"
HOST_B, USER_B, PW_B = "imap.b.example", "user-b@b.example", "PwB-Imap-Secret-2222"
OUT_A, OUT_B = "<out-a@mailer.a>", "<out-b@mailer.b>"


@pytest.fixture
def pg(monkeypatch):
    eng = create_engine(POSTGRES_TEST_URL, pool_size=10)
    Base.metadata.create_all(bind=eng)
    with eng.connect() as c:
        c.execute(text(
            "TRUNCATE messages, external_dispatches, contacts, campaigns, mailboxes, suppression_list CASCADE"))
        c.commit()
    sm = sessionmaker(bind=eng, autoflush=False)
    import mailer_agent.llm.provider_v2 as provider_module
    monkeypatch.setattr(provider_module, "is_llm_available", lambda: False)
    yield SimpleNamespace(eng=eng, sm=sm)
    eng.dispose()


def _two_orgs(pg, world):
    world.add_account(HOST_A, USER_A, PW_A)
    world.add_account(HOST_B, USER_B, PW_B)
    with pg.sm() as db:
        mb_a, ct_a, _ = seed_org_with_imap_mailbox(
            db, org="org-a", host=HOST_A, imap_user=USER_A, imap_password=PW_A, outbound_message_id=OUT_A)
        mb_b, ct_b, _ = seed_org_with_imap_mailbox(
            db, org="org-b", host=HOST_B, imap_user=USER_B, imap_password=PW_B, outbound_message_id=OUT_B)
        return SimpleNamespace(mb_a=mb_a.id, mb_b=mb_b.id, ct_a=ct_a.id, ct_b=ct_b.id)


def _count(pg, contact_id=None):
    with pg.sm() as db:
        return len(inbound_rows(db, contact_id))


# --- schema semantics -------------------------------------------------------------

def test_partial_unique_indexes_have_the_intended_semantics(pg):
    ids = _two_orgs(pg, FakeImapWorld())
    with pg.sm() as db:
        mk = lambda cid, mid, hdr, d="inbound": Message(  # noqa: E731
            contact_id=cid, mailbox_id=mid, direction=d, body="b", status="received", message_id_header=hdr)
        db.add_all([mk(ids.ct_a, ids.mb_a, "<same@x>"), mk(ids.ct_b, ids.mb_b, "<same@x>")])
        db.commit()                                              # same id, different mailboxes: allowed
        db.add(mk(ids.ct_a, ids.mb_a, "<same@x>"))
        with pytest.raises(IntegrityError):
            db.commit()                                          # same mailbox twice: rejected
        db.rollback()
        db.add(mk(ids.ct_a, None, "<legacy@x>"))
        db.commit()
        db.add(mk(ids.ct_b, None, "<legacy@x>"))
        with pytest.raises(IntegrityError):
            db.commit()                                          # no-mailbox rows: still globally unique
        db.rollback()
        db.add_all([mk(ids.ct_a, ids.mb_a, None), mk(ids.ct_a, ids.mb_a, None), mk(ids.ct_a, None, None)])
        db.commit()                                              # NULL Message-IDs unconstrained


def test_migration_009_upgrade_and_downgrade_are_idempotent_on_postgres(pg):
    spec = importlib.util.spec_from_file_location(
        "mig009", Path(__file__).parent.parent / "migrations" / "009_messages_mailbox_scoped_dedupe.py")
    mig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mig)

    def index_names(db):
        return {r[0] for r in db.execute(text("SELECT indexname FROM pg_indexes WHERE tablename='messages'"))}

    with pg.sm() as db:
        assert mig.downgrade_schema(db) is True
        db.commit()
        names = index_names(db)
        assert not names & {mig.IDX_NO_MAILBOX, mig.IDX_MAILBOX}
        assert "uq_messages_message_id_header" in names
        assert not any(c["name"] == "mailbox_id" for c in __import__("sqlalchemy").inspect(db.connection()).get_columns("messages"))
        mig.upgrade_schema(db)
        mig.upgrade_schema(db)                                   # second run: no-op, no error
        db.commit()
        names = index_names(db)
        assert {mig.IDX_NO_MAILBOX, mig.IDX_MAILBOX} <= names and "uq_messages_message_id_header" not in names
        assert mig.verify(db)


# --- tenant isolation through real polling ----------------------------------------

def test_same_message_id_polled_for_two_orgs_concurrently_yields_one_row_per_org(pg, monkeypatch):
    world = FakeImapWorld()
    monkeypatch.setattr(imap_reader.imaplib, "IMAP4_SSL", world.connection_factory())
    ids = _two_orgs(pg, world)
    shared = raw_email(message_id="<shared-cc@list.example>")
    world.inject(HOST_A, USER_A, shared)
    world.inject(HOST_B, USER_B, shared)
    outcomes = mailbox_inbound.poll_all_mailboxes(pg.sm)
    assert {o.status for o in outcomes} == {mailbox_inbound.STATUS_OK}
    with pg.sm() as db:
        rows = inbound_rows(db)
        assert sorted((r.contact_id, r.mailbox_id) for r in rows) == sorted(
            [(ids.ct_a, ids.mb_a), (ids.ct_b, ids.mb_b)])


# --- concurrency ------------------------------------------------------------------

def test_concurrent_processing_of_one_inbound_message_creates_exactly_one_row(pg):
    ids = _two_orgs(pg, FakeImapWorld())
    email_in = imap_reader.parse_inbound_bytes(raw_email(message_id="<race@p>", in_reply_to=OUT_A))
    barrier, results, errors = threading.Barrier(6), [], []

    def worker():
        db = pg.sm()
        try:
            barrier.wait()
            r = rh.process_mailbox_inbound(db, email_in, mailbox_id=ids.mb_a, organization_id="org-a")
            db.commit()
            results.append(r["action"])
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))
            db.rollback()
        finally:
            db.close()

    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert errors == []                                          # nobody crashed on the unique violation
    assert _count(pg, ids.ct_a) == 1
    assert results.count("skipped_duplicate") >= 1 and len(results) == 6


def test_two_overlapping_polls_of_the_same_mailbox_are_safe(pg, monkeypatch):
    world = FakeImapWorld()
    monkeypatch.setattr(imap_reader.imaplib, "IMAP4_SSL", world.connection_factory())
    ids = _two_orgs(pg, world)
    for i in range(5):
        world.inject(HOST_A, USER_A, raw_email(message_id=f"<ov{i}@p>", in_reply_to=OUT_A))
    barrier, outs = threading.Barrier(2), []

    def poll():
        target = mailbox_inbound.load_pollable_target(ids.mb_a, pg.sm)
        barrier.wait()
        outs.append(mailbox_inbound.poll_mailbox(target, pg.sm))

    threads = [threading.Thread(target=poll) for _ in range(2)]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert all(o.status == mailbox_inbound.STATUS_OK for o in outs)
    assert _count(pg, ids.ct_a) == 5                             # five messages, once each
    assert world.unseen_uids(HOST_A, USER_A) == []


# --- transactional retry ----------------------------------------------------------

def test_failed_commit_rolls_back_fully_and_the_retry_succeeds(pg, monkeypatch):
    world = FakeImapWorld()
    monkeypatch.setattr(imap_reader.imaplib, "IMAP4_SSL", world.connection_factory())
    ids = _two_orgs(pg, world)
    uid = world.inject(HOST_A, USER_A, raw_email(message_id="<tx@p>", in_reply_to=OUT_A))
    state = {"fail": True}

    def factory():
        s = pg.sm()
        real = s.commit

        def commit():
            if state["fail"]:
                raise OperationalError("COMMIT", {}, Exception("connection lost"))
            return real()

        s.commit = commit
        return s

    mailbox_inbound.poll_all_mailboxes(factory)
    assert _count(pg) == 0 and not world.is_seen(HOST_A, USER_A, uid)
    state["fail"] = False
    mailbox_inbound.poll_all_mailboxes(factory)
    assert _count(pg, ids.ct_a) == 1 and world.is_seen(HOST_A, USER_A, uid)
