"""
Real-PostgreSQL checks for the C9.3 conversation read.

Skipped (never faked with SQLite) unless POSTGRES_TEST_URL points at a reachable
PostgreSQL -- same convention as test_leadboost_reconciliation_postgres.py.
Dedicated test database only: it is TRUNCATEd.

What SQLite cannot show and these do:
  * the read really runs in ONE REPEATABLE READ, READ ONLY transaction, so a
    dispatch that changes mid-request can never produce a torn action/message
    view (with a negative control proving the test can detect tearing);
  * the read takes no locks: it neither blocks, nor is blocked by, FOR UPDATE
    SKIP LOCKED claims or a writer holding row locks;
  * the isolation / read-only setting does not leak onto pooled connections;
  * organization and mailbox isolation hold with timezone-aware timestamps;
  * real M3 inbound (the actual poll path) appears, and concurrent reads do not
    disturb mailbox-scoped dedupe;
  * the query's predicates are index-usable (EXPLAIN).
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from types import SimpleNamespace
from urllib.parse import quote

import pytest

pytest.importorskip("psycopg2", reason="psycopg2 not installed -- cannot test real PostgreSQL")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, event, text  # noqa: E402
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

from mailer_agent.api import integrations_conversation as conv_module  # noqa: E402
from mailer_agent.api.deps import get_integration_org_id  # noqa: E402
from mailer_agent.api.main import app  # noqa: E402
from mailer_agent.db import get_db  # noqa: E402
from mailer_agent.mail import imap_reader, mailbox_inbound  # noqa: E402
from mailer_agent.models import Base, ExternalDispatch, Message  # noqa: E402
from tests.dispatch_support import seed_dispatch  # noqa: E402
from tests.m3_support import FakeImapWorld, inbound_rows, raw_email, seed_org_with_imap_mailbox  # noqa: E402

BASE = "/integrations/leadboost/outreach-actions"
_TRUNC = (
    "TRUNCATE external_dispatches, mailboxes, messages, contacts, campaigns, "
    "suppression_list RESTART IDENTITY CASCADE"
)


def url(key: str) -> str:
    return f"{BASE}/{quote(key, safe='')}/conversation"


def _wire(eng, holder):
    """get_db yields a FRESH session per request and closes it -- like production
    (a shared session would already be in a transaction, defeating the snapshot)."""
    sm = sessionmaker(bind=eng)

    def _get_db():
        s = sm()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_integration_org_id] = lambda: holder["org"]
    return sm


@pytest.fixture()
def pg():
    eng = create_engine(POSTGRES_TEST_URL, pool_size=10)
    Base.metadata.create_all(bind=eng)
    with eng.begin() as c:
        c.execute(text(_TRUNC))
    holder = {"org": "org-a"}
    sm = _wire(eng, holder)
    writer = create_engine(POSTGRES_TEST_URL)       # an independent "worker" connection source
    try:
        yield SimpleNamespace(eng=eng, sm=sm, holder=holder, writer=writer, client=TestClient(app))
    finally:
        app.dependency_overrides.clear()
        writer.dispose()
        with eng.begin() as c:
            c.execute(text(_TRUNC))
        eng.dispose()


def _inbound(sm, contact_id, mailbox_id, body, *, header=None):
    with sm() as db:
        db.add(Message(contact_id=contact_id, direction="inbound", subject="Re: hi", body=body,
                       status="received", mailbox_id=mailbox_id, message_id_header=header))
        db.commit()


def _seed(pg, **kw):
    with pg.sm() as db:
        d = seed_dispatch(db, **kw)
        return SimpleNamespace(id=d.id, contact_id=d.contact_id, message_id=d.message_id,
                               mailbox_id=d.mailbox_id, ref=d.public_reference)


# ---------------------------------------------------------------------------
# Isolation, states, timestamps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state", ["queued", "sending", "sent", "failed", "unknown", "generating"])
def test_states_and_cross_tenant_404_on_postgres(pg, state):
    d = _seed(pg, org="org-a", idem="k1", state=state)
    expected = "queued" if state == "generating" else state
    body = pg.client.get(url("k1")).json()
    assert body["action"]["state"] == expected and body["messages"][0]["delivery_state"] == expected
    assert body["action"]["mailing_agent_reference"] == d.ref
    for stamp in (body["action"]["created_at"], body["action"]["updated_at"], body["messages"][0]["created_at"]):
        assert datetime.fromisoformat(stamp.replace("Z", "+00:00")).tzinfo is not None   # timestamptz survives
    pg.holder["org"] = "org-b"
    cross = pg.client.get(url("k1"))
    missing = pg.client.get(url("never"))
    assert cross.status_code == missing.status_code == 404 and cross.json() == missing.json()
    assert d.ref not in cross.text


def test_organization_and_mailbox_isolation_on_postgres(pg):
    a = _seed(pg, org="org-a", idem="k1", email="same@example.com")
    b = _seed(pg, org="org-b", idem="k1", email="same@example.com", sender_email="outreach@b.example.org")
    _inbound(pg.sm, a.contact_id, a.mailbox_id, "A-OWN-REPLY")                  # bound to A's mailbox: shown
    _inbound(pg.sm, a.contact_id, None, "NULL-MAILBOX-SENTINEL")                # webhook / legacy: hidden
    _inbound(pg.sm, a.contact_id, b.mailbox_id, "FOREIGN-MAILBOX-SENTINEL")     # B's mailbox on A's contact: hidden
    _inbound(pg.sm, b.contact_id, b.mailbox_id, "B-OWN-REPLY")

    ra = pg.client.get(url("k1"))
    assert [m["body"][:5] for m in ra.json()["messages"]] == ["Hello", "A-OWN"]
    for leak in ("NULL-MAILBOX-SENTINEL", "FOREIGN-MAILBOX-SENTINEL", "B-OWN-REPLY", b.ref):
        assert leak not in ra.text
    pg.holder["org"] = "org-b"
    rb = pg.client.get(url("k1"))
    assert [m["body"][:5] for m in rb.json()["messages"]] == ["Hello", "B-OWN"]
    assert "A-OWN-REPLY" not in rb.text and a.ref not in rb.text


def test_read_mutates_nothing_on_postgres(pg):
    d = _seed(pg, org="org-a", idem="k1", state="sending")
    _inbound(pg.sm, d.contact_id, d.mailbox_id, "reply")

    def raw():
        with pg.eng.connect() as c:
            return [
                [tuple(r) for r in c.execute(text(f"SELECT * FROM {t} ORDER BY id"))]
                for t in ("external_dispatches", "messages", "contacts", "campaigns", "mailboxes")
            ]

    before = raw()
    for _ in range(3):
        assert pg.client.get(url("k1")).status_code == 200
    assert raw() == before


# ---------------------------------------------------------------------------
# One REPEATABLE READ, READ ONLY snapshot
# ---------------------------------------------------------------------------

def _probe(eng, sink):
    """Record, on the request's own connection, the transaction mode in effect
    at its first statement (via the raw DBAPI cursor, so no recursion)."""
    def _hook(conn, cursor, statement, parameters, context, executemany):
        if "iso" in sink:
            return
        cur = conn.connection.cursor()
        cur.execute("SHOW transaction_isolation")
        sink["iso"] = cur.fetchone()[0]
        cur.execute("SHOW transaction_read_only")
        sink["ro"] = cur.fetchone()[0]
        cur.close()
    event.listen(eng, "before_cursor_execute", _hook)
    return _hook


def test_read_runs_in_a_repeatable_read_read_only_transaction(pg):
    _seed(pg, org="org-a", idem="k1")
    sink: dict = {}
    hook = _probe(pg.eng, sink)
    try:
        assert pg.client.get(url("k1")).status_code == 200
    finally:
        event.remove(pg.eng, "before_cursor_execute", hook)
    assert sink == {"iso": "repeatable read", "ro": "on"}


def test_a_write_attempted_inside_the_read_transaction_would_be_refused(pg):
    """READ ONLY is enforced by the server: prove a write on the request's own
    connection fails, not merely that the endpoint chooses not to write."""
    _seed(pg, org="org-a", idem="k1")
    outcome: dict = {}

    def _hook(conn, cursor, statement, parameters, context, executemany):
        if "tried" in outcome:
            return
        outcome["tried"] = True
        cur = conn.connection.cursor()
        try:
            cur.execute("UPDATE external_dispatches SET state = 'failed'")
            outcome["error"] = None
        except Exception as exc:  # psycopg2.errors.ReadOnlySqlTransaction
            outcome["error"] = type(exc).__name__
        finally:
            cur.close()
    event.listen(pg.eng, "before_cursor_execute", _hook)
    try:
        # The refused write aborts that transaction, so the request itself may 500;
        # the assertion is about the server-side refusal, not the HTTP outcome.
        try:
            pg.client.get(url("k1"))
        except Exception:
            pass
    finally:
        event.remove(pg.eng, "before_cursor_execute", _hook)
    assert outcome["error"] == "ReadOnlySqlTransaction"
    with pg.eng.connect() as c:
        assert c.execute(text("SELECT state FROM external_dispatches")).scalar() == "queued"


def _claim_mid_request(pg, d, sink):
    """After the request's FIRST statement (the action lookup), a 'worker' claims
    the dispatch -- FOR UPDATE SKIP LOCKED, then queued -> sending -- and commits.
    Mirrors external_dispatch_worker's claim, on an independent connection."""
    def _hook(conn, cursor, statement, parameters, context, executemany):
        if sink.get("fired") or "FROM external_dispatches" not in statement or "FROM messages" in statement:
            return
        sink["fired"] = True
        with pg.writer.begin() as w:
            got = w.execute(text(
                "SELECT id FROM external_dispatches WHERE id = :i AND state = 'queued' "
                "FOR UPDATE SKIP LOCKED"), {"i": d.id}).fetchall()
            sink["claimed_rows"] = len(got)             # 1 == not blocked, not skipped by the open read
            w.execute(text("UPDATE external_dispatches SET state = 'sending', claimed_by = 'w1' WHERE id = :i"),
                      {"i": d.id})
            w.execute(text("UPDATE messages SET status = 'sending' WHERE id = :m"), {"m": d.message_id})
    event.listen(pg.eng, "after_cursor_execute", _hook)
    return _hook


def test_no_torn_view_when_a_worker_claims_mid_request_and_the_claim_is_not_blocked(pg):
    d = _seed(pg, org="org-a", idem="k1", state="queued")
    sink: dict = {}
    hook = _claim_mid_request(pg, d, sink)
    try:
        body = pg.client.get(url("k1")).json()
    finally:
        event.remove(pg.eng, "after_cursor_execute", hook)
    assert sink["fired"] and sink["claimed_rows"] == 1            # the worker's claim was neither blocked nor skipped
    # one snapshot: action and message agree, both as of before the claim
    assert body["action"]["state"] == "queued"
    assert [m["delivery_state"] for m in body["messages"]] == ["queued"]
    # ...and the claim really did commit meanwhile: the next read sees it, consistently
    after = pg.client.get(url("k1")).json()
    assert after["action"]["state"] == "sending" and after["messages"][0]["delivery_state"] == "sending"


def test_negative_control_without_the_snapshot_the_same_race_tears_the_view(pg, monkeypatch):
    """Proves the test above can fail: disable the snapshot and the identical race
    yields action=queued but message=sending."""
    monkeypatch.setattr(conv_module, "_begin_consistent_read", lambda db: None)
    d = _seed(pg, org="org-a", idem="k1", state="queued")
    sink: dict = {}
    hook = _claim_mid_request(pg, d, sink)
    try:
        body = pg.client.get(url("k1")).json()
    finally:
        event.remove(pg.eng, "after_cursor_execute", hook)
    assert body["action"]["state"] == "queued"
    assert body["messages"][0]["delivery_state"] == "sending"     # torn without REPEATABLE READ


def test_read_is_not_blocked_by_row_locks_held_by_a_writer(pg):
    d = _seed(pg, org="org-a", idem="k1", state="queued")
    result: dict = {}
    with pg.writer.connect() as w:
        w.execute(text("SELECT id FROM external_dispatches WHERE id = :i FOR UPDATE"), {"i": d.id})
        w.execute(text("SELECT id FROM messages WHERE id = :m FOR UPDATE"), {"m": d.message_id})
        w.execute(text("SELECT id FROM contacts WHERE id = :c FOR UPDATE"), {"c": d.contact_id})

        def go():
            result["r"] = pg.client.get(url("k1"))

        t = threading.Thread(target=go)
        t.start()
        t.join(15)
        assert not t.is_alive(), "the conversation read blocked on a writer's row locks"
        w.rollback()
    assert result["r"].status_code == 200 and result["r"].json()["action"]["state"] == "queued"


def test_snapshot_settings_do_not_leak_onto_pooled_connections():
    eng = create_engine(POSTGRES_TEST_URL, pool_size=1, max_overflow=0)   # force connection reuse
    Base.metadata.create_all(bind=eng)
    with eng.begin() as c:
        c.execute(text(_TRUNC))
    holder = {"org": "org-a"}
    sm = _wire(eng, holder)
    try:
        with sm() as db:
            seed_dispatch(db, org="org-a", idem="k1")
        client = TestClient(app)
        for _ in range(3):
            assert client.get(url("k1")).status_code == 200
        with eng.connect() as c:                                         # the very same pooled connection
            assert c.execute(text("SHOW transaction_read_only")).scalar() == "off"
            assert c.execute(text("SHOW transaction_isolation")).scalar() == "read committed"
        with eng.begin() as c:                                           # and it can write again
            assert c.execute(text("UPDATE contacts SET name = 'writable-after-read'")).rowcount == 1
    finally:
        app.dependency_overrides.clear()
        with eng.begin() as c:
            c.execute(text(_TRUNC))
        eng.dispose()


# ---------------------------------------------------------------------------
# Real M3 inbound through the real poll path
# ---------------------------------------------------------------------------

HOST_A, USER_A, PW_A = "imap.a.example", "user-a@a.example", "PwA-Imap-Secret-1111"
HOST_B, USER_B, PW_B = "imap.b.example", "user-b@b.example", "PwB-Imap-Secret-2222"
OUT_A, OUT_B = "<out-a@mailer.a>", "<out-b@mailer.b>"


def _two_org_imap(pg, world):
    world.add_account(HOST_A, USER_A, PW_A)
    world.add_account(HOST_B, USER_B, PW_B)
    with pg.sm() as db:
        mb_a, ct_a, _ = seed_org_with_imap_mailbox(
            db, org="org-a", host=HOST_A, imap_user=USER_A, imap_password=PW_A, outbound_message_id=OUT_A)
        mb_b, ct_b, _ = seed_org_with_imap_mailbox(
            db, org="org-b", host=HOST_B, imap_user=USER_B, imap_password=PW_B, outbound_message_id=OUT_B)
        keys = {}
        for org in ("org-a", "org-b"):
            keys[org] = db.query(ExternalDispatch).filter(ExternalDispatch.organization_id == org).one().idempotency_key
        return SimpleNamespace(mb_a=mb_a, mb_b=mb_b, ct_a=ct_a.id, ct_b=ct_b.id, keys=keys,
                               ref_a=mb_a.public_reference, ref_b=mb_b.public_reference)


@pytest.fixture()
def imap(pg, monkeypatch):
    import mailer_agent.llm.provider_v2 as provider_module
    monkeypatch.setattr(provider_module, "is_llm_available", lambda: False)
    world = FakeImapWorld()
    monkeypatch.setattr(imap_reader.imaplib, "IMAP4_SSL", world.connection_factory())
    return SimpleNamespace(world=world, ids=_two_org_imap(pg, world))


def test_real_m3_inbound_appears_and_stays_inside_its_organization(pg, imap):
    ids, world = imap.ids, imap.world
    world.inject(HOST_A, USER_A, raw_email(message_id="<r1@p>", in_reply_to=OUT_A, body="Reply for A only."))
    world.inject(HOST_B, USER_B, raw_email(message_id="<r2@p>", in_reply_to=OUT_B, body="Reply for B only."))
    outcomes = mailbox_inbound.poll_all_mailboxes(pg.sm)
    assert {o.status for o in outcomes} == {mailbox_inbound.STATUS_OK}

    ra = pg.client.get(url(ids.keys["org-a"]))
    inbound = [m for m in ra.json()["messages"] if m["direction"] == "inbound"]
    assert [m["body"].strip() for m in inbound] == ["Reply for A only."]
    assert inbound[0]["mailbox_reference"] == ids.ref_a
    assert "Reply for B only." not in ra.text
    assert "r1@p" not in ra.text and "imap" not in ra.text.lower().replace("inbound", "")   # no Message-ID / IMAP detail
    pg.holder["org"] = "org-b"
    rb = pg.client.get(url(ids.keys["org-b"]))
    assert "Reply for A only." not in rb.text and "Reply for B only." in rb.text


def test_polling_dedupe_is_unaffected_and_reads_stay_consistent_under_concurrent_polls(pg, imap):
    ids, world = imap.ids, imap.world
    for i in range(5):
        world.inject(HOST_A, USER_A, raw_email(message_id=f"<ov{i}@p>", in_reply_to=OUT_A, body=f"msg {i}"))
    key = ids.keys["org-a"]
    barrier, outs, seen, bad = threading.Barrier(3), [], [], []

    def poll():
        target = mailbox_inbound.load_pollable_target(ids.mb_a.id, pg.sm)
        barrier.wait()
        outs.append(mailbox_inbound.poll_mailbox(target, pg.sm))

    def read():
        barrier.wait()
        for _ in range(15):
            r = pg.client.get(url(key))
            if r.status_code != 200:
                bad.append(r.status_code)
                continue
            msgs = r.json()["messages"]
            seen.append(sum(m["direction"] == "inbound" for m in msgs))
            if len({m["body"] for m in msgs}) != len(msgs):
                bad.append("duplicate message in one response")

    threads = [threading.Thread(target=poll), threading.Thread(target=poll), threading.Thread(target=read)]
    [t.start() for t in threads]
    [t.join(90) for t in threads]
    assert bad == []
    assert all(o.status == mailbox_inbound.STATUS_OK for o in outs)
    with pg.sm() as db:
        assert len(inbound_rows(db, ids.ct_a)) == 5                      # five messages, once each, despite the reader
    assert max(seen) <= 5 and seen == sorted(seen) or max(seen) <= 5     # never more than exist; never a dup
    final = [m for m in pg.client.get(url(key)).json()["messages"] if m["direction"] == "inbound"]
    assert len(final) == 5
    mailbox_inbound.poll_all_mailboxes(pg.sm)                            # a further poll adds nothing
    assert len([m for m in pg.client.get(url(key)).json()["messages"] if m["direction"] == "inbound"]) == 5


# ---------------------------------------------------------------------------
# Index usability (EXPLAIN)
# ---------------------------------------------------------------------------

def _seed_filler_orgs(eng, orgs=3, contacts=300):
    """Bulk rows in other organizations so the planner has real statistics to choose
    with (an empty, never-analyzed table makes every access path look equal)."""
    with eng.begin() as c:
        for o in range(orgs):
            org = f"filler-{o}"
            c.execute(text(
                "INSERT INTO campaigns (name, organization_id, integration_source, sender_name, sender_org, "
                "sender_email, value_prop) VALUES (:n, :o, 'leadboost', 's', 's', 's@x.org', 'v')"),
                {"n": f"f{o}", "o": org})
            cid = c.execute(text("SELECT id FROM campaigns WHERE organization_id = :o"), {"o": org}).scalar()
            c.execute(text("INSERT INTO contacts (campaign_id, email) "
                           "SELECT :c, 'f'||g||'@ex.com' FROM generate_series(1, :n) g"), {"c": cid, "n": contacts})
            c.execute(text("INSERT INTO messages (contact_id, direction, subject, body, status) "
                           "SELECT ct.id, 'outbound', 's', 'b', 'sent' FROM contacts ct WHERE ct.campaign_id = :c"),
                      {"c": cid})
            c.execute(text(
                "INSERT INTO external_dispatches (organization_id, idempotency_key, campaign_id, contact_id, "
                "message_id, request_fingerprint, public_reference, state) "
                "SELECT :o, 'fk-'||ct.id, :c, ct.id, m.id, 'f', md5('r'||ct.id), 'sent' "
                "FROM contacts ct JOIN messages m ON m.contact_id = ct.id WHERE ct.campaign_id = :c"),
                {"o": org, "c": cid})
        c.execute(text("ANALYZE"))


def test_query_predicates_are_index_usable(pg):
    """The action lookup is an index lookup (never a sequential scan of dispatches) and
    the message window uses messages(contact_id) -- an index that
    exists only via migration 002, not create_all, so it is created here exactly as
    the migration does. Which of the two usable dispatch indexes -- the (organization_id,
    idempotency_key) unique constraint or the single-column organization_id index -- wins
    is a planner cost tie at one-row estimates, so it is not asserted by name (the
    measured realistic-volume plan used the unique constraint). Seqscan is disabled for the messages plan only so a small
    table can't hide the question. C9.3 adds no index. NOTE: the dispatch side of
    the message join has no contact_id/message_id index and therefore scales with the
    size of external_dispatches; that measured finding is recorded in the C9.3 report
    and deliberately NOT asserted here as if it were a guarantee."""
    _seed_filler_orgs(pg.eng)
    with pg.eng.begin() as c:
        c.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_contact_id ON messages(contact_id)"))
    try:
        d = _seed(pg, org="org-a", idem="k1")
        _inbound(pg.sm, d.contact_id, d.mailbox_id, "reply")
        with pg.eng.begin() as c:
            c.execute(text("ANALYZE"))
        captured: list[tuple[str, dict]] = []

        def _cap(conn, cursor, statement, parameters, context, executemany):
            captured.append((statement, parameters))
        event.listen(pg.eng, "before_cursor_execute", _cap)
        try:
            assert pg.client.get(url("k1")).status_code == 200
        finally:
            event.remove(pg.eng, "before_cursor_execute", _cap)

        root_sql, root_params = next((s, p) for s, p in captured if "FROM external_dispatches" in s and "FROM messages" not in s)
        msg_sql, msg_params = next((s, p) for s, p in captured if "FROM messages" in s)
        with pg.eng.begin() as c:
            root_plan = "\n".join(r[0] for r in c.exec_driver_sql("EXPLAIN " + root_sql, root_params))
        with pg.eng.begin() as c:
            c.exec_driver_sql("SET LOCAL enable_seqscan = off")
            msg_plan = "\n".join(r[0] for r in c.exec_driver_sql("EXPLAIN " + msg_sql, msg_params))
        assert "Seq Scan on external_dispatches" not in root_plan, root_plan
        assert "Index Scan using" in root_plan and "on external_dispatches" in root_plan, root_plan
        assert "idx_messages_contact_id" in msg_plan, msg_plan
    finally:
        with pg.eng.begin() as c:
            c.execute(text("DROP INDEX IF EXISTS idx_messages_contact_id"))
