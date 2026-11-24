#!/usr/bin/env python3
"""Rehearse the Mailer's C9.5 database release on SYNTHETIC data in a throw-away PostgreSQL database.

What this is
------------
An operator harness for the rollout (migrations 010 -> 013 through the real release wrapper, including the
migration-013 legacy-reply gate). It never touches the real Mailer database: there is no Mailer production dump
and none is needed, every row it creates is invented here.

SAFETY (refuses, never warns)
-----------------------------
* The PostgreSQL server must be 127.0.0.1:56432 (the throw-away ``c95-pg16`` container) and every database this
  script creates, copies, restores or drops must be named ``c95_rehearsal_*``. Anything else exits 2 before a
  single statement is sent.
* ``DATABASE_URL`` and every other secret-bearing variable of the caller are NOT forwarded to child processes: the
  children get a small allow-list plus synthetic values generated here (an in-memory Fernet key, throw-away API keys).
* It refuses to run when a ``.env`` file exists in the repository root (``mailer_agent.config.Settings`` reads
  ``.env`` from the working directory; a real one would leak real settings into the rehearsal).
* ``LIVE_SENDING_ENABLED=false`` and ``AUTO_REPLY_ENABLED=false`` for every child; no SMTP/IMAP/LLM is configured.
* ``cleanup`` drops ONLY databases recorded in this script's own state file.

Usage (from the repository root, with the venv's python)::

    export REHEARSAL_ADMIN_URL=postgresql://postgres:<password>@127.0.0.1:56432/postgres   # never printed
    export REHEARSAL_STATE_DIR="$TMPD/rehearsal"                                            # any writable dir
    python scripts/rehearse_c95_migrations.py run            # whole scenario, transcript on stdout, exit 0 = all ok
    python scripts/rehearse_c95_migrations.py status         # databases recorded so far
    python scripts/rehearse_c95_migrations.py cleanup        # drop what THIS script created

Optional: ``REHEARSAL_PG_CLIENT=docker:c95-pg16`` runs pg_dump/pg_restore inside that container (stdin/stdout only,
no file is left in the container); default ``local`` uses pg_dump/pg_restore on PATH.
``REHEARSAL_BULK_MESSAGES`` (default 20000) sets the synthetic ``messages`` volume used for timings.

Exit status: 0 every expectation held; 1 an expectation failed; 2 refused (guard); 3 environment problem.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent

GUARD_HOST = "127.0.0.1"
GUARD_PORT = 56432
DB_PREFIX = "c95_rehearsal_"
EXIT_OK, EXIT_EXPECT_FAILED, EXIT_REFUSED, EXIT_ENV = 0, 1, 2, 3
_DB_NAME_OK = re.compile(r"^c95_rehearsal_[a-z0-9_]{1,40}$")

# The wrapper's own sequence (scripts/migrate_release.sh STEPS), in the README's canonical order.
STEPS_001_009 = [
    "migrations/001_production_hardening.py",
    "migrations/002_constraints_and_multitenancy.py",
    "migrations/002_work_claiming.py",
    "migrate_db.py",
    "migrations/003_message_id_unique_constraint.py",
    "migrations/004_external_dispatch_and_campaign_integration_source.py",
    "migrations/005_external_dispatch_grounding_context.py",
    "migrations/006_mailboxes.py",
    "migrations/007_external_dispatch_mailbox.py",
    "migrations/008_external_dispatch_deferred_message.py",
    "migrations/009_messages_mailbox_scoped_dedupe.py",
]
REWIND = [  # newest first; the repository's own downgrade scripts, run on an EMPTY-of-meaning synthetic DB only
    "migrations/013_external_dispatch_content_approval.py",
    "migrations/012_message_reply_reference.py",
    "migrations/011_unmatched_inbound_messages.py",
    "migrations/010_worker_runtime_status.py",
]
# Read-only: how many legacy-shaped managed replies exist, by state (the same predicate the 013 gate uses).
# `::jsonb` makes it valid whether grounding_context was created as json or jsonb.
LEGACY_COUNT_SQL = (
    "SELECT state, (message_id IS NOT NULL) AS has_message, count(*) FROM external_dispatches "
    "WHERE jsonb_typeof(grounding_context::jsonb -> 'reply_parent') = 'object' GROUP BY 1, 2 ORDER BY 1, 2"
)
CHILD_ENV_ALLOW = ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONPYCACHEPREFIX", "TMPDIR", "SYSTEMROOT", "REHEARSAL_BULK_MESSAGES")


class Refused(Exception):
    pass


# --------------------------------------------------------------------------------------------------------------
# Guard (pure functions: unit-tested in tests/test_rehearsal_guard.py)
# --------------------------------------------------------------------------------------------------------------

def assert_rehearsal_server(host: str | None, port: int | None) -> None:
    if (host or "").lower() != GUARD_HOST or port != GUARD_PORT:
        raise Refused(
            f"refusing: the PostgreSQL server must be {GUARD_HOST}:{GUARD_PORT} (the throw-away c95-pg16 container); "
            f"got host={host!r} port={port!r}. Nothing was sent."
        )


def assert_rehearsal_db_name(name: str) -> None:
    if not isinstance(name, str) or not _DB_NAME_OK.match(name):
        raise Refused(f"refusing: database name {name!r} does not match {DB_PREFIX}[a-z0-9_]+ . Nothing was sent.")


def parse_admin_url(raw: str) -> dict:
    if not raw:
        raise Refused("refusing: REHEARSAL_ADMIN_URL is not set (e.g. postgresql://postgres:<pw>@127.0.0.1:56432/postgres).")
    parts = urlsplit(raw.replace("postgresql+psycopg2://", "postgresql://", 1))
    if parts.scheme not in ("postgresql", "postgres"):
        raise Refused("refusing: REHEARSAL_ADMIN_URL is not a PostgreSQL URL.")
    assert_rehearsal_server(parts.hostname, parts.port)
    return {
        "host": parts.hostname, "port": parts.port, "user": unquote(parts.username or "postgres"),
        "password": unquote(parts.password or ""), "admin_db": (parts.path or "/postgres").lstrip("/") or "postgres",
    }


def db_url(admin: dict, dbname: str, driver: bool = True) -> str:
    assert_rehearsal_server(admin["host"], admin["port"])
    assert_rehearsal_db_name(dbname)
    scheme = "postgresql+psycopg2" if driver else "postgresql"
    return f"{scheme}://{quote(admin['user'])}:{quote(admin['password'], safe='')}@{admin['host']}:{admin['port']}/{dbname}"


def make_db_name(label: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dt%H%M%Sz")
    name = f"{DB_PREFIX}{re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')}_{stamp}"
    assert_rehearsal_db_name(name)
    return name


def extract_readme_sql(readme_text: str, marker: str) -> str:
    """The first ```sql block after ``<!-- marker -->``. Parsing the operator README (not a copy) means this
    rehearsal exercises the exact statement an operator would paste."""
    m = re.search(r"<!--\s*" + re.escape(marker) + r"\s*-->\s*```sql\n(.*?)```", readme_text, re.S)
    if not m:
        raise Refused(f"README snippet {marker!r} not found (the README changed: update the harness, do not guess).")
    return m.group(1).strip().rstrip(";")


# --------------------------------------------------------------------------------------------------------------
# Transcript + expectations
# --------------------------------------------------------------------------------------------------------------

class Transcript:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []
        self.t0 = time.monotonic()

    def say(self, msg: str = "") -> None:
        print(msg, flush=True)

    def step(self, title: str) -> None:
        self.say(f"\n=== {title}  [t+{time.monotonic() - self.t0:6.1f}s]")

    def expect(self, name: str, ok: bool, got: str = "") -> bool:
        self.results.append((name, bool(ok), got))
        self.say(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   (got: {got})" if got else ""))
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(ok for _n, ok, _g in self.results)


T = Transcript()


# --------------------------------------------------------------------------------------------------------------
# Environment plumbing
# --------------------------------------------------------------------------------------------------------------

def python_exe() -> str:
    return os.environ.get("PYTHON") or sys.executable


def child_env(admin: dict, dbname: str, extra: dict | None = None) -> dict:
    env = {k: os.environ[k] for k in CHILD_ENV_ALLOW if k in os.environ}
    env.update({
        "PYTHON": python_exe(),
        "DATABASE_URL": db_url(admin, dbname),
        "LIVE_SENDING_ENABLED": "false", "AUTO_REPLY_ENABLED": "false", "RUN_SCHEDULER_IN_PROCESS": "false",
        "MAILBOX_ENCRYPTION_KEY": _SYNTH["fernet"], "ORG_KEY_MAP": json.dumps({_SYNTH["tenant_key"]: "rehearsal-org"}),
        "SYSTEM_OPERATOR_API_KEY": _SYNTH["operator_key"],
    })
    if extra:
        env.update(extra)
    return env


_SYNTH: dict = {}


def init_synthetic_secrets() -> None:
    from cryptography.fernet import Fernet  # present in the Mailer's requirements

    _SYNTH["fernet"] = Fernet.generate_key().decode()
    _SYNTH["tenant_key"] = "rh-tenant-" + secrets.token_hex(16)
    _SYNTH["operator_key"] = "rh-operator-" + secrets.token_hex(16)


def sh(args: list[str], env: dict, timeout: int = 1800, stdin_path: Path | None = None, stdout_path: Path | None = None):
    t = time.monotonic()
    stdin = open(stdin_path, "rb") if stdin_path else None
    stdout = open(stdout_path, "wb") if stdout_path else subprocess.PIPE
    try:
        p = subprocess.run(args, cwd=ROOT, env=env, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE,
                           timeout=timeout, check=False)
    finally:
        if stdin:
            stdin.close()
        if stdout_path:
            stdout.close()
    out = (p.stdout or b"").decode("utf-8", "replace") if not stdout_path else ""
    err = (p.stderr or b"").decode("utf-8", "replace")
    return p.returncode, out + err, time.monotonic() - t


def tail(text: str, n: int = 6) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return " | ".join(lines[-n:])[:600]


# --------------------------------------------------------------------------------------------------------------
# Database helpers (admin connection = autocommit, server/prefix guarded on every call)
# --------------------------------------------------------------------------------------------------------------

def _engine(admin: dict, dbname: str | None = None):
    from sqlalchemy import create_engine

    if dbname is None:
        assert_rehearsal_server(admin["host"], admin["port"])
        url = f"postgresql+psycopg2://{quote(admin['user'])}:{quote(admin['password'], safe='')}@{admin['host']}:{admin['port']}/{admin['admin_db']}"
    else:
        url = db_url(admin, dbname)
    return create_engine(url, isolation_level="AUTOCOMMIT")


def state_path() -> Path:
    d = os.environ.get("REHEARSAL_STATE_DIR")
    if not d:
        raise Refused("refusing: REHEARSAL_STATE_DIR is not set (a writable directory outside the repository).")
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p / "rehearsal_mailer_state.json"


def load_state() -> dict:
    p = state_path()
    return json.loads(p.read_text()) if p.exists() else {"databases": []}


def record(name: str) -> None:
    st = load_state()
    if name not in st["databases"]:
        st["databases"].append(name)
    state_path().write_text(json.dumps(st, indent=2))


def create_db(admin: dict, name: str, template: str | None = None) -> None:
    from sqlalchemy import text

    assert_rehearsal_db_name(name)
    if template:
        assert_rehearsal_db_name(template)
    with _engine(admin).connect() as c:
        c.execute(text(f'CREATE DATABASE "{name}"' + (f' TEMPLATE "{template}"' if template else "")))
    record(name)


def drop_db(admin: dict, name: str) -> None:
    from sqlalchemy import text

    assert_rehearsal_db_name(name)
    with _engine(admin).connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


def q(admin: dict, dbname: str, sql: str, params: dict | None = None):
    from sqlalchemy import text

    eng = _engine(admin, dbname)
    try:
        with eng.connect() as c:
            res = c.execute(text(sql), params or {})
            return res.fetchall() if res.returns_rows else res.rowcount
    finally:
        eng.dispose()


def table_counts(admin: dict, dbname: str) -> dict[str, int]:
    names = [r[0] for r in q(admin, dbname, "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1")]
    return {n: q(admin, dbname, f'SELECT count(*) FROM "{n}"')[0][0] for n in names}


def pg_client_cmd(admin: dict, tool: str, dbname: str, extra: list[str]) -> tuple[list[str], dict]:
    mode = os.environ.get("REHEARSAL_PG_CLIENT", "local")
    env = {k: os.environ[k] for k in CHILD_ENV_ALLOW if k in os.environ}
    env["PGPASSWORD"] = admin["password"]
    if mode.startswith("docker:"):
        container = mode.split(":", 1)[1]
        if not container.startswith("c95-"):
            raise Refused("refusing: REHEARSAL_PG_CLIENT container must be named c95-*")
        return (["docker", "exec", "-i", "-e", "PGPASSWORD", container, tool, "-U", admin["user"], "-d", dbname, *extra], env)
    if not shutil.which(tool):
        raise Refused(f"{tool} not found on PATH (set REHEARSAL_PG_CLIENT=docker:c95-pg16 or install the PostgreSQL client).")
    return ([tool, "-h", admin["host"], "-p", str(admin["port"]), "-U", admin["user"], "-d", dbname, *extra], env)


# --------------------------------------------------------------------------------------------------------------
# Child processes that need the Mailer's ORM (the parent never imports mailer_agent)
# --------------------------------------------------------------------------------------------------------------

def child_seed() -> int:
    """Runs in a child with DATABASE_URL pointing at the rehearsal DB, CURRENT schema. Synthetic rows only."""
    import uuid

    sys.path.insert(0, str(ROOT))
    from mailer_agent.db import SessionLocal  # noqa: PLC0415
    from mailer_agent.mail.exact_message import create_authorized_message
    from mailer_agent.mailbox_secrets import encrypt_secret
    from mailer_agent.models import (Campaign, Contact, ExternalDispatch, Mailbox, Message, MessageDirection,
                                     MessageType, MessageStatus)

    bulk = int(os.environ.get("REHEARSAL_BULK_MESSAGES", "20000"))
    org = "rehearsal-org"
    out: dict[str, int] = {}
    with SessionLocal() as db:
        camp = Campaign(name="leadboost-rehearsal", organization_id=org, integration_source="leadboost",
                        sender_name="Synthetic", sender_org="Synthetic", sender_email="outreach@sender.invalid",
                        value_prop="synthetic value prop")
        db.add(camp); db.flush()
        mbx = Mailbox(organization_id=org, email_address="outreach@sender.invalid", smtp_host="smtp.invalid",
                      smtp_port=2525, smtp_use_tls=False, smtp_username="outreach@sender.invalid",
                      smtp_password_enc=encrypt_secret("synthetic-not-a-secret"), status="active")
        db.add(mbx); db.flush()

        def reply_gc(parent_id: int) -> dict:
            return {"version": 2, "value_prop": "synthetic", "context_notes": None, "conversation_transcript": None,
                    "recipient": {"name": "Synthetic", "title": None, "company": None},
                    "reply_parent": {"message_id": parent_id, "subject": "Re: synthetic",
                                     "in_reply_to_header": "<parent@synthetic.invalid>",
                                     "references_header": "<parent@synthetic.invalid>"}}

        def mk(i: int, idem: str, state: str, *, with_message: bool, reply: bool, claimed: bool = False):
            contact = Contact(campaign_id=camp.id, name=f"Synthetic {i}", email=f"synthetic{i}@example.invalid",
                              next_action_at=None)
            db.add(contact); db.flush()
            parent = Message(contact_id=contact.id, direction=MessageDirection.INBOUND.value,
                             message_type=MessageType.INITIAL.value, subject="synthetic inbound", body="synthetic",
                             status=MessageStatus.RECEIVED.value if hasattr(MessageStatus, "RECEIVED") else "received",
                             message_id_header=f"<parent{i}@synthetic.invalid>", mailbox_id=mbx.id)
            db.add(parent); db.flush()
            msg = None
            if with_message:
                msg = create_authorized_message(contact_id=contact.id, subject="Re: synthetic", body="synthetic body")
                db.add(msg); db.flush()
            d = ExternalDispatch(
                organization_id=org, idempotency_key=idem, external_action_id=str(1000 + i), correlation_id=f"c{i}",
                campaign_id=camp.id, contact_id=contact.id, message_id=msg.id if msg else None, mailbox_id=mbx.id,
                request_fingerprint="f" * 64, public_reference=uuid.uuid4().hex, state=state,
                claimed_by="old-worker-1" if claimed else None,
                grounding_context=reply_gc(parent.id) if reply else None,
                content_approval_required=False)  # LEGACY: accepted before approval-by-digest existed
            db.add(d); db.flush()
            out[idem] = d.id

        # (legacy replies: the states the README says block the gate, plus the ones it says do not)
        mk(1, "legacy-queued-nomsg", "queued", with_message=False, reply=True)
        mk(2, "legacy-queued-msg", "queued", with_message=True, reply=True)
        mk(3, "legacy-generating", "generating", with_message=False, reply=True, claimed=True)
        mk(4, "legacy-sending", "sending", with_message=True, reply=True, claimed=True)
        mk(5, "legacy-unknown", "unknown", with_message=True, reply=True)
        mk(6, "legacy-sent", "sent", with_message=True, reply=True)
        mk(7, "legacy-failed", "failed", with_message=True, reply=True)
        # harmless: ordinary (non-reply) dispatches, including one still queued
        mk(8, "ordinary-queued", "queued", with_message=True, reply=False)
        mk(9, "ordinary-sent", "sent", with_message=True, reply=False)
        db.commit()

        # synthetic volume on messages (for the index builds / backfill timings; labelled synthetic)
        ct = Contact(campaign_id=camp.id, name="Bulk", email="bulk@example.invalid", next_action_at=None)
        db.add(ct); db.commit()
        cid = ct.id
    if bulk:
        from sqlalchemy import text

        from mailer_agent.db import engine

        with engine.begin() as c:
            c.execute(text(
                "INSERT INTO messages (contact_id, direction, message_type, subject, body, status, mailbox_id, "
                "message_id_header, created_at) "
                "SELECT :cid, CASE WHEN g % 2 = 0 THEN 'inbound' ELSE 'outbound' END, 'initial', 'bulk ' || g, 'bulk', "
                "CASE WHEN g % 2 = 0 THEN 'received' ELSE 'sent' END, NULL, '<bulk' || g || '@synthetic.invalid>', "
                "now() - (g || ' seconds')::interval FROM generate_series(1, :n) g"), {"cid": cid, "n": bulk})
    print(json.dumps({"ids": out, "bulk": bulk}))
    return 0


def child_api_probe(mode: str) -> int:
    """Start the real Mailer API in-process (FastAPI TestClient runs the real lifespan) and probe it.

    mode ``startup``: only report whether startup succeeded.
    mode ``dispatch-query``: after startup, run the ORM query every dispatch path runs (selects the 013 columns).
    mode ``legacy``: read/cancel legacy replies through the real draft endpoints + operator metrics.
    Prints ONE JSON line. Never prints a row of data beyond states/status codes.
    """
    sys.path.insert(0, str(ROOT))
    res: dict = {"mode": mode, "schema_init_mode": os.environ.get("SCHEMA_INIT_MODE", "create")}
    try:
        from fastapi.testclient import TestClient

        from mailer_agent.api.main import app
    except Exception as exc:  # noqa: BLE001
        res.update(startup="import-failed", error=type(exc).__name__)
        print(json.dumps(res)); return 0
    try:
        with TestClient(app) as client:
            res["startup"] = "ok"
            hdr = {"X-API-Key": _tenant_key()}
            if mode == "dispatch-query":
                from mailer_agent.db import SessionLocal
                from mailer_agent.models import ExternalDispatch

                try:
                    with SessionLocal() as db:
                        res["dispatch_query_rows"] = db.query(ExternalDispatch).count()
                    res["dispatch_query"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    res["dispatch_query"] = "failed"
                    res["error"] = type(exc).__name__ + ": " + str(exc).splitlines()[0][:160]
            if mode == "legacy":
                keys = json.loads(os.environ["REHEARSAL_KEYS"])
                for label in ("legacy-unknown", "legacy-failed"):
                    k = keys[label]
                    g = client.get(f"/integrations/leadboost/reply-drafts/{k}", headers=hdr)
                    p = client.post(f"/integrations/leadboost/reply-drafts/{k}/cancel", headers=hdr)
                    res[label] = {"get": g.status_code, "get_detail": g.json().get("detail") if g.status_code != 200 else None,
                                  "cancel": p.status_code, "cancel_detail": p.json().get("detail") if p.status_code != 200 else None}
                oh = {"X-System-Operator-Key": os.environ["SYSTEM_OPERATOR_API_KEY"]}
                for path in ("dispatch", "worker", "inbound"):
                    r = client.get(f"/system/metrics/{path}", headers=oh)
                    res[f"metrics_{path}"] = r.status_code
    except Exception as exc:  # noqa: BLE001
        res.setdefault("startup", "failed")
        res["error"] = type(exc).__name__ + ": " + str(exc).splitlines()[0][:300]
    print(json.dumps(res))
    return 0


def _tenant_key() -> str:
    return next(iter(json.loads(os.environ["ORG_KEY_MAP"])))


def run_child(admin: dict, dbname: str, sub: list[str], extra_env: dict | None = None, timeout: int = 900):
    env = child_env(admin, dbname, extra_env)
    return sh([python_exe(), str(Path(__file__).resolve()), *sub], env, timeout=timeout)


def last_json(text: str) -> dict:
    for ln in reversed(text.splitlines()):
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return json.loads(ln)
            except ValueError:
                continue
    return {}


# --------------------------------------------------------------------------------------------------------------
# The scenario
# --------------------------------------------------------------------------------------------------------------

def preflight(admin: dict, dbname: str, sub: str, apply: bool = False, extra: dict | None = None):
    args = [python_exe(), "scripts/migrate_preflight.py", sub] + (["--apply"] if apply else [])
    return sh(args, child_env(admin, dbname, extra), timeout=300)


def wrapper(admin: dict, dbname: str, mode: str | None, ack: bool = False, lock_ms: int | None = None):
    extra = {}
    if ack:
        extra["MIGRATE_CONFIRM_BACKUP"] = f"I_HAVE_A_VERIFIED_BACKUP_OF_{dbname}"
    if lock_ms:
        extra["MIGRATE_LOCK_TIMEOUT_MS"] = str(lock_ms)
    args = ["bash", "scripts/migrate_release.sh"] + ([mode] if mode else [])
    return sh(args, child_env(admin, dbname, extra), timeout=3600)


def marker_summary(text: str) -> tuple[int, int]:
    present = len(re.findall(r"\[present\]", text))
    absent = len(re.findall(r"\[ABSENT \]", text))
    return present, absent


def run_scenario(admin: dict) -> int:
    init_synthetic_secrets()
    readme = (ROOT / "migrations" / "README.md").read_text(encoding="utf-8")
    pre013_sql = extract_readme_sql(readme, "legacy-abandon-sql:pre-013")
    post013_sql = extract_readme_sql(readme, "legacy-abandon-sql:post-013")
    if (ROOT / ".env").exists():
        raise Refused("refusing: a .env file exists in the repository root; Settings would read it. Use a fresh worktree.")

    main_db = make_db_name("mailer")
    T.step(f"S0  create throw-away database {main_db} on {GUARD_HOST}:{GUARD_PORT} (synthetic data only)")
    create_db(admin, main_db)
    ver = q(admin, main_db, "SHOW server_version")[0][0]
    T.say(f"  PostgreSQL {ver}")

    # ---- S1 build + seed + rewind ---------------------------------------------------------------------------
    T.step("S1  build the schema, seed synthetic rows, rewind to the hosted state (after the 11-step run)")
    rc, out, dt = preflight(admin, main_db, "init")
    T.expect("init_db() on the empty database", rc == 0, f"rc={rc} {dt:.1f}s")
    for step in STEPS_001_009:
        rc, out, _ = sh([python_exe(), step], child_env(admin, main_db), timeout=300)
        if not T.expect(f"apply {step}", rc == 0, f"rc={rc}"):
            T.say("  " + tail(out)); return EXIT_EXPECT_FAILED
    rc, out, dt = run_child(admin, main_db, ["child-seed"])
    seeded = last_json(out)
    if not T.expect("seed synthetic rows (current schema, flag FALSE = legacy)", rc == 0 and bool(seeded.get("ids")), f"rc={rc} {dt:.1f}s"):
        T.say("  " + tail(out)); return EXIT_EXPECT_FAILED
    ids = seeded["ids"]
    T.say(f"  seeded {len(ids)} dispatches and {seeded['bulk']} bulk messages (all synthetic)")
    T.say("  NOTE: init_db() builds tables from the CURRENT models, so a fresh database already has the 010-013 objects.")
    T.say("  The rewind below uses the repository's own --downgrade scripts to recreate the state of the hosted database")
    T.say("  (11 steps applied). It is a rehearsal device, NOT part of the real procedure.")
    for step in REWIND:
        rc, out, _ = sh([python_exe(), step, "--downgrade"], child_env(admin, main_db), timeout=300)
        if not T.expect(f"rewind {step} --downgrade", rc == 0, f"rc={rc}"):
            T.say("  " + tail(out)); return EXIT_EXPECT_FAILED
    rc, out, _ = preflight(admin, main_db, "status")
    pres, absn = marker_summary(out)
    T.expect("--status: all 010-013 markers ABSENT, the rest present", absn == 8 and pres == 13, f"present={pres} absent={absn}")
    counts_before = table_counts(admin, main_db)
    T.say("  table counts before (counts only): " + json.dumps({k: v for k, v in counts_before.items() if v}))

    # ---- S2 wrong order: new code started before the migration (on a COPY) -----------------------------------
    T.step("S2  WRONG ORDER on a copy: new API code started before the migration")
    wrong_db = make_db_name("mailer_wrongorder")
    create_db(admin, wrong_db, template=main_db)
    for mode in ("verify", "create"):
        rc, out, dt = run_child(admin, wrong_db, ["child-api-probe", "dispatch-query"], {"SCHEMA_INIT_MODE": mode})
        r = last_json(out)
        if mode == "verify":
            T.expect("SCHEMA_INIT_MODE=verify refuses to start on the pre-013 schema",
                     r.get("startup") == "failed" and "not ready" in r.get("error", "").lower(), r.get("error", "")[:140])
        else:
            T.expect("SCHEMA_INIT_MODE=create (the deployed default) STARTS...", r.get("startup") == "ok", str(r.get("startup")))
            T.expect("...but the first dispatch query then FAILS (column missing)", r.get("dispatch_query") == "failed",
                     r.get("error", "")[:140])
    rc, out, _ = preflight(admin, wrong_db, "status")
    pres, absn = marker_summary(out)
    T.say(f"  after the create-mode start the copy shows present={pres} absent={absn} (init_db() created missing TABLES only)")
    T.expect("create-mode start did not add the 013 columns (create_all never alters)",
             "[ABSENT ] 013  external_dispatches.content_approval_required" in out, "")

    # ---- S3 the gate ----------------------------------------------------------------------------------------
    T.step("S3  migration-013 gate on the seeded legacy replies (read-only)")
    rc, out, _ = preflight(admin, main_db, "gate")
    T.expect("gate exits 4 while unresolved legacy managed replies exist", rc == 4, f"rc={rc}")
    blocking = re.findall(r"id=(\d+) org=\S+ key=(\S+) state=(\S+)", out)
    T.say("  blocking rows (id key state): " + "; ".join(f"{i} {k} {s}" for i, k, s in blocking))
    T.expect("exactly the four blocking states are listed (queued, queued+message, generating, sending+message)",
             sorted(s for _i, _k, s in blocking) == ["generating", "queued", "queued+message", "sending+message"],
             str(sorted(s for _i, _k, s in blocking)))
    T.expect("ordinary queued dispatch and finished/unknown legacy rows do not block",
             not any(k.startswith(("ordinary", "legacy-sent", "legacy-failed", "legacy-unknown")) for _i, k, _s in blocking), "")
    T.expect("gate reports the legacy 'unknown' row as informational", "state 'unknown'" in out, "")
    # The runbook's read-only "do any legacy managed replies exist?" query (works for json AND jsonb columns).
    count_sql = LEGACY_COUNT_SQL
    got = {(st, hm): n for st, hm, n in q(admin, main_db, count_sql)}
    T.expect("runbook count query reports every legacy reply by state (read-only SELECT)",
             got == {("queued", False): 1, ("queued", True): 1, ("generating", False): 1, ("sending", True): 1,
                     ("unknown", True): 1, ("sent", True): 1, ("failed", True): 1}, str(got))
    rc2, out2, _ = wrapper(admin, main_db, "--check-only")
    T.expect("release wrapper --check-only also exits 4 (the gate is its first step)", rc2 == 4, f"rc={rc2}")

    # ---- S4 abandon ONE row with the README snippet ---------------------------------------------------------
    T.step("S4  abandon ONE unclaimed queued row with the README's pre-013 statement")
    q_id = ids["legacy-queued-nomsg"]
    got = q(admin, main_db, pre013_sql.replace(":dispatch_id", str(int(q_id))))
    T.expect("snippet returned exactly one row (id, 'failed')", len(got) == 1 and got[0][1] == "failed", str(got))
    got2 = q(admin, main_db, pre013_sql.replace(":dispatch_id", str(int(ids["legacy-sending"]))))
    T.expect("the same statement does nothing to a 'sending' row (guarded on state/claim)", len(got2) == 0, str(got2))
    rc, out, _ = preflight(admin, main_db, "gate")
    T.expect("gate still exits 4 for the remaining rows", rc == 4, f"rc={rc}")

    # ---- S5 resolve the rest (simulating the OLD workers) ---------------------------------------------------
    T.step("S5  resolve the rest the way the README says (old workers drain; lease sweeps) and re-run the gate")
    T.say("  SIMULATED (we have no old workers here): generating -> lease sweep returns it to queued -> abandoned;")
    T.say("  queued+message -> abandoned; sending -> the old SMTP worker's lease sweep resolves it to 'unknown'.")
    q(admin, main_db, "UPDATE external_dispatches SET state='queued', claimed_by=NULL, claimed_at=NULL WHERE id=:i",
      {"i": ids["legacy-generating"]})
    for k in ("legacy-generating", "legacy-queued-msg"):
        got = q(admin, main_db, pre013_sql.replace(":dispatch_id", str(int(ids[k]))))
        T.expect(f"abandon {k}", len(got) == 1, str(got))
    q(admin, main_db, "UPDATE external_dispatches SET state='unknown', claimed_by=NULL WHERE id=:i", {"i": ids["legacy-sending"]})
    rc, out, _ = preflight(admin, main_db, "gate")
    T.expect("gate exits 0 now (two legacy rows are 'unknown': reported, not blocking)", rc == 0, f"rc={rc}")
    T.expect("gate output notes the unknown legacy replies", "'unknown'" in out, "")

    # ---- S6 the wrong-order copy continues: wrapper after a create-mode start --------------------------------
    T.step("S6  the wrapper on the wrong-order copy (after the create-mode start) - does it recover?")
    q(admin, wrong_db, "UPDATE external_dispatches SET state='queued', claimed_by=NULL, claimed_at=NULL WHERE id=:i",
      {"i": ids["legacy-generating"]})
    for k in ("legacy-queued-nomsg", "legacy-queued-msg", "legacy-generating"):
        q(admin, wrong_db, pre013_sql.replace(":dispatch_id", str(int(ids[k]))))
    q(admin, wrong_db, "UPDATE external_dispatches SET state='unknown', claimed_by=NULL WHERE id=:i", {"i": ids["legacy-sending"]})
    rc, out, dt = wrapper(admin, wrong_db, None, ack=True, lock_ms=5000)
    T.expect("wrapper applies cleanly on the copy that new code had already started against", rc == 0, f"rc={rc} {dt:.1f}s")
    if rc != 0:
        T.say("  " + tail(out, 10))

    # ---- S7 the real release on the main database -----------------------------------------------------------
    T.step("S7  release wrapper: apply 010 -> 013 (with MIGRATE_LOCK_TIMEOUT_MS=5000)")
    rc, out, _ = wrapper(admin, main_db, None, ack=False)
    T.expect("apply WITHOUT the backup acknowledgement is refused (exit 2)", rc == 2, f"rc={rc}")
    rc, out, dt = wrapper(admin, main_db, None, ack=True, lock_ms=5000)
    T.expect("apply exits 0 and reports all 15 steps", rc == 0 and "all 15 steps completed" in out, f"rc={rc} {dt:.1f}s")
    step_lines = re.findall(r"\[migrate\] \((\d+)/15\) (\S+)", out)
    T.say(f"  steps run: {len(step_lines)}; whole wrapper took {dt:.1f}s on {seeded['bulk']} synthetic messages "
          "(a laptop/sandbox number: it says nothing about the real table)")
    if rc != 0:
        T.say("  " + tail(out, 12)); return EXIT_EXPECT_FAILED
    rc, out, _ = wrapper(admin, main_db, "--status")
    pres, absn = marker_summary(out)
    T.expect("--status after the release: every marker present", absn == 0 and pres == 21, f"present={pres} absent={absn}")
    rc, out, _ = preflight(admin, main_db, "verify")
    T.expect("verify (the SCHEMA_INIT_MODE=verify dry run) exits 0", rc == 0, f"rc={rc}")
    rc, out, _ = wrapper(admin, main_db, None, ack=True, lock_ms=5000)
    T.expect("re-running the whole wrapper is a no-op (idempotent), exit 0", rc == 0, f"rc={rc}")

    # ---- S8 facts about 013 ---------------------------------------------------------------------------------
    T.step("S8  what migration 013 (and 010-012) really did")
    cols = {r[0]: (r[1], r[2], r[3]) for r in q(admin, main_db, "SELECT column_name, data_type, is_nullable, column_default "
            "FROM information_schema.columns WHERE table_name='external_dispatches'")}
    T.expect("content_approval_required BOOLEAN NOT NULL DEFAULT false",
             cols.get("content_approval_required", ("",) * 3)[0] == "boolean" and cols["content_approval_required"][1] == "NO"
             and "false" in str(cols["content_approval_required"][2]), str(cols.get("content_approval_required")))
    T.expect("approved_content_sha256 nullable, approved_at nullable",
             cols.get("approved_content_sha256", ("", "NO", ""))[1] == "YES" and cols.get("approved_at", ("", "NO", ""))[1] == "YES", "")
    flags = q(admin, main_db, "SELECT content_approval_required, count(*) FROM external_dispatches GROUP BY 1")
    T.expect("every pre-existing dispatch got FALSE (nothing starts requiring approval)", flags == [(False, len(ids))], str(flags))
    nullref = q(admin, main_db, "SELECT count(*) FROM messages WHERE reply_reference IS NULL")[0][0]
    T.expect("012 backfilled a reply_reference on every message", nullref == 0, f"null={nullref}")
    invalid = q(admin, main_db, "SELECT count(*) FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid "
                "WHERE c.relname IN ('ix_messages_inbound_created_at','uq_messages_reply_reference') AND NOT i.indisvalid")[0][0]
    T.expect("the two new indexes on messages are VALID", invalid == 0, f"invalid={invalid}")
    counts_after = table_counts(admin, main_db)
    same = {k: (counts_before.get(k), v) for k, v in counts_after.items() if counts_before.get(k) not in (None, v)}
    T.expect("row counts of every pre-existing table unchanged by the release", not same, str(same))
    T.say("  table counts after (counts only): " + json.dumps({k: v for k, v in counts_after.items() if v}))

    # ---- S9 behaviour after 013 -----------------------------------------------------------------------------
    T.step("S9  real API (in-process) on the migrated database: legacy replies and operator endpoints")
    rc, out, dt = run_child(admin, main_db, ["child-api-probe", "legacy"],
                            {"SCHEMA_INIT_MODE": "verify", "REHEARSAL_KEYS": json.dumps({k: k for k in ids})})
    r = last_json(out)
    T.expect("API starts in SCHEMA_INIT_MODE=verify after the migration", r.get("startup") == "ok", str(r.get("error", r.get("startup"))))
    u = r.get("legacy-unknown", {}); f = r.get("legacy-failed", {})
    T.expect("legacy 'unknown' reply: read and cancel answer 409 legacy_reply_requires_reconciliation",
             (u.get("get"), u.get("get_detail"), u.get("cancel"), u.get("cancel_detail")) == (409, "legacy_reply_requires_reconciliation", 409, "legacy_reply_requires_reconciliation"), str(u))
    T.expect("legacy 'failed' (abandoned) reply: read 409, cancel 200 (nothing was or can be sent)",
             (f.get("get"), f.get("cancel")) == (409, 200), str(f))
    T.expect("operator endpoints answer 200 (dispatch, worker, inbound)",
             (r.get("metrics_dispatch"), r.get("metrics_worker"), r.get("metrics_inbound")) == (200, 200, 200),
             f"{r.get('metrics_dispatch')},{r.get('metrics_worker')},{r.get('metrics_inbound')}")
    got = q(admin, main_db, post013_sql.replace(":dispatch_id", str(int(ids["legacy-failed"]))))
    T.expect("README post-013 abandon statement is valid SQL on the migrated schema (0 rows: not queued)", len(got) == 0, str(got))

    # ---- S10 downgrade of 013 -------------------------------------------------------------------------------
    T.step("S10 migration 013 --downgrade: refusal first, then success; what it destroys")
    q(admin, main_db, "INSERT INTO external_dispatches (organization_id, idempotency_key, campaign_id, contact_id, message_id, "
      "mailbox_id, request_fingerprint, public_reference, state, content_approval_required, approved_content_sha256, approved_at, created_at, updated_at) "
      "SELECT organization_id, 'synthetic-held', campaign_id, contact_id, message_id, mailbox_id, request_fingerprint, 'synthheld' || md5(random()::text), "
      "'awaiting_approval', true, NULL, NULL, now(), now() FROM external_dispatches WHERE idempotency_key='ordinary-sent'")
    rc, out, _ = sh([python_exe(), "migrations/013_external_dispatch_content_approval.py", "--downgrade"], child_env(admin, main_db))
    T.expect("downgrade REFUSES (exit 1) while a held managed-reply draft depends on the columns",
             rc == 1 and "REFUSING downgrade" in out, f"rc={rc}")
    have = q(admin, main_db, "SELECT count(*) FROM information_schema.columns WHERE table_name='external_dispatches' AND column_name='content_approval_required'")[0][0]
    T.expect("a refused downgrade changed nothing", have == 1, f"column present={have}")
    q(admin, main_db, "UPDATE external_dispatches SET state='sent', approved_content_sha256=repeat('a',64), approved_at=now() WHERE idempotency_key='synthetic-held'")
    ev = q(admin, main_db, "SELECT count(*) FROM external_dispatches WHERE approved_content_sha256 IS NOT NULL")[0][0]
    rc, out, _ = sh([python_exe(), "migrations/013_external_dispatch_content_approval.py", "--downgrade"], child_env(admin, main_db))
    T.expect("downgrade succeeds once the managed reply is finished", rc == 0, f"rc={rc}")
    T.say(f"  it DROPPED the columns and with them {ev} recorded approval digest(s): the approval evidence is gone")
    rc, out, _ = sh([python_exe(), "migrations/013_external_dispatch_content_approval.py"], child_env(admin, main_db))
    T.expect("re-applying 013 works (idempotent)", rc == 0, f"rc={rc}")
    gone = q(admin, main_db, "SELECT count(*) FROM external_dispatches WHERE approved_content_sha256 IS NOT NULL")[0][0]
    T.expect("after downgrade + re-apply the digest is NOT restored (documented loss)", gone == 0, f"digests={gone}")

    # ---- S11 restore drill (A5, Mailer, synthetic) -----------------------------------------------------------
    T.step("S11 restore drill on synthetic data: dump the migrated database, restore into a new one, check it")
    dump = state_path().parent / "mailer_rehearsal.dump"
    cmd, env = pg_client_cmd(admin, "pg_dump", main_db, ["-Fc", "--no-owner", "--no-privileges"])
    rc, _o, dt_dump = sh(cmd, env, stdout_path=dump)
    T.expect("pg_dump -Fc of the migrated synthetic database", rc == 0 and dump.exists() and dump.stat().st_size > 0, f"rc={rc} {dt_dump:.1f}s {dump.stat().st_size if dump.exists() else 0} bytes")
    rest_db = make_db_name("mailer_restored")
    create_db(admin, rest_db)
    cmd, env = pg_client_cmd(admin, "pg_restore", rest_db, ["--no-owner", "--no-privileges", "--exit-on-error"])
    rc, o, dt_rest = sh(cmd, env, stdin_path=dump)
    T.expect("pg_restore into a new rehearsal database", rc == 0, f"rc={rc} {dt_rest:.1f}s " + tail(o, 2))
    cnt_r = table_counts(admin, rest_db)
    cnt_m = table_counts(admin, main_db)
    T.expect("restored row counts equal the source", cnt_r == cnt_m, "" if cnt_r == cnt_m else str({k: (cnt_m.get(k), cnt_r.get(k)) for k in set(cnt_m) | set(cnt_r) if cnt_m.get(k) != cnt_r.get(k)}))
    rc, out, _ = wrapper(admin, rest_db, "--status")
    pres, absn = marker_summary(out)
    T.expect("restored database: every migration marker present", absn == 0 and pres == 21, f"present={pres} absent={absn}")
    rc, out, _ = preflight(admin, rest_db, "gate")
    T.expect("restored database: the gate passes (exit 0)", rc == 0, f"rc={rc}")
    rc, out, dt = run_child(admin, rest_db, ["child-api-probe", "startup"], {"SCHEMA_INIT_MODE": "verify"})
    T.expect("restored database: the API starts in SCHEMA_INIT_MODE=verify", last_json(out).get("startup") == "ok", f"{dt:.1f}s")
    T.say(f"  restore time on this machine: dump {dt_dump:.1f}s + restore {dt_rest:.1f}s for a synthetic database with "
          f"{seeded['bulk']} messages (an upper-bound-free laptop measurement; the real database is a different size)")
    try:
        dump.unlink()
    except OSError:
        pass

    T.step("SUMMARY")
    failed = [(n, g) for n, ok, g in T.results if not ok]
    T.say(f"  expectations: {len(T.results)}, failed: {len(failed)}")
    for n, g in failed:
        T.say(f"  FAILED: {n}  {g}")
    (state_path().parent / "rehearsal_mailer_results.json").write_text(json.dumps(
        {"databases": load_state()["databases"], "results": [{"name": n, "ok": ok, "got": g} for n, ok, g in T.results]}, indent=2))
    T.say("  databases created (drop them with:  python scripts/rehearse_c95_migrations.py cleanup):")
    for d in load_state()["databases"]:
        T.say(f"    {d}")
    return EXIT_OK if T.ok else EXIT_EXPECT_FAILED


def cmd_cleanup(admin: dict) -> int:
    st = load_state()
    for name in list(st["databases"]):
        assert_rehearsal_db_name(name)
        drop_db(admin, name)
        print(f"dropped {name}")
    st["databases"] = []
    state_path().write_text(json.dumps(st, indent=2))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("run", "status", "cleanup"):
        sub.add_parser(n)
    sub.add_parser("child-seed")
    cp = sub.add_parser("child-api-probe"); cp.add_argument("mode")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "child-seed":
            return child_seed()
        if args.cmd == "child-api-probe":
            return child_api_probe(args.mode)
        admin = parse_admin_url(os.environ.get("REHEARSAL_ADMIN_URL", ""))
        if args.cmd == "status":
            print(json.dumps(load_state(), indent=2)); return EXIT_OK
        if args.cmd == "cleanup":
            return cmd_cleanup(admin)
        return run_scenario(admin)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
