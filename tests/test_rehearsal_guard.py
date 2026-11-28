"""Safety guard of scripts/rehearse_c95_migrations.py (pure logic and refusals; no database needed).

The rehearsal harness must be unable to touch anything but a throw-away ``c95_rehearsal_*`` database on
127.0.0.1:56432. These tests pin that, plus the two README statements the harness executes verbatim.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "rehearse_c95_migrations.py"
README = ROOT / "migrations" / "README.md"

_spec = importlib.util.spec_from_file_location("rehearse_c95_migrations", SCRIPT)
rh = importlib.util.module_from_spec(_spec)
sys.modules["rehearse_c95_migrations"] = rh
_spec.loader.exec_module(rh)


@pytest.mark.parametrize("host,port", [("db.example.com", 56432), ("127.0.0.1", 5432), ("localhost", 56432), (None, None), ("127.0.0.1", None)])
def test_server_other_than_the_throwaway_container_is_refused(host, port):
    with pytest.raises(rh.Refused):
        rh.assert_rehearsal_server(host, port)


def test_the_throwaway_server_is_accepted():
    rh.assert_rehearsal_server("127.0.0.1", 56432)


@pytest.mark.parametrize("name", [
    "mailer_agent", "postgres", "c95_rehearsal_", "C95_REHEARSAL_x", "x_c95_rehearsal_y", "c95_rehearsal_a-b",
    'c95_rehearsal_x"; DROP DATABASE postgres; --', "c95_rehearsal_" + "a" * 60, "", None,
])
def test_database_names_outside_the_prefix_are_refused(name):
    with pytest.raises(rh.Refused):
        rh.assert_rehearsal_db_name(name)


def test_generated_names_are_valid_and_prefixed():
    name = rh.make_db_name("Mailer wrong-order!", datetime(2026, 11, 24, 12, 30, 5, tzinfo=timezone.utc))
    assert name == "c95_rehearsal_mailer_wrong_order_20261124t123005z"
    rh.assert_rehearsal_db_name(name)


@pytest.mark.parametrize("url", [
    "", "mysql://u:p@127.0.0.1:56432/postgres", "postgresql://u:p@prod.supabase.example:5432/postgres",
    "postgresql://u:p@127.0.0.1:5432/postgres",
])
def test_admin_url_must_point_at_the_throwaway_server(url):
    with pytest.raises(rh.Refused):
        rh.parse_admin_url(url)


def test_admin_url_accepts_the_throwaway_server_and_db_urls_are_guarded():
    admin = rh.parse_admin_url("postgresql+psycopg2://postgres:p%40ss@127.0.0.1:56432/postgres")
    assert (admin["host"], admin["port"], admin["password"]) == ("127.0.0.1", 56432, "p@ss")
    assert rh.db_url(admin, "c95_rehearsal_ok").endswith("/c95_rehearsal_ok")
    with pytest.raises(rh.Refused):
        rh.db_url(admin, "mailer_agent")


def test_children_never_receive_the_callers_secrets(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real_looking")
    monkeypatch.setenv("SMTP_PASSWORD", "real-smtp-password")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:REALPW@prod.example:5432/prod")
    rh.init_synthetic_secrets()
    admin = rh.parse_admin_url("postgresql://postgres:pw@127.0.0.1:56432/postgres")
    env = rh.child_env(admin, "c95_rehearsal_x")
    assert "GROQ_API_KEY" not in env and "SMTP_PASSWORD" not in env
    assert env["DATABASE_URL"].endswith("@127.0.0.1:56432/c95_rehearsal_x")
    assert env["LIVE_SENDING_ENABLED"] == "false" and env["AUTO_REPLY_ENABLED"] == "false"
    assert "REALPW" not in "".join(env.values())


def test_readme_snippets_the_harness_executes_exist_and_are_guarded_updates():
    text = README.read_text(encoding="utf-8")
    for marker in ("legacy-abandon-sql:pre-013", "legacy-abandon-sql:post-013"):
        sql = rh.extract_readme_sql(text, marker)
        assert sql.upper().startswith("-- ") or sql.upper().lstrip().startswith(("UPDATE", "--"))
        assert ":dispatch_id" in sql and "state = 'queued'" in sql and "claimed_by IS NULL" in sql
    with pytest.raises(rh.Refused):
        rh.extract_readme_sql(text, "no-such-marker")


def _run(env_extra: dict, *args: str) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ.get("PATH", "")}
    env.update(env_extra)
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("url", ["postgresql://u:SECRETPW@db.example.com:5432/postgres", "postgresql://u:SECRETPW@127.0.0.1:5432/postgres"])
def test_cli_refuses_a_non_throwaway_server_before_connecting(url, tmp_path):
    p = _run({"REHEARSAL_ADMIN_URL": url, "REHEARSAL_STATE_DIR": str(tmp_path)}, "run")
    assert p.returncode == 2, p.stdout + p.stderr
    assert "REFUSED" in p.stderr and "SECRETPW" not in p.stdout + p.stderr


def test_cli_refuses_without_admin_url_or_state_dir(tmp_path):
    assert _run({}, "run").returncode == 2
    p = _run({"REHEARSAL_ADMIN_URL": "postgresql://postgres:x@127.0.0.1:56432/postgres"}, "cleanup")
    assert p.returncode == 2 and "REHEARSAL_STATE_DIR" in p.stderr


def _fake_docker(tmp_path, monkeypatch, published: str, rc: int = 0):
    """A `docker` on PATH that only answers `docker port <container> 5432/tcp`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "docker"
    script.write_text(f'#!/bin/sh\nif [ "$1" = port ]; then printf \'%s\\n\' "{published}"; exit {rc}; fi\nexit 9\n')
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    rh._CONTAINER_VERIFIED.clear()


@pytest.mark.parametrize(
    "published,rc,accepted",
    [
        ("127.0.0.1:56432", 0, True),
        ("127.0.0.1:56433", 0, False),  # another throw-away server: not the validated endpoint
        ("", 0, False),
        ("0.0.0.0:56432", 0, False),  # not loopback-only
        ("127.0.0.1:56432", 1, False),  # docker itself failed
    ],
)
def test_docker_client_container_must_serve_the_validated_endpoint(tmp_path, monkeypatch, published, rc, accepted):
    _fake_docker(tmp_path, monkeypatch, published, rc)
    monkeypatch.setenv("REHEARSAL_PG_CLIENT", "docker:c95-pg16")
    admin = rh.parse_admin_url("postgresql://postgres:pw@127.0.0.1:56432/postgres")
    if accepted:
        cmd, env = rh.pg_client_cmd(admin, "pg_dump", "c95_rehearsal_x", ["-Fc"])
        assert cmd[:7] == ["docker", "exec", "-i", "-e", "PGPASSWORD", "c95-pg16", "pg_dump"]
        assert env["PGPASSWORD"] == "pw"
    else:
        with pytest.raises(rh.Refused, match="must be the guarded server|cannot ask docker"):
            rh.pg_client_cmd(admin, "pg_dump", "c95_rehearsal_x", ["-Fc"])


def test_docker_client_with_a_non_c95_name_is_still_refused_before_docker_is_asked(tmp_path, monkeypatch):
    _fake_docker(tmp_path, monkeypatch, "127.0.0.1:56432")
    monkeypatch.setenv("REHEARSAL_PG_CLIENT", "docker:my-production-db")
    admin = rh.parse_admin_url("postgresql://postgres:pw@127.0.0.1:56432/postgres")
    with pytest.raises(rh.Refused, match="named c95-"):
        rh.pg_client_cmd(admin, "pg_dump", "c95_rehearsal_x", [])
