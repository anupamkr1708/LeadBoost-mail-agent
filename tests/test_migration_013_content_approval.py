"""
Migration 013 (external_dispatches content-approval columns): additive, idempotent, safe defaults, and a
downgrade that refuses to silently strip an approval requirement. Runs on a throwaway SQLite database whose
schema is first rewound to its pre-013 shape, with an existing row, so "existing rows keep their meaning" is
tested against a real legacy row rather than assumed. (The PostgreSQL DDL branch is exercised by the
postgres-marked release-wrapper test, which runs the whole canonical sequence.)
"""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from mailer_agent.models import Base
from tests.dispatch_support import seed_dispatch

_PATH = Path(__file__).resolve().parent.parent / "migrations" / "013_external_dispatch_content_approval.py"
NEW_COLUMNS = ("content_approval_required", "approved_content_sha256", "approved_at")


def _load_migration():
    spec = importlib.util.spec_from_file_location("migration_013", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def legacy(tmp_path, monkeypatch):
    """A SQLite database in its pre-013 shape holding one ordinary QUEUED dispatch."""
    engine = create_engine(f"sqlite:///{tmp_path / 'm013.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as db:
        seed_dispatch(db, idem="legacy-row", state="queued")
    with engine.begin() as conn:
        for name in NEW_COLUMNS:
            conn.execute(text(f"ALTER TABLE external_dispatches DROP COLUMN {name}"))
    assert not set(NEW_COLUMNS) & {c["name"] for c in inspect(engine).get_columns("external_dispatches")}

    mod = _load_migration()

    @contextmanager
    def scope():
        db = Session()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    monkeypatch.setattr(mod, "session_scope", scope)
    yield mod, engine, Session
    engine.dispose()


def _cols(engine):
    return {c["name"]: c for c in inspect(engine).get_columns("external_dispatches")}


def test_upgrade_adds_columns_and_existing_rows_keep_ordinary_semantics(legacy):
    mod, engine, _ = legacy
    assert mod.run() is True
    cols = _cols(engine)
    assert set(NEW_COLUMNS) <= set(cols)
    assert cols["content_approval_required"]["nullable"] is False
    assert cols["approved_content_sha256"]["nullable"] is True
    with engine.connect() as c:
        row = c.execute(
            text("SELECT state, content_approval_required, approved_content_sha256, approved_at "
                 "FROM external_dispatches WHERE idempotency_key='legacy-row'")
        ).one()
    assert row[0] == "queued"
    assert not row[1], "an existing row must NOT start requiring approval (its behaviour is unchanged)"
    assert row[2] is None and row[3] is None


def test_upgrade_is_idempotent(legacy):
    mod, engine, _ = legacy
    assert mod.run() is True
    assert mod.run() is True
    assert set(NEW_COLUMNS) <= set(_cols(engine))


def test_verify_reports_missing_columns(legacy):
    mod, _, Session = legacy
    with Session() as db:
        assert mod.verify(db) is False


def test_downgrade_round_trip_when_nothing_depends_on_the_flag(legacy):
    mod, engine, _ = legacy
    assert mod.run() is True
    assert mod.downgrade() is True
    assert not set(NEW_COLUMNS) & set(_cols(engine))
    assert mod.run() is True  # re-apply works


@pytest.mark.parametrize("state", ["queued", "generating", "awaiting_approval", "sending"])
def test_downgrade_refuses_while_a_row_depends_on_the_approval_requirement(legacy, state):
    mod, engine, Session = legacy
    assert mod.run() is True
    with Session() as db:
        seed_dispatch(db, idem="held", email="held@example.com", state=state, content_approval_required=True)
    assert mod.downgrade() is False
    assert set(NEW_COLUMNS) <= set(_cols(engine)), "a refused downgrade must change nothing"


@pytest.mark.parametrize("state", ["sent", "failed", "unknown", "cancelled"])
def test_downgrade_is_not_blocked_by_finished_rows_even_when_they_carry_approval_evidence(legacy, state):
    """Terminal rows no longer depend on the flag: the downgrade proceeds (and drops their recorded evidence, which is
    documented), so it is not held hostage by history."""
    mod, engine, Session = legacy
    assert mod.run() is True
    with Session() as db:
        seed_dispatch(db, idem="done", email="done@example.com", state=state, content_approval_required=True,
                      approved_content_sha256="a" * 64)
    assert mod.downgrade() is True
    assert not set(NEW_COLUMNS) & set(_cols(engine))
