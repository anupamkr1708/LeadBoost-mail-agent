"""
Tests for the exact-message execution boundary (Phase C, C5).

Proves, deterministically (no SMTP, no LLM, no network):

  EXACTNESS          subject/body survive every hand-off unchanged, including
                     through the real send_email() up to the SMTP boundary;
                     an absent subject is never invented.
  WRONG TARGET       the boundary refuses inbound / already-final / mis-wired
                     messages.
  NO DUPLICATE       ONE operation -> ONE Message, correctly associated, and
  + PRE-SEND STATE   still pre-send (DRAFT / QUEUED).
  PURITY             the boundary functions never write through a session.
  ISOLATION          no LLM / drafting / SMTP on this path: allowlist + AST
                     (incl. lazy/relative/dynamic imports) + fresh interpreter
                     + behavioral spies.
  GROUNDING          the existing validate_grounding, hard-block-only,
                     read-only, never edits the message.
  TENANCY/IDEMPOTENCY  behavior from C2-C4 is unchanged (regression).

Fixture style follows tests/test_leadboost_integration.py (per-test
in-memory SQLite + dependency_overrides). The static import checks follow
the AST convention of tests/test_architecture.py; tests/test_architecture.py
also independently applies its SMTP/vendor-SDK rules to exact_message.py.
"""

from __future__ import annotations

import ast
import email.header
import inspect
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from mailer_agent.api import integrations as integrations_module
from mailer_agent.api.deps import get_current_org_id, require_api_key
from mailer_agent.api.main import app
from mailer_agent.config import get_settings
from mailer_agent.db import get_db
from mailer_agent.llm.grounding import validate_grounding
from mailer_agent.mail import exact_message as em
from mailer_agent.mail.exact_message import (
    ExactMessageError,
    build_exact_send_input,
    create_authorized_message,
    evaluate_exact_message_grounding,
)
from mailer_agent.mail.sender import send_email
from mailer_agent.memory.store import build_conversation_context
from mailer_agent.models import (
    Base,
    Campaign,
    Contact,
    ExternalDispatch,
    ExternalDispatchState,
    Message,
    MessageDirection,
    MessageStatus,
)

ROOT = Path(__file__).resolve().parent.parent
EXACT_MESSAGE_PY = ROOT / "mailer_agent" / "mail" / "exact_message.py"
INTEGRATIONS_PY = ROOT / "mailer_agent" / "api" / "integrations.py"

from tests.dispatch_support import seed_org_mailboxes

ORG_A, ORG_B = "org-a", "org-b"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _sender_identity(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "leadboost_integration_sender_email", "outreach@mailer.example.com")
    monkeypatch.setattr(s, "leadboost_integration_sender_name", "Test Sender")
    monkeypatch.setattr(s, "leadboost_integration_sender_org", "Test Org")


class _Org:
    def __init__(self, org_id):
        self.org_id = org_id


@pytest.fixture()
def org():
    return _Org(ORG_A)


@pytest.fixture()
def db_session():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=eng)
    session = sessionmaker(bind=eng)()
    seed_org_mailboxes(session)
    try:
        yield session
    finally:
        session.close()
        eng.dispose()


@pytest.fixture()
def client(db_session, org):
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_current_org_id] = lambda: org.org_id
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _payload(*, key="idem-1", subject="Hi", body="Hello there", email="lead@example.com"):
    return {
        "external_action_id": "481",
        "idempotency_key": key,
        "correlation_id": None,
        "recipient": {"email": email, "name": "Jane Doe"},
        "message": {"subject": subject, "body": body},
    }


def _post(client, **kw):
    return client.post("/integrations/leadboost/outreach-actions", json=_payload(**kw))


def _accepted(client, **kw):
    r = _post(client, **kw)
    assert r.status_code == 202, r.text
    return r


def _wired(db):
    """(dispatch, message, contact, campaign) for the single operation in db."""
    d = db.query(ExternalDispatch).one()
    return d, db.get(Message, d.message_id), db.get(Contact, d.contact_id), db.get(Campaign, d.campaign_id)


def _row(obj):
    return {col.name: getattr(obj, col.name) for col in obj.__table__.columns}


def _snapshot(d, m, c, k):
    return (_row(d), _row(m), _row(c), _row(k))


@pytest.fixture()
def smtp_layer_capture(monkeypatch):
    """Run the REAL send_email() but stop at the SMTP transport, capturing the
    MIME message it would transmit. No network, no SMTP."""
    from mailer_agent.mail import sender as snd

    captured = []
    monkeypatch.setattr(snd.settings, "live_sending_enabled", True)
    monkeypatch.setattr(snd, "_smtp_send_with_retry", lambda msg, frm, to: captured.append(msg))
    return captured


def _decoded(mime):
    part = mime.get_payload()[0]
    body = part.get_payload(decode=True).decode(part.get_content_charset() or "ascii")
    subject = str(email.header.make_header(email.header.decode_header(mime["Subject"])))
    return subject, body


# ===========================================================================
# EXACTNESS
# ===========================================================================

_NFD = unicodedata.normalize("NFD", "Café")  # 'e' + combining acute; NFC would differ
EXACT_CASES = [
    pytest.param("Hi", "Hello there", id="plain"),
    pytest.param("  Quick note  ", "  \n Hello  \n\n", id="leading-trailing-whitespace"),
    pytest.param("Hi", "Line one\n\nLine two\n   indented\nLine three\n", id="multiline-blank-lines"),
    pytest.param("Hi", "Line one\r\nLine two\r\n", id="crlf"),
    pytest.param("Hi\tthere", "a\tb  \nc \t\n", id="tabs-and-trailing-spaces"),
    pytest.param(f"{_NFD} — ✓", f"{_NFD} naïve 日本語 😀", id="unicode-no-normalization"),
    pytest.param("{0} %s {{name}}", "<b>Hi</b> **x** {{first_name}} %s {0} ${x}", id="template-lookalikes"),
    pytest.param("Hi", ("para " * 4000) + "\nend", id="long-body"),
]


@pytest.mark.parametrize("subject,body", EXACT_CASES)
def test_exact_through_http_db_and_send_input(client, db_session, subject, body):
    """Hand-offs 1+2: HTTP body -> persisted Message -> ExactSendInput."""
    _accepted(client, subject=subject, body=body)
    _, msg, contact, campaign = _wired(db_session)
    si = build_exact_send_input(msg, contact, campaign)
    for got, want in ((msg.subject, subject), (msg.body, body),
                      (si.subject, subject), (si.body_text, body)):
        assert got == want and got.encode("utf-8") == want.encode("utf-8")
    assert (si.to_email, si.from_email, si.from_name, si.reply_to) == (
        contact.email, campaign.sender_email, campaign.sender_name, campaign.reply_to_email)


@pytest.mark.parametrize("subject,body", EXACT_CASES)
def test_exact_through_real_send_email_to_smtp_layer(client, db_session, smtp_layer_capture, subject, body):
    """Hand-off 3, where the value actually changes hands: what the real
    send_email() builds from ExactSendInput is exact at the SMTP boundary."""
    _accepted(client, subject=subject, body=body)
    _, msg, contact, campaign = _wired(db_session)
    send_email(**build_exact_send_input(msg, contact, campaign).as_send_email_kwargs())
    (mime,) = smtp_layer_capture
    assert _decoded(mime) == (subject, body)


def test_absent_subject_is_never_invented(client, db_session, smtp_layer_capture):
    """None -> "" is a transport representation (send_email needs a str; at the
    MIME layer both give an empty Subject header). It must never become text."""
    _accepted(client, subject=None)
    _, msg, contact, campaign = _wired(db_session)
    assert msg.subject is None                            # persisted snapshot untouched
    si = build_exact_send_input(msg, contact, campaign)
    assert si.subject == ""
    for invented in (campaign.sender_org, campaign.name, campaign.value_prop):
        assert si.subject != invented
    send_email(**si.as_send_email_kwargs())
    (mime,) = smtp_layer_capture
    assert _decoded(mime)[0] == ""
    assert "Subject: \n" in mime.as_string()              # empty header, nothing invented


def test_send_input_kwargs_match_send_email_signature():
    """Drift guard: names exactly send_email's parameters, satisfies every
    required one, and carries no reply-threading (this path is initial outreach)."""
    params = inspect.signature(send_email).parameters
    required = {n for n, p in params.items() if p.default is inspect.Parameter.empty}
    kwargs = em.ExactSendInput("t@x.com", "f@x.com", "F", "s", "b", None).as_send_email_kwargs()
    assert set(kwargs) <= set(params) and required <= set(kwargs)
    assert "in_reply_to_header" not in kwargs and "references_header" not in kwargs


def test_unsafe_looking_content_is_accepted_and_stored_verbatim(client, db_session):
    """No accept-time gate or rewrite is added here: grounding is evaluated
    later (C7). A body that WILL hard-block is still stored untouched, QUEUED."""
    body = "We guarantee a 90% lift within 2 days."
    _accepted(client, body=body)
    d, msg, _, _ = _wired(db_session)
    assert msg.body == body and d.state == ExternalDispatchState.QUEUED.value


# ===========================================================================
# WRONG TARGET
# ===========================================================================

@pytest.mark.parametrize("mutate", [
    pytest.param(lambda m, c, k: setattr(m, "status", MessageStatus.SENT.value), id="already-sent"),
    pytest.param(lambda m, c, k: setattr(m, "status", MessageStatus.UNKNOWN.value), id="unknown-outcome"),
    pytest.param(lambda m, c, k: setattr(m, "status", MessageStatus.FAILED.value), id="failed"),
    pytest.param(lambda m, c, k: setattr(m, "direction", MessageDirection.INBOUND.value), id="inbound"),
    pytest.param(lambda m, c, k: setattr(m, "contact_id", c.id + 999), id="foreign-contact"),
    pytest.param(lambda m, c, k: setattr(c, "campaign_id", k.id + 999), id="foreign-campaign"),
])
def test_boundary_refuses_miswired_or_non_pre_send_input(client, db_session, mutate):
    _accepted(client)
    _, msg, contact, campaign = _wired(db_session)
    mutate(msg, contact, campaign)
    with pytest.raises(ExactMessageError):
        build_exact_send_input(msg, contact, campaign)
    with pytest.raises(ExactMessageError):
        evaluate_exact_message_grounding(msg, contact, campaign)
    db_session.rollback()


# ===========================================================================
# NO DUPLICATE MESSAGE / ASSOCIATION / PRE-SEND STATE
# ===========================================================================

def test_exactly_one_message_per_operation(client, db_session, monkeypatch):
    calls = []
    real = integrations_module.create_authorized_message
    monkeypatch.setattr(
        integrations_module, "create_authorized_message",
        lambda **kw: (calls.append(kw), real(**kw))[1],
    )
    _accepted(client, key="k1")                                    # POST
    assert len(calls) == 1 and db_session.query(Message).count() == 1
    _accepted(client, key="k1")                                    # same-key replay
    assert len(calls) == 1 and db_session.query(Message).count() == 1
    assert _post(client, key="k1", body="changed").status_code == 409   # modified replay
    assert len(calls) == 1 and db_session.query(Message).count() == 1
    (msg,) = db_session.query(Message).all()
    assert msg.body == "Hello there"                               # snapshot immutable


def test_second_operation_same_recipient_gets_its_own_exact_message(client, db_session):
    _accepted(client, key="k1", body="first body")
    _accepted(client, key="k2", body="second body")
    assert sorted(m.body for m in db_session.query(Message).all()) == ["first body", "second body"]
    assert db_session.query(Contact).count() == 1        # contact reused
    assert db_session.query(Campaign).count() == 1       # one integration campaign


def test_dispatch_message_contact_campaign_association(client, db_session):
    _accepted(client)
    d, msg, contact, campaign = _wired(db_session)
    assert d.message_id == msg.id
    assert msg.contact_id == d.contact_id == contact.id
    assert contact.campaign_id == d.campaign_id == campaign.id
    assert campaign.integration_source == "leadboost"
    assert campaign.organization_id == d.organization_id == ORG_A
    assert msg.direction == MessageDirection.OUTBOUND.value


def test_pre_send_state_and_boundary_calls_change_nothing(client, db_session):
    _accepted(client, body="Hello plain text")
    d, msg, contact, campaign = _wired(db_session)
    assert msg.status == MessageStatus.DRAFT.value and msg.message_id_header is None
    assert d.state == ExternalDispatchState.QUEUED.value
    assert d.claimed_by is None and d.claimed_at is None

    before = _snapshot(d, msg, contact, campaign)
    build_exact_send_input(msg, contact, campaign)
    evaluate_exact_message_grounding(msg, contact, campaign)
    assert not db_session.dirty and not db_session.new and not db_session.deleted
    db_session.expire_all()
    assert _snapshot(*_wired(db_session)) == before


# ===========================================================================
# PURITY: the boundary never writes through a session
# ===========================================================================

def test_create_authorized_message_returns_a_transient_unsaved_message():
    m = create_authorized_message(contact_id=7, subject=None, body="  x \n")
    assert isinstance(m, Message) and sa_inspect(m).transient      # not in any session
    assert m.id is None and m.contact_id == 7
    assert m.subject is None and m.body == "  x \n"                # verbatim, no defaulting
    assert m.status == MessageStatus.DRAFT.value
    assert m.direction == MessageDirection.OUTBOUND.value
    assert not any("session" in n or n == "db" for n in inspect.signature(create_authorized_message).parameters)


def test_boundary_functions_never_write_through_a_session(client, db_session, monkeypatch):
    _accepted(client)
    _, msg, contact, campaign = _wired(db_session)

    def boom(*a, **k):
        raise AssertionError("exact-message boundary must not write through a session")

    for name in ("add", "add_all", "delete", "merge", "flush", "commit"):
        monkeypatch.setattr(Session, name, boom)
    create_authorized_message(contact_id=contact.id, subject="s", body="b")
    build_exact_send_input(msg, contact, campaign)
    evaluate_exact_message_grounding(msg, contact, campaign)


# ===========================================================================
# ISOLATION: no LLM / drafting / SMTP
# ===========================================================================

# The complete set of modules exact_message.py may import (see its docstring).
ALLOWED_EXACT_MESSAGE_IMPORTS = {
    "__future__",
    "dataclasses",
    "mailer_agent.models",
    "mailer_agent.llm.grounding",     # regex-only safety net this path must reuse
    "mailer_agent.semantic_models",   # GroundingValidation result type
}
# Anything that generates, personalizes, sends, or transitively imports them.
FORBIDDEN_FOR_EXACT_PATH = (
    "mailer_agent.llm.agent", "mailer_agent.llm.provider", "mailer_agent.llm.provider_v2",
    "mailer_agent.llm.prompts", "mailer_agent.followup", "mailer_agent.memory",
    "mailer_agent.policy", "mailer_agent.semantic.", "groq",
)
FORBIDDEN_SMTP = ("mailer_agent.mail.sender", "smtplib", "imaplib")
DYNAMIC_IMPORT_NAMES = {"__import__", "import_module"}


def _import_edges(source: str, package: tuple[str, ...]) -> list[tuple[str, str | None]]:
    """(module, imported-name) for EVERY import in `source`: top-level, deferred
    (inside functions), and relative (resolved against `package`)."""
    edges: list[tuple[str, str | None]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            edges += [(a.name, None) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(package[: max(len(package) - (node.level - 1), 0)])
                if node.module:
                    base.append(node.module)
                mod = ".".join(base)
            else:
                mod = node.module or ""
            edges += [(mod, a.name) for a in node.names]
    return edges


def _dotted(edges) -> set[str]:
    return {m for m, _ in edges} | {f"{m}.{n}" for m, n in edges if n}


def _dynamic_imports(source: str) -> list[int]:
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in DYNAMIC_IMPORT_NAMES:
                hits.append(node.lineno)
    return hits


def _violations(dotted: set[str], forbidden: tuple[str, ...]) -> list[str]:
    return sorted(d for d in dotted for bad in forbidden
                  if d == bad.rstrip(".") or d.startswith(bad.rstrip(".") + "."))


def test_exact_message_imports_only_the_allowlist():
    src = EXACT_MESSAGE_PY.read_text(encoding="utf-8")
    modules = {m for m, _ in _import_edges(src, ("mailer_agent", "mail"))}
    assert modules <= ALLOWED_EXACT_MESSAGE_IMPORTS, sorted(modules - ALLOWED_EXACT_MESSAGE_IMPORTS)
    # It reuses (not reimplements) the existing safety net, and nothing more:
    assert {"mailer_agent.models", "mailer_agent.llm.grounding"} <= modules
    assert _dynamic_imports(src) == []


def test_integration_endpoint_imports_no_llm_no_smtp():
    src = INTEGRATIONS_PY.read_text(encoding="utf-8")
    dotted = _dotted(_import_edges(src, ("mailer_agent", "api")))
    assert _violations(dotted, FORBIDDEN_FOR_EXACT_PATH) == []
    assert _violations(dotted, FORBIDDEN_SMTP) == []
    assert "mailer_agent.mail.exact_message" in dotted
    assert _dynamic_imports(src) == []


@pytest.mark.parametrize("snippet,why", [
    ("from mailer_agent.llm.agent import draft_message", "direct llm.agent"),
    ("from mailer_agent.llm import provider", "direct llm.provider via from-import"),
    ("from mailer_agent.memory.store import build_conversation_context", "transitive memory.store"),
    ("from mailer_agent import followup", "package import of followup"),
    ("from ..followup.engine import is_suppressed", "relative import of followup.engine"),
    ("from ..memory import store", "relative import of memory"),
    ("from mailer_agent.mail.sender import send_email", "sender"),
    ("import smtplib", "smtplib"),
    ("def f():\n    from mailer_agent.llm.agent import draft_message", "lazy import in a function"),
    ("import importlib\nimportlib.import_module('mailer_agent.llm.agent')", "dynamic import_module"),
    ("__import__('mailer_agent.llm.agent')", "dynamic __import__"),
])
def test_static_checker_catches_evasive_import_forms(snippet, why):
    """The checker itself: each of these must be flagged for a module living in
    mailer_agent/mail/ (so a regression in exact_message.py cannot hide)."""
    edges = _import_edges(snippet, ("mailer_agent", "mail"))
    outside_allowlist = {m for m, _ in edges} - ALLOWED_EXACT_MESSAGE_IMPORTS
    assert outside_allowlist or _dynamic_imports(snippet), why


def test_exact_path_modules_load_without_llm_or_smtp_at_runtime():
    """Fresh interpreter, so nothing another test imported can mask a
    transitive import (memory.store -> llm.provider is exactly that kind)."""
    watched = [
        "mailer_agent.llm.agent", "mailer_agent.llm.provider", "mailer_agent.llm.provider_v2",
        "mailer_agent.llm.prompts", "mailer_agent.followup.engine", "mailer_agent.followup.engine_v2",
        "mailer_agent.memory.store", "mailer_agent.mail.sender", "groq", "smtplib", "imaplib",
    ]
    code = (
        "import sys\n"
        "import mailer_agent.mail.exact_message\n"
        "import mailer_agent.api.integrations\n"
        f"bad=[m for m in {watched!r} if m in sys.modules]\n"
        "print('LOADED:'+','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "LOADED:", out.stdout


def test_no_llm_or_drafting_at_runtime(client, db_session, fake_llm, monkeypatch):
    from mailer_agent.llm import agent

    def boom(*a, **k):
        raise AssertionError("drafting/LLM must not be reached on the exact path")

    monkeypatch.setattr(agent, "draft_message", boom)
    _accepted(client, body="Exactly this, please.")
    _, msg, contact, campaign = _wired(db_session)
    build_exact_send_input(msg, contact, campaign)
    evaluate_exact_message_grounding(msg, contact, campaign)
    assert fake_llm.call_count == 0
    assert msg.body == "Exactly this, please."


def test_no_smtp_at_runtime(client, db_session, monkeypatch):
    import smtplib
    from mailer_agent.mail import sender as sender_module

    def boom(*a, **k):
        raise AssertionError("SMTP/send_email must not be reached in C5")

    monkeypatch.setattr(smtplib, "SMTP", boom)
    monkeypatch.setattr(smtplib, "SMTP_SSL", boom)
    monkeypatch.setattr(sender_module, "send_email", boom)
    _accepted(client)
    _, msg, contact, campaign = _wired(db_session)
    build_exact_send_input(msg, contact, campaign)
    evaluate_exact_message_grounding(msg, contact, campaign)


# ===========================================================================
# GROUNDING (existing validate_grounding; hard-block only; read-only)
# ===========================================================================

_seq = iter(range(10_000))


def _decision_for(client, db_session, body, *, proof_points=None):
    _accepted(client, key=f"k-{next(_seq)}", body=body)
    d, msg, contact, campaign = _wired(db_session)
    if proof_points is not None:
        campaign.proof_points = proof_points
        db_session.commit()
    return evaluate_exact_message_grounding(msg, contact, campaign), (d, msg)


def test_plain_message_is_allowed(client, db_session):
    dec, _ = _decision_for(client, db_session, "Hi Jane,\n\nWorth a quick chat?\n\nBest")
    assert dec.blocked is False and dec.review_required is False and dec.error_message is None


def test_fabricated_metric_hard_blocks_without_editing_or_advancing_state(client, db_session):
    body = "Our customers see a 40% lift in reply rates."
    dec, (d, msg) = _decision_for(client, db_session, body)
    assert dec.blocked is True and dec.validation.hard_block is True
    assert "grounding" in dec.error_message and dec.validation.unsupported_claims
    db_session.refresh(msg)
    db_session.refresh(d)
    assert msg.body == body                                    # never rewritten to pass
    assert d.state == ExternalDispatchState.QUEUED.value       # state mapping is C7's job


def test_review_only_content_is_surfaced_but_not_blocked(client, db_session):
    dec, _ = _decision_for(client, db_session, "Happy to share pricing if useful.")
    assert dec.review_required is True and dec.blocked is False


def test_operator_supplied_proof_points_can_ground_a_metric(client, db_session):
    dec, _ = _decision_for(client, db_session, "Our customers see a 40% lift in reply rates.",
                           proof_points="Customers see a 40% lift in reply rates.")
    assert dec.blocked is False


def _legacy_verdict(db_session, msg, contact, campaign):
    return validate_grounding(
        msg.body,
        proof_points=campaign.proof_points,
        context_notes=contact.context_notes,
        conversation_transcript=build_conversation_context(db_session, contact),
        value_prop=campaign.value_prop,
    )


GROUNDING_BODIES = [
    "Hi Jane,\n\nWorth a quick chat?",
    "Our customers see a 40% lift in reply rates.",
    "We can get you set up within 24 hours.",
    "Happy to share pricing if useful.",
    "We guarantee results.",
    "You mentioned a 40% lift in reply rates -- happy to help.",
]


@pytest.mark.parametrize("body", GROUNDING_BODIES)
def test_no_history_matches_legacy_approval_gate(client, db_session, body):
    _accepted(client, body=body)
    _, msg, contact, campaign = _wired(db_session)
    legacy = _legacy_verdict(db_session, msg, contact, campaign)
    ours = evaluate_exact_message_grounding(msg, contact, campaign)
    assert ours.blocked == legacy.hard_block
    assert ours.review_required == legacy.review_required
    assert ours.validation.unsupported_claims == legacy.unsupported_claims


@pytest.mark.parametrize("body", GROUNDING_BODIES)
def test_with_history_never_unblocks_what_legacy_blocks(client, db_session, body):
    """Transcript is unavailable here (None), so the corpus can only be
    smaller than the legacy one: whatever legacy blocks, this blocks too.
    Property of the current validate_grounding, locked here -- not a guarantee."""
    _accepted(client, body=body)
    _, msg, contact, campaign = _wired(db_session)
    db_session.add(Message(
        contact_id=contact.id, direction=MessageDirection.INBOUND.value,
        subject="Re: earlier", body="We already see a 40% lift in reply rates.",
        status=MessageStatus.RECEIVED.value,
    ))
    db_session.commit()
    db_session.refresh(contact)
    legacy = _legacy_verdict(db_session, msg, contact, campaign)
    ours = evaluate_exact_message_grounding(msg, contact, campaign)
    assert (not legacy.hard_block) or ours.blocked


def test_history_grounded_claim_is_blocked_here_but_not_by_legacy(client, db_session):
    """The documented divergence, as a witness: a claim only the prospect's own
    earlier reply supports."""
    _accepted(client, body="You mentioned a 40% lift in reply rates -- happy to help.")
    _, msg, contact, campaign = _wired(db_session)
    db_session.add(Message(
        contact_id=contact.id, direction=MessageDirection.INBOUND.value,
        subject="Re: earlier", body="We already see a 40% lift in reply rates.",
        status=MessageStatus.RECEIVED.value,
    ))
    db_session.commit()
    db_session.refresh(contact)
    assert _legacy_verdict(db_session, msg, contact, campaign).hard_block is False
    assert evaluate_exact_message_grounding(msg, contact, campaign).blocked is True


# ===========================================================================
# TENANCY / IDEMPOTENCY (C2-C4 behavior unchanged)
# ===========================================================================

def test_two_orgs_same_key_get_independent_exact_messages(client, db_session, org):
    org.org_id = ORG_A
    ra = _accepted(client, key="shared", body="A's exact body ")
    org.org_id = ORG_B
    rb = _accepted(client, key="shared", body="B's exact body\n")
    assert ra.json()["mailing_agent_reference"] != rb.json()["mailing_agent_reference"]

    bodies = {d.organization_id: db_session.get(Message, d.message_id).body
              for d in db_session.query(ExternalDispatch).all()}
    assert bodies == {ORG_A: "A's exact body ", ORG_B: "B's exact body\n"}
    assert db_session.query(Campaign).filter(Campaign.integration_source == "leadboost").count() == 2

    org.org_id = ORG_A                                    # replay is scoped to A
    assert _accepted(client, key="shared", body="A's exact body ").json() == ra.json()
    assert db_session.query(Message).count() == 2


def test_idempotent_replay_returns_same_reference_without_mutation(client, db_session):
    r1 = _accepted(client, key="k")
    before = _snapshot(*_wired(db_session))
    r2 = _accepted(client, key="k")
    assert r1.json() == r2.json()
    db_session.expire_all()
    assert _snapshot(*_wired(db_session)) == before
