"""
Architecture tests (spec §33): enforce import boundaries between the
deterministic execution layer and the intelligent/semantic layer.

These parse each module's import statements with Python's `ast` module
rather than actually importing the modules at runtime. That's a
deliberate choice, not a shortcut: it means these tests can run (and
fail loudly) in any Python environment, including one that doesn't have
fastapi/sqlalchemy/groq installed at all -- an architecture boundary
violation is a property of the source code, and checking it shouldn't
require standing up the whole application first. It also can't be
fooled by a lazy/deferred import inside a function body being missed by
a runtime import-graph check that only inspects sys.modules after one
particular code path ran.

What "boundary violation" means here, concretely: a module in a given
layer directly importing a name from a module it has no business
depending on, per the layering this codebase settled on across two
hardening sessions (see docs/FINAL_PRODUCTION_READINESS.md):

  mailer_agent.semantic.*   -- interpretation only. No SMTP/IMAP, no
                                vendor LLM SDK, no direct DB session
                                mutation capability beyond what it's
                                handed.
  mailer_agent.policy.*     -- planner (LLM-backed, but only through
                                llm.provider/llm.provider_v2) and
                                guardrails (pure deterministic logic).
                                Neither may import mail-sending or DB
                                session code -- they propose/authorize,
                                they don't execute.
  mailer_agent.llm.*        -- the vendor SDK (groq) is confined to
                                llm/provider_v2.py specifically, not
                                scattered across llm/agent.py,
                                llm/prompts.py, llm/grounding.py, or
                                policy/*.
  mailer_agent.mail.*       -- the only layer allowed to import
                                smtplib/imaplib.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MAILER_AGENT_ROOT = Path(__file__).resolve().parent.parent / "mailer_agent"

# Modules that own a given "forbidden elsewhere" capability.
SMTP_IMAP_OWNERS = {"mailer_agent/mail/sender.py", "mailer_agent/mail/imap_reader.py"}
VENDOR_LLM_SDK_OWNERS = {"mailer_agent/llm/provider_v2.py"}

FORBIDDEN_SMTP_IMAP_MODULES = {"smtplib", "imaplib"}
FORBIDDEN_VENDOR_SDK_MODULES = {"groq"}


def _all_imported_module_names(filepath: Path) -> set[str]:
    """
    Every top-level module name this file imports, from both `import x`
    and `from x import y` (including deferred imports inside function
    bodies -- ast.walk visits the whole tree, not just module level, so
    a lazy `from groq import Groq` hidden inside a function is still
    caught).
    """
    tree = ast.parse(filepath.read_text(), filename=str(filepath))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def _python_files(subdir: str) -> list[Path]:
    d = MAILER_AGENT_ROOT / subdir
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*.py") if p.name != "__init__.py")


def _rel(filepath: Path) -> str:
    return str(filepath.relative_to(MAILER_AGENT_ROOT.parent)).replace("\\", "/")


# ---------------------------------------------------------------------------
# SMTP/IMAP confinement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filepath",
    [p for p in MAILER_AGENT_ROOT.rglob("*.py") if p.name != "__init__.py"],
    ids=lambda p: _rel(p),
)
def test_smtp_imap_confined_to_mail_module(filepath: Path):
    if _rel(filepath) in SMTP_IMAP_OWNERS:
        pytest.skip("This file is the designated owner of SMTP/IMAP access.")

    imported = _all_imported_module_names(filepath)
    violations = imported & FORBIDDEN_SMTP_IMAP_MODULES
    assert not violations, (
        f"{_rel(filepath)} imports {violations} directly -- SMTP/IMAP access "
        f"must be confined to mailer_agent/mail/sender.py and "
        f"mailer_agent/mail/imap_reader.py."
    )


# ---------------------------------------------------------------------------
# Vendor LLM SDK confinement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filepath",
    [p for p in MAILER_AGENT_ROOT.rglob("*.py") if p.name != "__init__.py"],
    ids=lambda p: _rel(p),
)
def test_vendor_llm_sdk_confined_to_provider_v2(filepath: Path):
    if _rel(filepath) in VENDOR_LLM_SDK_OWNERS:
        pytest.skip("This file is the designated owner of the vendor LLM SDK.")

    imported = _all_imported_module_names(filepath)
    violations = imported & FORBIDDEN_VENDOR_SDK_MODULES
    assert not violations, (
        f"{_rel(filepath)} imports {violations} directly -- the vendor LLM SDK "
        f"must be confined to mailer_agent/llm/provider_v2.py. Everything else, "
        f"including mailer_agent/semantic/classifier.py and "
        f"mailer_agent/policy/next_action.py, must call through "
        f"llm.provider/llm.provider_v2's functions, never import the SDK itself."
    )


# ---------------------------------------------------------------------------
# semantic/ layer: interpretation only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filepath", _python_files("semantic"), ids=lambda p: _rel(p))
def test_semantic_layer_does_not_import_mail_sending(filepath: Path):
    imported = _all_imported_module_names(filepath)
    # mailer_agent.mail as a dotted prefix -- check via ast module strings,
    # not just top-level name, since `from mailer_agent.mail.sender import
    # send_email` has top-level name "mailer_agent" (a local package, not
    # one of our "forbidden top-level module" names above). Check the
    # full dotted path instead for this one.
    tree = ast.parse(filepath.read_text())
    mail_imports = [
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mailer_agent.mail")
    ]
    assert not mail_imports, (
        f"{_rel(filepath)} imports from {mail_imports} -- the semantic "
        f"interpretation layer must not be able to send/receive mail directly."
    )


# ---------------------------------------------------------------------------
# policy/ layer: proposes and authorizes, never executes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filepath", _python_files("policy"), ids=lambda p: _rel(p))
def test_policy_layer_does_not_import_mail_sending_or_db_session(filepath: Path):
    tree = ast.parse(filepath.read_text())
    forbidden_prefixes = ("mailer_agent.mail", "mailer_agent.db")
    violations = [
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(forbidden_prefixes)
    ]
    assert not violations, (
        f"{_rel(filepath)} imports from {violations} -- the planner proposes "
        f"and guardrails authorize; neither may import mail-sending code or "
        f"the DB session/engine directly. The caller (mail/reply_handler_v2.py) "
        f"owns execution."
    )


def test_guardrails_module_has_no_llm_calls():
    """
    Guardrails are supposed to be pure deterministic policy -- no LLM
    call at all, not even an indirect one through llm.provider. Checked
    separately from the planner (policy/next_action.py), which is
    explicitly allowed to call the LLM (see that module's docstring for
    why planning is an intelligent-world responsibility, distinct from
    guardrail authorization, which is deterministic-world).
    """
    filepath = MAILER_AGENT_ROOT / "policy" / "guardrails.py"
    imported = _all_imported_module_names(filepath)
    tree = ast.parse(filepath.read_text())
    llm_module_imports = [
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mailer_agent.llm")
    ]
    assert not llm_module_imports, (
        f"policy/guardrails.py imports from {llm_module_imports} -- guardrails "
        f"must be pure deterministic logic with no LLM dependency at all."
    )


# ---------------------------------------------------------------------------
# Typed contract crossing: planner output must be the typed dataclass,
# not a raw dict, wherever it's consumed.
# ---------------------------------------------------------------------------

def test_guardrails_authorize_action_takes_typed_proposal():
    """
    LLM output must cross a typed contract before deterministic code acts
    on it (spec requirement). Checked concretely: guardrails.authorize_action's
    signature is typed against policy.next_action.NextActionProposal, a
    dataclass -- not `dict`, not `Any`, not an untyped **kwargs blob that
    would let a raw LLM response flow straight into policy logic.

    Uses typing.get_type_hints() rather than inspect.signature()'s raw
    .annotation, because guardrails.py has `from __future__ import
    annotations` (PEP 563 postponed evaluation) -- under that, every
    annotation in the module is stored as an unevaluated string
    ("NextActionProposal", not the class object), and
    inspect.signature() does not resolve those strings back into real
    types on its own. get_type_hints() does the resolution (against the
    function's own __globals__), which is the correct way to check an
    annotated type at runtime in a codebase that uses postponed
    annotations -- checking the raw string would still catch a
    completely wrong or missing annotation, but would call a correctly
    annotated parameter's own module style a failure, which is not the
    architectural property this test is trying to enforce.
    """
    import typing

    from mailer_agent.policy.guardrails import authorize_action
    from mailer_agent.policy.next_action import NextActionProposal

    hints = typing.get_type_hints(authorize_action)
    assert "proposal" in hints, "authorize_action must take an annotated `proposal` parameter"
    assert hints["proposal"] is NextActionProposal, (
        f"authorize_action's `proposal` parameter must be typed as "
        f"NextActionProposal, got {hints['proposal']!r} -- guardrails "
        f"must consume a typed contract, not raw LLM output."
    )
