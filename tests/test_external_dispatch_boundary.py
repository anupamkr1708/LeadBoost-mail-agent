"""
The exact-message dispatch path must be LLM-free and (for suppression)
sender-free. Three independent proofs, because each catches what the others
miss:

1. AST      -- the import statements written in the source.
2. Fresh interpreter -- what actually gets loaded transitively.
3. Behavioural -- the whole pipeline runs to SENT with every LLM/drafting
   module made UNIMPORTABLE, so any hidden lazy import would explode.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORBIDDEN_PREFIXES = (
    "mailer_agent.llm.agent",
    "mailer_agent.llm.provider",
    "mailer_agent.llm.provider_v2",
    "mailer_agent.llm.prompts",
    "mailer_agent.memory",
    "mailer_agent.followup.engine",
    "mailer_agent.followup.engine_v2",
    "mailer_agent.followup.conversation_aware",
    "mailer_agent.api",
    "groq",
    "openai",
)


def _imports(relpath: str) -> set[str]:
    tree = ast.parse(open(os.path.join(ROOT, relpath)).read())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{a.name}" for a in node.names}
    return found


def _bad(imports: set[str], prefixes) -> list[str]:
    return sorted(i for i in imports if any(i == p or i.startswith(p + ".") for p in prefixes))


def _fresh(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
        env={**os.environ, "PYTHONPATH": ROOT, "DATABASE_URL": "sqlite://"},
    )


# ---------------------------------------------------------------------- AST

def test_worker_source_imports_no_llm_or_drafting_module():
    assert _bad(_imports("mailer_agent/mail/external_dispatch_worker.py"), FORBIDDEN_PREFIXES) == []


def test_suppression_source_imports_only_models_and_orm():
    imports = _imports("mailer_agent/suppression.py")
    assert _bad(imports, FORBIDDEN_PREFIXES + ("mailer_agent.mail",)) == []
    assert {i for i in imports if i.startswith("mailer_agent")} <= {
        "mailer_agent.models", "mailer_agent.models.SuppressionEntry"}


def test_work_claiming_still_imports_neither_sender_nor_llm():
    assert _bad(_imports("mailer_agent/followup/work_claiming.py"),
                FORBIDDEN_PREFIXES + ("mailer_agent.mail",)) == []


# --------------------------------------------------------- fresh interpreter

def test_fresh_import_of_worker_loads_no_llm_provider_or_engine():
    out = _fresh(f"""
        import sys, mailer_agent.mail.external_dispatch_worker
        bad = sorted(m for m in sys.modules
                     if any(m == p or m.startswith(p + '.') for p in {FORBIDDEN_PREFIXES!r}))
        print(bad)
    """)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_fresh_import_of_suppression_loads_neither_sender_nor_llm():
    out = _fresh(f"""
        import sys, mailer_agent.suppression
        prefixes = {FORBIDDEN_PREFIXES!r} + ('mailer_agent.mail', 'smtplib')
        print(sorted(m for m in sys.modules
                     if any(m == p or m.startswith(p + '.') for p in prefixes)))
    """)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


# --------------------------------------------------------------- behavioural

def test_full_pipeline_runs_with_llm_and_engine_modules_unimportable(tmp_path):
    script = textwrap.dedent(f"""
        import sys, importlib.abc

        BLOCK = {FORBIDDEN_PREFIXES!r}
        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if any(name == p or name.startswith(p + '.') for p in BLOCK):
                    raise ImportError('BLOCKED IMPORT ON EXACT-MESSAGE PATH: ' + name)
        sys.meta_path.insert(0, Blocker())

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from mailer_agent.models import Base, ExternalDispatch
        from mailer_agent.mail import external_dispatch_worker as w
        from tests.dispatch_support import seed_dispatch, FakeSender

        eng = create_engine('sqlite:///{tmp_path / "b.db"}')
        Base.metadata.create_all(bind=eng)
        sm = sessionmaker(bind=eng)
        with sm() as s:
            did = seed_dispatch(s, body='Hi Jane,\\n\\nWorth a quick chat?\\n\\nBest').id

        w.settings.live_sending_enabled = True
        fake = FakeSender()
        w.send_email = fake
        res = w.process_next_external_dispatch(session_factory=sm, worker_id='w', runtime=w.DispatchRuntime())
        with sm() as s:
            state = s.get(ExternalDispatch, did).state
        print(res.outcome, state, len(fake.calls))
    """)
    from cryptography.fernet import Fernet

    out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True,
                         timeout=180,
                         env={**os.environ, "PYTHONPATH": ROOT,
                              "MAILBOX_ENCRYPTION_KEY": Fernet.generate_key().decode()})
    assert out.stdout.strip() == "sent sent 1", out.stdout + out.stderr
