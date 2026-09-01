"""
Integration/health facade for the Mailer Agent process.

History / why this file is small
---------------------------------
This used to be a much larger "unified interface" class (MailerAgentCore)
that re-exposed send_initial_outreach / send_followup / process_reply /
classify_reply / get_next_best_action / reschedule_after_reply as
delegating methods, plus module-level "backward compatible" functions
wrapping those.

None of that delegation was actually on the runtime path: the real send
path is followup/engine.py -> followup/engine_v2.py (IntegratedFollowUpEngine)
and mail/reply_handler.py -> mail/reply_handler_v2.py (process_inbound_email_v2),
called directly by the API routes and the scheduler. Nothing in the
codebase called through MailerAgentCore for any of that -- verified by
grepping for every import of `mailer_agent.integration`, which turned up
exactly one consumer (api/system.py), and it only ever used
`get_llm_metrics()` and `validate_integration()`.

Worse, the removed `get_system_health()` method referenced `Message`
without importing it at module scope -- a NameError waiting to happen.
That it went uncaught is itself evidence this path was never exercised,
despite prior "production ready" reports. The actual org-scoped, SQL-
aggregated health/metrics logic now lives directly in api/system.py,
which needs org_id for correct multi-tenant scoping anyway (this facade
has no request context to get that from).

What's left here is only the part that's genuinely process-wide and
does get called: LLM provider metrics, and a deployment self-check.
"""

from __future__ import annotations

import logging

from mailer_agent.config import get_settings
from mailer_agent.llm import provider_v2 as llm_provider

logger = logging.getLogger("mailer_agent.integration")
settings = get_settings()


class MailerAgentCore:
    """Process-wide health/integration-check facade. See module docstring."""

    def get_llm_metrics(self) -> dict:
        """Process-wide LLM provider call metrics (not org-scoped)."""
        return llm_provider.llm_metrics.get_stats()

    def validate_integration(self) -> dict:
        """
        Validate that all components are properly wired up in this
        deployment: LLM configured, core modules importable, database
        reachable. Booleans only -- no tenant data.
        """
        checks = {}

        checks["llm_available"] = llm_provider.is_llm_available()
        checks["groq_api_key_set"] = bool(settings.groq_api_key)
        checks["smtp_configured"] = bool(settings.smtp_username and settings.smtp_password)
        checks["imap_configured"] = bool(settings.imap_username and settings.imap_password)

        try:
            from mailer_agent.semantic_models import ClassificationResult, SemanticIntent  # noqa: F401
            checks["semantic_models_imported"] = True
        except Exception as e:
            checks["semantic_models_imported"] = False
            checks["semantic_models_error"] = str(e)

        try:
            from mailer_agent.state_machine import transition_contact_state  # noqa: F401
            checks["state_machine_imported"] = True
        except Exception as e:
            checks["state_machine_imported"] = False
            checks["state_machine_error"] = str(e)

        try:
            from mailer_agent.policy.next_action import NextBestActionPolicy  # noqa: F401
            checks["policy_engine_imported"] = True
        except Exception as e:
            checks["policy_engine_imported"] = False
            checks["policy_engine_error"] = str(e)

        try:
            from sqlalchemy import text

            from mailer_agent.db import engine
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            checks["database_connected"] = True
        except Exception as e:
            checks["database_connected"] = False
            checks["database_error"] = str(e)

        checks["integration_valid"] = all([
            checks.get("llm_available", False),
            checks.get("semantic_models_imported", False),
            checks.get("state_machine_imported", False),
            checks.get("policy_engine_imported", False),
            checks.get("database_connected", False),
        ])

        return checks


# Global instance -- stateless aside from the shared llm_metrics counters,
# safe to share across requests.
mailer_agent = MailerAgentCore()
