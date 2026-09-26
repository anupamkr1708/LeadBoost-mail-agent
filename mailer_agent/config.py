"""
Central configuration for the Mailer Agent.

Every tunable value (SMTP/IMAP creds, LLM model/key, poll intervals,
default follow-up cadence, safety toggles) lives here as an environment
variable with a sane default. No module outside this file should read
os.environ directly -- that is what makes the rest of the codebase free
of hardcoded behavior: change behavior by changing env vars / API
payloads, not by editing source.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database -----------------------------------------------------
    database_url: str = "sqlite:///./mailer_agent.db"

    # --- LLM ------------------------------------------------------------
    groq_api_key: str = ""
    # llama-3.3-70b-versatile was deprecated by Groq in June 2026 (see
    # .env.example) -- this default must track whatever .env.example
    # currently recommends, since anyone who doesn't set LLM_MODEL
    # explicitly gets THIS value, not the .env.example comment. If you
    # change the recommendation in .env.example, change it here too.
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.4
    llm_max_tokens: int = 500
    # Comma-separated, tried in order if llm_model fails with an
    # eligible-for-failover error (rate limit, timeout, provider 5xx,
    # model-not-found on a fallback, or malformed output after
    # strict+lenient parsing both failed on the primary -- see
    # llm/provider_v2.py's _FAILOVER_ELIGIBLE_TRANSIENT / ModelUnavailableError).
    # NOT tried for authentication failures or genuine request/schema
    # validation errors -- see the same module's routing logic and its
    # docstring for why.
    #
    # Default is openai/gpt-oss-20b only -- both gpt-oss models are
    # Groq PRODUCTION-tier (console.groq.com/docs/models). qwen/qwen3.8-27b
    # is currently a Groq PREVIEW model: Groq's own docs say preview
    # models are for evaluation and can be discontinued at short notice,
    # so it is deliberately NOT in the default production fallback chain
    # even though it has the same JSON Schema Mode + reasoning_effort
    # capabilities as the gpt-oss models (see MODEL_CAPABILITIES in
    # llm/provider_v2.py) and remains a fine explicit opt-in:
    # LLM_FALLBACK_MODELS=openai/gpt-oss-20b,qwen/qwen3.8-27b
    llm_fallback_models: str = "openai/gpt-oss-20b"
    # GPT-OSS models reject "none" outright (400) -- their valid values
    # are low/medium/high, defaulting server-side to "medium" if omitted.
    # qwen/qwen3.8-27b additionally supports low/medium/high (default
    # "none"). Only ever sent to a model whose capability entry in
    # llm/provider_v2.py's MODEL_CAPABILITIES says it supports this
    # value -- never sent blindly. See /docs/api-reference.
    llm_reasoning_effort: str = "low"

    # --- Outbound mail (SMTP) -------------------------------------------
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True

    # --- Inbound mail (IMAP, for reply detection) ------------------------
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    imap_username: str = ""
    imap_password: str = ""
    imap_poll_seconds: int = 120

    # --- Follow-up scheduler ---------------------------------------------
    followup_poll_seconds: int = 300
    # Used only if a campaign is created without an explicit cadence.
    # Campaigns should normally set their own follow_up_days_after_send.
    default_followup_days: list[int] = [3, 7, 14]
    # When running as a single process (local dev, or a single Render Web
    # Service), the scheduler starts in-process via the FastAPI lifespan.
    # In production with multiple gunicorn workers, set this False and run
    # `python worker.py` as a separate process (a Render Background Worker)
    # instead -- otherwise every worker process would start its own
    # scheduler and you'd get duplicate sends/duplicate reply polls.
    run_scheduler_in_process: bool = True

    # --- Safety toggles ----------------------------------------------------
    # When False, generated sends/replies are written to the DB as
    # status="draft" instead of actually being emailed -- lets you
    # inspect agent output before it touches a real inbox.
    live_sending_enabled: bool = False
    # When False, incoming replies are classified and a reply is drafted,
    # but never auto-sent -- someone has to call POST /threads/{id}/approve.
    auto_reply_enabled: bool = False
    # Hard ceiling regardless of campaign/contact count, per poll cycle.
    max_sends_per_cycle: int = 20
    # Real prospects landing seconds apart from the same sender reads as
    # bot activity (and hammers your SMTP/IMAP rate limits). This is the
    # base gap between consecutive sends within one dispatch batch;
    # send_jitter_seconds adds a random 0-N second variation on top so
    # the cadence doesn't look mechanically fixed either.
    send_delay_seconds: float = 8.0
    send_jitter_seconds: float = 7.0

    # --- API / Multi-tenancy --------------------------------------------
    # Single legacy key (maps to org "default"). For multi-tenant use,
    # prefer ORG_KEY_MAP (JSON) or individual ORG_KEYS_<org_id>=<key> vars
    # -- see mailer_agent/api/deps.py for the full resolution order.
    api_key: str = ""  # if set, required as `X-API-Key` header on all routes

    @property
    def llm_fallback_models_list(self) -> list[str]:
        """llm_fallback_models parsed into an ordered list, empty entries
        and whitespace stripped. A plain property (not a field) so it's
        always derived fresh from the current llm_fallback_models value
        rather than parsed once and risking drift."""
        return [m.strip() for m in self.llm_fallback_models.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
