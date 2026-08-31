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
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.4
    llm_max_tokens: int = 500

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
