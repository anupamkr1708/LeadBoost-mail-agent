from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from mailer_agent.config import get_settings

settings = get_settings()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    No-op if API_KEY is unset (local dev). Once you set API_KEY in the
    environment, every route requires a matching `X-API-Key` header --
    this is what LeadBoost (or anything else) authenticates with when
    calling this service. Uses a constant-time comparison so response
    timing can't be used to guess the key byte-by-byte.
    """
    if not settings.api_key:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
