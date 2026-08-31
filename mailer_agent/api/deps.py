"""
Authentication and authorization dependencies.

Multi-tenancy model (simple, no OAuth):
- Each organization gets one API key, set as ORG_KEYS_<KEY>=<org_id> in the environment,
  or as a JSON mapping in ORG_KEY_MAP (see config.py).
- If the single legacy API_KEY is set and no ORG_KEY_MAP is configured, we fall back to
  the old single-key behaviour and treat every caller as org "default".
- When a key resolves to an org, the org_id is injected into the request via
  get_current_org_id(); every data-access query must filter by it.

Key-to-org resolution order:
  1. ORG_KEY_MAP env var (JSON dict: {"<key>": "<org_id>", ...})  ← preferred for prod
  2. Individual ORG_KEYS_<n> env vars: ORG_KEYS_acme=api_key_xxx  ← handy for .env files
  3. Legacy API_KEY → org_id "default"                            ← backward compat

Usage in endpoints:
    @router.get("/campaigns")
    def list_campaigns(
        org_id: str = Depends(get_current_org_id),
        db: Session = Depends(get_db),
    ):
        return db.query(Campaign).filter(Campaign.organization_id == org_id).all()
"""

from __future__ import annotations

import json
import logging
import os
import secrets

from fastapi import Depends, Header, HTTPException, status

from mailer_agent.config import get_settings

logger = logging.getLogger("mailer_agent.api.deps")

settings = get_settings()

# ---------------------------------------------------------------------------
# Build the key→org_id lookup table at import time (cheap, done once).
# ---------------------------------------------------------------------------

def _build_key_map() -> dict[str, str]:
    """
    Build {api_key: org_id} from environment configuration.

    Three sources, merged in priority order (highest first):
      1. ORG_KEY_MAP – JSON dict in one env var, easiest for secrets managers.
      2. ORG_KEYS_<org_id>=<key> – individual per-org vars, easier in .env files.
      3. Legacy API_KEY – single key → org "default".
    """
    mapping: dict[str, str] = {}

    # 3. Legacy single key (lowest priority, may be overridden)
    if settings.api_key:
        mapping[settings.api_key] = "default"

    # 2. ORG_KEYS_<org_id>=<key> — scan environment
    for env_key, env_val in os.environ.items():
        if env_key.upper().startswith("ORG_KEYS_") and env_val.strip():
            org_id = env_key[len("ORG_KEYS_"):].lower()
            api_key = env_val.strip()
            mapping[api_key] = org_id

    # 1. ORG_KEY_MAP JSON (highest priority, overwrites anything above)
    org_key_map_raw = os.environ.get("ORG_KEY_MAP", "").strip()
    if org_key_map_raw:
        try:
            parsed: dict[str, str] = json.loads(org_key_map_raw)
            for key, org_id in parsed.items():
                if key and org_id:
                    mapping[key] = str(org_id)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error(
                "ORG_KEY_MAP is set but not valid JSON — ignoring it. Error: %s", exc
            )

    if not mapping:
        logger.warning(
            "No API keys configured (API_KEY, ORG_KEY_MAP, or ORG_KEYS_* are all unset). "
            "All requests will be allowed with org_id='default'. "
            "Set at least API_KEY in production."
        )

    return mapping


# Module-level singleton; recreated only if you reload the module.
_KEY_MAP: dict[str, str] = _build_key_map()


def _resolve_org(api_key: str) -> str | None:
    """
    Resolve an API key to its org_id.  Returns None if the key is unknown.
    Uses constant-time comparison for every candidate to prevent timing attacks.
    """
    matched_org: str | None = None
    for known_key, org_id in _KEY_MAP.items():
        # compare_digest always compares full strings — safe against timing.
        if secrets.compare_digest(api_key, known_key):
            matched_org = org_id
    return matched_org


# ---------------------------------------------------------------------------
# FastAPI dependency functions
# ---------------------------------------------------------------------------

def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    Backward-compatible auth check.  Raises 401 if the key is invalid when
    any keys are configured.  Use get_current_org_id() in routes that need
    the org_id for data-isolation filtering.
    """
    if not _KEY_MAP:
        # No keys configured — open access (dev mode).
        return
    if not x_api_key or _resolve_org(x_api_key) is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def get_current_org_id(x_api_key: str | None = Header(default=None)) -> str:
    """
    Resolve the calling organization from the X-API-Key header.

    Returns the org_id string (e.g. "acme", "leadboost", "default").

    If no keys are configured at all (dev mode), returns "default" so
    all data is still accessible without any filtering breakage.
    """
    if not _KEY_MAP:
        return "default"
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    org_id = _resolve_org(x_api_key)
    if org_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return org_id
