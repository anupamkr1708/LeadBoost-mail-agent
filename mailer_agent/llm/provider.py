"""
Unified LLM provider (backward compatibility wrapper).

This module now delegates to provider_v2.py as the single source of truth.
All LLM calls use the improved retry logic, error classification, and metrics.
"""

from __future__ import annotations

import logging
from typing import Any

# Import from v2 as authoritative implementation
from mailer_agent.llm.provider_v2 import (
    call_llm_json as _call_llm_json_v2,
    call_llm_text as _call_llm_text_v2,
    is_llm_available,
    LLMProviderError,
    MalformedOutputError,
    ProviderUnavailableError,
)

logger = logging.getLogger("mailer_agent.llm.provider")

# Backward compatibility exception aliases
class LLMUnavailableError(ProviderUnavailableError):
    """Backward compatible alias for ProviderUnavailableError."""
    pass


class LLMOutputError(MalformedOutputError):
    """Backward compatible alias for MalformedOutputError."""
    pass


def call_llm_text(
    system_prompt: str,
    human_prompt: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Free-text completion (delegates to provider_v2).
    
    Raises LLMUnavailableError/LLMOutputError on failure --
    callers are expected to fall back to a deterministic path (see agent.py).
    """
    try:
        return _call_llm_text_v2(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except ProviderUnavailableError as e:
        raise LLMUnavailableError(str(e)) from e
    except MalformedOutputError as e:
        raise LLMOutputError(str(e)) from e
    except LLMProviderError as e:
        # Map other provider errors to output error for backward compat
        logger.error(f"LLM call failed: {e}")
        raise LLMOutputError(str(e)) from e


def call_llm_json(
    system_prompt: str,
    human_prompt: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """
    JSON completion (delegates to provider_v2).
    
    Raises LLMUnavailableError/LLMOutputError on failure.
    """
    try:
        return _call_llm_json_v2(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except ProviderUnavailableError as e:
        raise LLMUnavailableError(str(e)) from e
    except MalformedOutputError as e:
        raise LLMOutputError(str(e)) from e
    except LLMProviderError as e:
        # Map other provider errors to output error for backward compat
        logger.error(f"LLM JSON call failed: {e}")
        raise LLMOutputError(str(e)) from e


# Re-export is_llm_available for compatibility
__all__ = [
    "call_llm_text",
    "call_llm_json",
    "is_llm_available",
    "LLMUnavailableError",
    "LLMOutputError",
]
