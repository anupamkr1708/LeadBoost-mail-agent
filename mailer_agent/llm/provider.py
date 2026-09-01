"""
Unified LLM provider (backward compatibility wrapper).

This module delegates to provider_v2.py as the single source of truth.
All LLM calls use the improved retry logic, error classification, and
metrics.

Important: this module calls through to provider_v2's module attributes
at call time (`provider_v2.call_llm_json(...)`), not via import-time
name bindings (`from provider_v2 import call_llm_json`). That distinction
matters for tests: fixtures monkeypatch provider_v2's attributes (see
tests/fake_llm_provider.py) to swap in a deterministic fake. An
import-time binding would have captured the real function object before
the monkeypatch ever ran, silently defeating the fake and letting code
that goes through this wrapper (agent.py's draft_message, in
particular) make real network calls whenever a real GROQ_API_KEY happens
to be set in the environment.
"""

from __future__ import annotations

import logging
from typing import Any

from mailer_agent.llm import provider_v2
from mailer_agent.llm.provider_v2 import (
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


def is_llm_available() -> bool:
    """Delegates to provider_v2 at call time (see module docstring)."""
    return provider_v2.is_llm_available()


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
        return provider_v2.call_llm_text(
            system_prompt,
            human_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
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
        return provider_v2.call_llm_json(
            system_prompt,
            human_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except ProviderUnavailableError as e:
        raise LLMUnavailableError(str(e)) from e
    except MalformedOutputError as e:
        raise LLMOutputError(str(e)) from e
    except LLMProviderError as e:
        # Map other provider errors to output error for backward compat
        logger.error(f"LLM JSON call failed: {e}")
        raise LLMOutputError(str(e)) from e


__all__ = [
    "call_llm_text",
    "call_llm_json",
    "is_llm_available",
    "LLMUnavailableError",
    "LLMOutputError",
]
