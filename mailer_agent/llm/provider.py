"""
Thin, resilient wrapper around the Groq chat-completions API.

This is the *only* file in the codebase that imports the `groq` SDK --
everything else talks to `call_llm_json()` / `call_llm_text()`. That
makes swapping providers (OpenAI, Anthropic, a local model) a one-file
change.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mailer_agent.config import get_settings

logger = logging.getLogger("mailer_agent.llm")

settings = get_settings()


class LLMUnavailableError(Exception):
    """Raised when the LLM cannot be reached/authenticated at all."""


class LLMOutputError(Exception):
    """Raised when the LLM responded but the output couldn't be used."""


def _get_client():
    if not settings.groq_api_key:
        raise LLMUnavailableError("GROQ_API_KEY is not configured")
    from groq import Groq  # imported lazily so the package is optional at import time

    return Groq(api_key=settings.groq_api_key)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)
def _chat(messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise LLMOutputError("LLM returned empty content")
    return content


def call_llm_text(
    system_prompt: str,
    human_prompt: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Free-text completion. Raises LLMUnavailableError/LLMOutputError on failure --
    callers are expected to fall back to a deterministic path (see agent.py)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]
    try:
        return _chat(
            messages,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_tokens=max_tokens if max_tokens is not None else settings.llm_max_tokens,
        ).strip()
    except LLMUnavailableError:
        raise
    except Exception as e:  # network errors, auth errors, rate limits after retries
        logger.error("LLM text call failed: %s", e)
        raise LLMOutputError(str(e)) from e


def call_llm_json(
    system_prompt: str,
    human_prompt: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """
    JSON completion, with a two-tier strategy learned from a real, well-
    documented Groq issue: some models (gpt-oss in particular) reject
    their own output against strict `response_format={"type":"json_object"}`
    validation with a 400 `json_validate_failed` on a meaningful fraction
    of calls -- and simply retrying the identical strict request rarely
    fixes it, since the failure is a property of that generation, not a
    transient network blip.

    Strategy: try strict JSON mode first (fast path when the model
    honors it). If it fails specifically on JSON validation, don't waste
    retries repeating the same doomed strict call -- immediately fall
    back to a plain completion (no response_format constraint) with the
    same prompt, and parse its output leniently. Only after *that* also
    fails does this raise, handing control back to the caller's
    deterministic fallback template.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]
    temp = temperature if temperature is not None else settings.llm_temperature
    tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

    try:
        client = _get_client()
    except LLMUnavailableError:
        raise

    try:
        content = _chat_json_strict(client, messages, temp, tokens)
        return _parse_json(content)
    except Exception as e:
        if "json_validate_failed" in str(e):
            logger.info("Strict JSON mode rejected by model, retrying without response_format constraint")
        else:
            logger.warning("Strict JSON mode call failed (%s), retrying without response_format constraint", e)

    try:
        content = _chat(messages, temperature=temp, max_tokens=tokens)
        return _parse_json(content)
    except LLMUnavailableError:
        raise
    except Exception as e:
        logger.error("LLM JSON call failed on both strict and lenient attempts: %s", e)
        raise LLMOutputError(str(e)) from e


@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=8),
)
def _chat_json_strict(client, messages, temperature, max_tokens) -> str:
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise LLMOutputError("LLM returned empty JSON content")
    return content


def _parse_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Lenient path (used for the non-strict retry, where the model may
    # wrap the JSON in prose): grab the first balanced-looking {...} block.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as e:
            raise LLMOutputError(f"LLM returned invalid JSON: {e}") from e
    raise LLMOutputError("LLM returned invalid JSON: no JSON object found in output")


def is_llm_available() -> bool:
    return bool(settings.groq_api_key)
