"""
Improved LLM provider with coordinated retry policy and explicit failure handling.

Key improvements over original provider.py:
1. Single coordinated retry policy (not stacked)
2. Different strategies for different error types
3. Explicit failure classification
4. Rate limit backoff
5. No SDK-level retry conflicts
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from mailer_agent.config import get_settings
from mailer_agent.semantic_models import ClassificationFailureReason

logger = logging.getLogger("mailer_agent.llm.provider_v2")
settings = get_settings()


class LLMProviderError(Exception):
    """Base class for LLM provider errors."""
    def __init__(self, message: str, reason: ClassificationFailureReason, retryable: bool = False):
        super().__init__(message)
        self.reason = reason
        self.retryable = retryable


class RateLimitError(LLMProviderError):
    """Rate limit exceeded."""
    def __init__(self, message: str):
        super().__init__(message, ClassificationFailureReason.RATE_LIMITED, retryable=True)


class TimeoutError(LLMProviderError):
    """Request timeout."""
    def __init__(self, message: str):
        super().__init__(message, ClassificationFailureReason.TIMEOUT, retryable=True)


class MalformedOutputError(LLMProviderError):
    """LLM returned unparseable output."""
    def __init__(self, message: str):
        super().__init__(message, ClassificationFailureReason.MALFORMED_OUTPUT, retryable=False)


class ValidationError(LLMProviderError):
    """Output failed schema validation."""
    def __init__(self, message: str):
        super().__init__(message, ClassificationFailureReason.VALIDATION_ERROR, retryable=False)


class ProviderUnavailableError(LLMProviderError):
    """Provider service unavailable (connection issue, 5xx, etc) -- worth retrying."""
    def __init__(self, message: str):
        super().__init__(message, ClassificationFailureReason.PROVIDER_UNAVAILABLE, retryable=True)


class AuthenticationError(LLMProviderError):
    """
    Invalid/expired/missing API credentials.

    Deliberately NOT a subclass of ProviderUnavailableError and
    deliberately excluded from _chat_with_retry's retry-eligible
    exception tuple: a bad API key fails identically on every attempt,
    so retrying it doesn't just fail to help, it triples the latency of
    a failure that was already certain on the first try. This is what
    "authentication failures should not be retried" means in practice --
    tenacity's retry_if_exception_type matches by isinstance(), not by
    consulting an instance attribute, so this has to be a genuinely
    separate exception type to actually be excluded, not merely flagged
    with retryable=False.
    """
    def __init__(self, message: str):
        super().__init__(message, ClassificationFailureReason.PROVIDER_UNAVAILABLE, retryable=False)


def _get_client():
    """Get Groq client with retry disabled at SDK level."""
    if not settings.groq_api_key:
        raise ProviderUnavailableError("GROQ_API_KEY is not configured")
    
    from groq import Groq
    
    # Disable SDK-level retries to prevent stacking with our application retry
    return Groq(
        api_key=settings.groq_api_key,
        max_retries=0,  # ← Disable SDK retry, we handle it
        timeout=30.0
    )


def _classify_error(error: Exception) -> LLMProviderError:
    """Convert generic exceptions to classified LLM errors."""
    if isinstance(error, LLMProviderError):
        return error

    error_str = str(error).lower()

    # Order matters throughout this function: explicit HTTP status-code
    # semantics must win over generic substring/textual matching,
    # because a status reason phrase can contain a word that would
    # otherwise match a different, wrong branch. The concrete case that
    # motivated this ordering: "504 Gateway Timeout" contains the word
    # "timeout", and used to hit the generic timeout check below before
    # ever reaching the 5xx check -- misclassifying a server-side
    # condition as a client-side timeout. Every explicit status-code
    # branch below (401/403, 429, 500/502/503/504, 400) is checked
    # before the generic text-only branches (timeout, malformed JSON)
    # that exist to classify errors with no HTTP status code available
    # at all (e.g. a raw network-level timeout).

    if "401" in error_str or "403" in error_str or "authentication" in error_str or "invalid api key" in error_str:
        return AuthenticationError(f"Authentication failed: {error}")

    if "429" in error_str or "rate_limit" in error_str or "too many requests" in error_str:
        return RateLimitError(f"Rate limit exceeded: {error}")

    if "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
        return ProviderUnavailableError(f"Provider server error: {error}")

    if "400" in error_str:
        return ValidationError(f"Bad request: {error}")

    # Below this point: no explicit HTTP status code was present in the
    # error text, so fall back to generic textual classification.

    if "timeout" in error_str or "timed out" in error_str:
        return TimeoutError(f"Request timeout: {error}")
    
    if "json_validate_failed" in error_str or "invalid json" in error_str:
        return MalformedOutputError(f"Invalid JSON from model: {error}")
    
    # Unknown error - not retryable by default
    return LLMProviderError(str(error), ClassificationFailureReason.UNKNOWN_ERROR, retryable=False)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=10) + wait_random(0, 2),  # Jitter to prevent thundering herd
    retry=retry_if_exception_type(
        (RateLimitError, TimeoutError, ProviderUnavailableError)
    )
)
def _chat_with_retry(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    response_format: Optional[dict] = None
) -> str:
    """
    Single retry wrapper with coordinated backoff.
    
    Only retries rate limits, timeouts, and provider errors.
    Does NOT retry malformed output or validation errors (those are model issues).
    """
    try:
        client = _get_client()
        
        kwargs = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        response = client.chat.completions.create(**kwargs)
        
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise MalformedOutputError("LLM returned empty content")
        
        return content.strip()
        
    except Exception as e:
        # Classify and re-raise as appropriate error type
        classified = _classify_error(e)
        logger.warning(f"LLM call failed: {classified.reason.value} - {e}")
        raise classified


def call_llm_text(
    system_prompt: str,
    human_prompt: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Free-text completion with coordinated retry.
    
    Raises LLMProviderError subclasses on failure - callers should catch
    and handle appropriately (fallback, human review, etc).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]
    
    temp = temperature if temperature is not None else settings.llm_temperature
    tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
    
    return _chat_with_retry(messages, temperature=temp, max_tokens=tokens)


def call_llm_json(
    system_prompt: str,
    human_prompt: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """
    JSON completion with strict-then-lenient fallback strategy.
    
    Strategy:
    1. Try strict JSON mode first (fast path)
    2. If strict mode fails with json_validate_failed, fall back to lenient
    3. Parse output with lenient JSON extraction
    
    Raises LLMProviderError subclasses on failure.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]
    
    temp = temperature if temperature is not None else settings.llm_temperature
    tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
    
    # Try strict JSON mode first
    try:
        content = _chat_with_retry(
            messages,
            temperature=temp,
            max_tokens=tokens,
            response_format={"type": "json_object"}
        )
        return _parse_json(content)
        
    except MalformedOutputError as e:
        # Strict JSON mode failed with validation error
        # This is a known Groq issue with some models
        # Fall back to lenient mode without response_format constraint
        logger.info(f"Strict JSON mode failed ({e}), falling back to lenient parsing")
        
        try:
            content = _chat_with_retry(
                messages,
                temperature=temp,
                max_tokens=tokens,
                response_format=None  # No constraint
            )
            return _parse_json(content)
            
        except Exception as lenient_error:
            # Both attempts failed
            classified = _classify_error(lenient_error)
            logger.error(f"JSON call failed on both strict and lenient: {classified.reason.value}")
            raise classified


def _parse_json(raw: str) -> dict[str, Any]:
    """
    Parse JSON with lenient extraction for markdown-wrapped output.
    
    Handles:
    - Clean JSON: {"key": "value"}
    - Markdown wrapped: ```json\n{"key": "value"}\n```
    - Prose wrapped: "Here's the result: {"key": "value"}"
    """
    cleaned = raw.strip()
    
    # Remove markdown code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    
    # Try parsing directly
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Extract first {...} block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as e:
            raise MalformedOutputError(f"Could not extract valid JSON: {e}")
    
    raise MalformedOutputError(f"No JSON object found in output: {raw[:200]}...")


def is_llm_available() -> bool:
    """Check if LLM is configured and reachable."""
    return bool(settings.groq_api_key)


# Metrics tracking
class LLMMetrics:
    """Simple metrics tracker for LLM calls."""
    
    def __init__(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rate_limits = 0
        self.timeouts = 0
        self.malformed_outputs = 0
        self.total_retry_attempts = 0
        
    def record_success(self):
        self.total_calls += 1
        self.successful_calls += 1
    
    def record_failure(self, error: LLMProviderError):
        self.total_calls += 1
        self.failed_calls += 1
        
        if isinstance(error, RateLimitError):
            self.rate_limits += 1
        elif isinstance(error, TimeoutError):
            self.timeouts += 1
        elif isinstance(error, MalformedOutputError):
            self.malformed_outputs += 1
    
    def get_stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls if self.total_calls > 0 else 0,
            "rate_limits": self.rate_limits,
            "timeouts": self.timeouts,
            "malformed_outputs": self.malformed_outputs,
        }


# Global metrics instance
llm_metrics = LLMMetrics()
