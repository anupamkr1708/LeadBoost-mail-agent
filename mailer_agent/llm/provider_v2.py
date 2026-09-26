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
import re
import time
from dataclasses import dataclass
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
    def __init__(self, message: str, retry_after_seconds: Optional[float] = None):
        super().__init__(message, ClassificationFailureReason.RATE_LIMITED, retryable=True)
        # See _parse_retry_after / _wait_strategy below: when Groq tells us
        # how long to actually wait, honor that instead of guessing with a
        # fixed backoff that may be shorter than what's required -- an
        # observed real failure mode (see
        # tests/test_provider_error_classification.py), where every retry
        # attempt landed inside the same still-rate-limited window and
        # burned all 3 attempts without ever succeeding.
        self.retry_after_seconds = retry_after_seconds


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


class ModelUnavailableError(LLMProviderError):
    """
    The requested model ID doesn't exist / isn't available on the
    provider's end (HTTP 404, or a "model not found"/"does not exist"
    style message) -- distinct from ProviderUnavailableError (the
    PROVIDER itself is degraded, a 5xx) and from RateLimitError (the
    model exists but is temporarily over capacity). Retrying the SAME
    model can never succeed here -- the model ID itself is invalid,
    deprecated, or renamed. Reuses ClassificationFailureReason.
    PROVIDER_UNAVAILABLE for the coarse downstream category (no new
    enum member needed for this), but is its own exception type so the
    router can give it the asymmetric primary-vs-fallback handling
    described where it's caught (see call_llm_json/call_llm_text): a
    404 on the PRIMARY model is very likely an operator configuration
    error (typo, a model ID that's since been deprecated) and should
    fail loudly rather than be silently routed around; a 404 on a
    FALLBACK model is far more forgivable to just skip past.
    """
    def __init__(self, message: str):
        super().__init__(message, ClassificationFailureReason.PROVIDER_UNAVAILABLE, retryable=False)


def _redact_secrets(text: str) -> str:
    """
    Strip the configured GROQ_API_KEY value out of text before it's
    logged. Applied at the one place in this module that logs raw
    exception text (see _chat_with_retry) -- every other log call in
    this module logs structured fields (model name, error CLASS name,
    attempt counts) rather than raw exception text, so this is the only
    point where a provider error message's literal content reaches a
    log line and could, in principle, echo back something sensitive
    (e.g. some providers include the offending credential in a 401
    body). Defense in depth: this should never actually trigger against
    real Groq responses, but "should never happen" is exactly the case
    logging code needs to be defensive about, not optimistic about.
    """
    key = settings.groq_api_key
    if not key:
        return text
    return text.replace(key, "***REDACTED***")


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


# ---------------------------------------------------------------------------
# Model capability registry
# ---------------------------------------------------------------------------
#
# Verified against current Groq documentation at the time this was added:
# console.groq.com/docs/models, /docs/reasoning, /docs/api-reference,
# /docs/deprecations, /docs/model/openai/gpt-oss-120b, /docs/model/qwen/qwen3.8-27b.
#
# openai/gpt-oss-120b and openai/gpt-oss-20b: reasoning_effort supports
# 'low'/'medium'/'high' only -- 'none' is REJECTED with a 400 (confirmed
# against a real request during live testing). Server-side default when
# omitted is 'medium'.
#
# qwen/qwen3.8-27b (NOT qwen/qwen3.6-27b -- these are two distinct,
# currently-real models; 3.8 is the newer one and is the one with both
# JSON Schema Mode and reasoning_effort support, confirmed via Groq's own
# docs) additionally supports low/medium/high on top of its default
# 'none'/'default'. Older qwen3-family models (qwen3-32b, qwen3.6-27b)
# only support 'none' to disable reasoning or 'default'/null for the
# model's own default -- NOT low/medium/high.
#
# qwen/qwen3.8-27b is registered here (so it's ready to use) but is NOT
# in config.py's default LLM_FALLBACK_MODELS -- Groq currently lists it
# as a PREVIEW model (console.groq.com/docs/models), and Groq's own
# guidance is that preview models are for evaluation and can be
# discontinued at short notice, which is not what you want silently
# in a production fallback chain. It remains available as an explicit
# opt-in: LLM_FALLBACK_MODELS=openai/gpt-oss-20b,qwen/qwen3.8-27b.
@dataclass(frozen=True)
class ModelCapabilities:
    supports_strict_json_schema: bool
    supports_reasoning_effort: bool
    reasoning_effort_values: frozenset = frozenset()


MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    "openai/gpt-oss-120b": ModelCapabilities(
        supports_strict_json_schema=True,
        supports_reasoning_effort=True,
        reasoning_effort_values=frozenset({"low", "medium", "high"}),
    ),
    "openai/gpt-oss-20b": ModelCapabilities(
        supports_strict_json_schema=True,
        supports_reasoning_effort=True,
        reasoning_effort_values=frozenset({"low", "medium", "high"}),
    ),
    "qwen/qwen3.8-27b": ModelCapabilities(
        supports_strict_json_schema=True,
        supports_reasoning_effort=True,
        reasoning_effort_values=frozenset({"none", "low", "medium", "high"}),
    ),
}

# Any model not in the registry above (a typo in LLM_MODEL, or a model
# added to Groq after this registry was last updated) gets the most
# conservative, widely-compatible capability set -- no strict schema, no
# reasoning_effort -- rather than guessing. This degrades gracefully to
# plain json_object mode instead of sending a parameter the model might
# reject outright.
_DEFAULT_CAPABILITIES = ModelCapabilities(
    supports_strict_json_schema=False,
    supports_reasoning_effort=False,
)


def get_model_capabilities(model: str) -> ModelCapabilities:
    return MODEL_CAPABILITIES.get(model, _DEFAULT_CAPABILITIES)


def _reasoning_kwargs(model: str) -> dict:
    """
    reasoning_effort is only ever sent when the target model's capability
    entry says it's supported AND the configured value is one that model
    actually accepts -- Groq rejects an unsupported value with a 400
    ("Values outside a model's supported set are rejected with a 400",
    per Groq's own API reference), so guessing wrong here would introduce
    a NEW failure mode rather than fix one.
    """
    caps = get_model_capabilities(model)
    if not caps.supports_reasoning_effort:
        return {}
    effort = settings.llm_reasoning_effort
    if effort not in caps.reasoning_effort_values:
        logger.warning(
            "LLM_REASONING_EFFORT=%r is not valid for model %s (valid: %s) -- "
            "omitting reasoning_effort for this call rather than risk a 400",
            effort, model, sorted(caps.reasoning_effort_values),
        )
        return {}
    return {"reasoning_effort": effort}


_RETRY_AFTER_PATTERN = re.compile(r"try again in ([\d.]+)\s*s", re.IGNORECASE)


def _bound_retry_after(seconds: float) -> float:
    return max(0.5, min(seconds, 30.0)) + 0.5  # small safety margin


def _parse_retry_after_from_body(error_str: str) -> Optional[float]:
    """
    Groq's 429 body includes a human-readable hint like "Please try
    again in 24.3525s." Used only when no structured Retry-After HTTP
    header is available (see _extract_retry_after_seconds, which is
    preferred and tries the header first) -- e.g. against a mocked
    client in tests, or a provider response that omits the header.
    Bounded to a sane range so a malformed or wildly large hint can't
    stall a request indefinitely.
    """
    match = _RETRY_AFTER_PATTERN.search(error_str)
    if not match:
        return None
    try:
        seconds = float(match.group(1))
    except ValueError:
        return None
    return _bound_retry_after(seconds)


def _extract_retry_after_seconds(error: Exception, error_str: str) -> Optional[float]:
    """
    Prefer the standard HTTP `Retry-After` response header -- a
    structured, provider-agnostic signal -- over parsing Groq's
    human-readable body text. Groq documents Retry-After alongside its
    rate-limit-remaining/reset headers (console.groq.com/docs/rate-limits);
    reading the header first means this also works correctly against
    any other OpenAI-compatible provider that sends a standard
    Retry-After header but phrases its body text differently. Falls
    back to _parse_retry_after_from_body only when no usable header is
    present.
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        raw = None
        try:
            raw = headers.get("retry-after") or headers.get("Retry-After")
        except AttributeError:
            pass
        if raw:
            try:
                return _bound_retry_after(float(raw))
            except (TypeError, ValueError):
                pass
    return _parse_retry_after_from_body(error_str)


# --- Strict-schema 400 disambiguation ---------------------------------------
#
# Groq's json_validate_failed error code (confirmed against a real
# observed 400 during live testing) means the MODEL'S GENERATED OUTPUT
# failed to validate against the schema -- the response carries a
# `failed_generation` field showing what the model produced, and the
# message text itself says "adjust your prompt", i.e. it blames the
# generation, driven by the prompt, not the schema definition. That is
# a legitimate, retryable-with-a-different-response-format condition
# (see MalformedOutputError below).
#
# A schema/request that the provider rejects OUTRIGHT -- the schema
# itself is malformed, uses an unsupported JSON Schema construct, or
# the response_format parameter itself is invalid -- is a DIFFERENT
# failure mode that must never be silently downgraded: retrying with
# json_object/lenient mode would just keep working around a real bug in
# our own schema (llm/schemas.py) forever instead of ever surfacing it.
# These are recognized by request-time rejection language that talks
# about the schema/request shape itself, not the model's output.
_SCHEMA_REQUEST_REJECTED_PATTERNS = (
    "invalid schema",
    "invalid json_schema",
    "invalid 'response_format'",
    "invalid response_format",
    "schema must be",
    "does not match json schema specification",
    "invalid_schema",
)


def _is_schema_request_rejected(error_str: str) -> bool:
    return any(p in error_str for p in _SCHEMA_REQUEST_REJECTED_PATTERNS)


_MODEL_UNAVAILABLE_PATTERNS = (
    "model_not_found", "does not exist", "model not found",
    "no such model", "unknown model", "has been decommissioned",
)


def _is_model_unavailable(error_str: str) -> bool:
    return any(p in error_str for p in _MODEL_UNAVAILABLE_PATTERNS)


# Exact status-code TOKEN matching (word-boundary regex), not bare
# substring matching. This is the direct fix for a real bug: Groq's 429
# body includes token-count fields like "Requested 4016", and a bare
# `"401" in error_str` check matched the "401" inside "4016", causing a
# rate limit to be misclassified as an authentication failure. `\b401\b`
# requires a non-digit/non-word boundary on both sides, so it does not
# match "401" embedded inside a longer number like "4016" or "14019".
# This is the fallback-only path (used when no structured status_code
# is available, e.g. a raw network error with no HTTP response at all)
# -- _extract_status_code's structured path above is still preferred
# whenever an SDK exception exposes a real status_code.
_TOKEN_401 = re.compile(r"\b401\b")
_TOKEN_403 = re.compile(r"\b403\b")
_TOKEN_429 = re.compile(r"\b429\b")
_TOKEN_400 = re.compile(r"\b400\b")
_TOKEN_404 = re.compile(r"\b404\b")
_TOKEN_500 = re.compile(r"\b500\b")
_TOKEN_502 = re.compile(r"\b502\b")
_TOKEN_503 = re.compile(r"\b503\b")
_TOKEN_504 = re.compile(r"\b504\b")


def _extract_status_code(error: Exception) -> Optional[int]:
    """
    Prefer the SDK's own structured status_code over text-matching the
    stringified error. groq.APIStatusError (the base for
    BadRequestError/RateLimitError/AuthenticationError/etc, confirmed by
    reading the installed groq SDK's source) sets `.status_code` directly
    from the real HTTP response -- an unambiguous, authoritative signal
    that doesn't depend on parsing text at all.
    """
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def _classify_error(error: Exception) -> LLMProviderError:
    """Convert generic exceptions to classified LLM errors."""
    if isinstance(error, LLMProviderError):
        return error

    error_str = str(error).lower()
    status_code = _extract_status_code(error)

    # Preferred path: an authoritative HTTP status code straight from the
    # SDK, not parsed from text at all.
    if status_code is not None:
        if status_code == 404 or _is_model_unavailable(error_str):
            return ModelUnavailableError(f"Model unavailable: {error}")
        if status_code in (401, 403):
            return AuthenticationError(f"Authentication failed: {error}")
        if status_code == 429:
            return RateLimitError(
                f"Rate limit exceeded: {error}",
                retry_after_seconds=_extract_retry_after_seconds(error, error_str),
            )
        if status_code in (500, 502, 503, 504):
            return ProviderUnavailableError(f"Provider server error: {error}")
        if status_code == 400:
            # See _is_schema_request_rejected / json_validate_failed
            # comments above -- schema/request rejection must be
            # checked FIRST (never eligible for model rotation), then
            # the more specific generation-validation-failure signal,
            # then the generic fallback.
            if _is_schema_request_rejected(error_str):
                return ValidationError(f"Bad request (schema/request rejected, not a model output issue): {error}")
            if "json_validate_failed" in error_str or "invalid json" in error_str:
                return MalformedOutputError(f"Invalid JSON from model: {error}")
            return ValidationError(f"Bad request: {error}")
        # Any other/unhandled structured status code: fall through to
        # the text-based classification below, same as if no status
        # code were available at all.

    # Fallback: no usable structured status code was available (e.g. a
    # raw network-level timeout that never got an HTTP response). Same
    # specific-before-generic ordering principle as above -- explicit,
    # unambiguous signals first, broad checks last -- but now using
    # exact status-code TOKEN matching (see _TOKEN_* above) instead of
    # bare substring matching, which is what let "401" match inside
    # "4016" in the first place.

    if _TOKEN_404.search(error_str) or _is_model_unavailable(error_str):
        return ModelUnavailableError(f"Model unavailable: {error}")

    if _TOKEN_401.search(error_str) or _TOKEN_403.search(error_str) or "authentication" in error_str or "invalid api key" in error_str:
        return AuthenticationError(f"Authentication failed: {error}")

    if _TOKEN_429.search(error_str) or "rate_limit" in error_str or "too many requests" in error_str:
        return RateLimitError(
            f"Rate limit exceeded: {error}",
            retry_after_seconds=_extract_retry_after_seconds(error, error_str),
        )

    if _TOKEN_500.search(error_str) or _TOKEN_502.search(error_str) or _TOKEN_503.search(error_str) or _TOKEN_504.search(error_str):
        return ProviderUnavailableError(f"Provider server error: {error}")

    if _is_schema_request_rejected(error_str):
        return ValidationError(f"Bad request (schema/request rejected, not a model output issue): {error}")

    if "json_validate_failed" in error_str or "invalid json" in error_str:
        return MalformedOutputError(f"Invalid JSON from model: {error}")

    if _TOKEN_400.search(error_str):
        return ValidationError(f"Bad request: {error}")

    if "timeout" in error_str or "timed out" in error_str:
        return TimeoutError(f"Request timeout: {error}")

    # Unknown error - not retryable by default
    return LLMProviderError(str(error), ClassificationFailureReason.UNKNOWN_ERROR, retryable=False)


def _wait_strategy(retry_state):
    """
    Honor Groq's own suggested retry-after time for rate limits when we
    have one (see RateLimitError.retry_after_seconds /
    _parse_retry_after); otherwise fall back to the same bounded
    exponential backoff + jitter as before.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, RateLimitError) and exc.retry_after_seconds is not None:
        return exc.retry_after_seconds
    return wait_exponential(multiplier=2, min=1, max=10)(retry_state) + wait_random(0, 2)(retry_state)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=_wait_strategy,
    retry=retry_if_exception_type(
        (RateLimitError, TimeoutError, ProviderUnavailableError)
    )
)
def _chat_with_retry(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: Optional[dict] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """
    Single retry wrapper with coordinated backoff, for ONE specific
    model. Only retries rate limits, timeouts, and provider errors on
    THIS model -- does not retry malformed output or validation errors
    (those are model/request issues, not transient), and does not know
    anything about other models -- model selection/fallback is the
    router's job (call_llm_json/call_llm_text below), not this
    function's.
    """
    try:
        client = _get_client()
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        
        response = client.chat.completions.create(**kwargs)
        
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise MalformedOutputError("LLM returned empty content")
        
        return content.strip()
        
    except Exception as e:
        # Classify and re-raise as appropriate error type
        classified = _classify_error(e)
        logger.warning(f"LLM call failed: model={model} reason={classified.reason.value} - {_redact_secrets(str(e))}")
        raise classified


# ---------------------------------------------------------------------------
# Model routing / fallback
# ---------------------------------------------------------------------------
#
#   PRIMARY MODEL (settings.llm_model)
#       |
#   strict JSON schema (if caller passed one AND model supports it)
#       | MalformedOutputError -> try next response mode, SAME model
#       | RateLimit/Timeout/ProviderUnavailable -> next MODEL (this
#       |   model/account is transiently overloaded -- trying a
#       |   different response_format on the SAME model wastes a call
#       |   against the SAME rate limit)
#       | Authentication/Validation -> raise immediately, no fallback
#       v
#   json_object mode (or the first attempt, if no schema/unsupported)
#       | same branching as above
#       v
#   lenient / no response_format constraint (last resort on this model)
#       | MalformedOutputError here means this model, in every response
#       |   mode we have, could not produce valid JSON for this prompt --
#       |   NOW it's reasonable to try a different model
#       | RateLimit/Timeout/ProviderUnavailable -> next MODEL
#       v
#   (repeat for each configured fallback model, in order)
#       v
#   all models/modes exhausted -> raise the last classified error
#
# Bounded total attempt budget: at most 3 response-mode attempts per
# model (each of which may itself retry internally up to 3x for
# transient errors, per _chat_with_retry's tenacity policy) -- but a
# rate-limit/timeout/provider-unavailable result on the FIRST response
# mode tried for a model skips straight to the next model rather than
# also trying the other 2 response modes against the same
# still-overloaded model, so the common "account is rate-limited"
# failure mode costs at most 3 attempts per model, not 9. This was a
# deliberate design choice after seeing a real rate-limited test run
# where the account's whole 8000 TPM budget was gone within 2-3 calls --
# a naive "try every mode on every model with full retries" design would
# have made that materially worse, not better.

_FAILOVER_ELIGIBLE_TRANSIENT = (RateLimitError, TimeoutError, ProviderUnavailableError)

# Bounds the true worst-case total HTTP call count, not just the
# router's own per-(model,mode) attempt counter -- see
# test_bounded_total_attempts_even_with_many_fallback_models in
# tests/test_provider_routing.py for the real bug this caught: capping
# only a router-level "attempts" counter still let a long
# LLM_FALLBACK_MODELS list multiply out to far more actual HTTP calls
# than intended, because each (model, mode) attempt can itself retry up
# to 3x internally via tenacity before _chat_with_retry ever returns
# control to the router. Capping the NUMBER OF MODELS actually tried
# (regardless of how many are configured) bounds the true worst case
# cleanly: at most _MAX_MODELS_TO_TRY models x up to 3 response modes x
# up to 3 tenacity attempts each = 27 raw HTTP calls in the most
# pathological blended-failure case, and typically far fewer in
# practice (a transient error skips straight to the next model without
# trying the other response modes against the same one).
_MAX_MODELS_TO_TRY = 3
_MAX_TOTAL_ATTEMPTS = 12  # secondary belt-and-suspenders cap on router-level (model,mode) attempts


@dataclass
class LLMJsonResult:
    """
    Return type of call_llm_json -- deliberately NOT a bare dict, so
    model_used is never silently lost or left misleadingly equal to
    settings.llm_model when a fallback model actually produced the
    result (a real, previously-true bug: model_used used to just be
    hardcoded to settings.llm_model regardless of what actually
    answered the request).
    """
    data: dict[str, Any]
    model_used: str
    requested_model: str
    attempts: int  # count of distinct (model, response_mode) attempts made by the router
    used_fallback: bool
    response_mode: str  # "strict_schema" | "json_object" | "lenient"


@dataclass
class LLMTextResult:
    """Return type of call_llm_text -- see LLMJsonResult for why this
    isn't a bare str."""
    text: str
    model_used: str
    requested_model: str
    attempts: int
    used_fallback: bool


def _models_to_try() -> list[str]:
    """
    Primary + configured fallbacks: order preserved, duplicates removed
    (an operator listing the primary again inside LLM_FALLBACK_MODELS,
    or repeating a fallback, must never cause the router to try the
    same model twice while a different configured model goes untried),
    then capped at _MAX_MODELS_TO_TRY regardless of how many are
    configured -- see _MAX_MODELS_TO_TRY's own comment for the exact
    bound this gives.
    """
    seen: set[str] = set()
    ordered_unique: list[str] = []
    for m in [settings.llm_model] + settings.llm_fallback_models_list:
        if m not in seen:
            seen.add(m)
            ordered_unique.append(m)
    if len(ordered_unique) > _MAX_MODELS_TO_TRY:
        logger.warning(
            "LLM_FALLBACK_MODELS configures %d distinct models but only the "
            "first %d will be tried per call (models=%s) -- trim the list "
            "if this is unexpected.",
            len(ordered_unique), _MAX_MODELS_TO_TRY, ordered_unique,
        )
    return ordered_unique[:_MAX_MODELS_TO_TRY]


def call_llm_text(
    system_prompt: str,
    human_prompt: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    operation: str = "unknown",
) -> LLMTextResult:
    """
    Free-text completion with coordinated retry AND model fallback.

    Raises LLMProviderError subclasses on failure - callers should catch
    and handle appropriately (fallback, human review, etc).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]
    temp = temperature if temperature is not None else settings.llm_temperature
    tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

    total_attempts = 0
    last_error: Optional[LLMProviderError] = None

    for model_index, model in enumerate(_models_to_try()):
        if total_attempts >= _MAX_TOTAL_ATTEMPTS:
            break
        is_fallback = model_index > 0
        total_attempts += 1
        try:
            content = _chat_with_retry(
                messages, model=model, temperature=temp, max_tokens=tokens,
                reasoning_effort=_reasoning_kwargs(model).get("reasoning_effort"),
            )
            logger.info(
                "llm_call operation=%s model=%s requested_model=%s fallback=%s attempts=%d outcome=success",
                operation, model, settings.llm_model, is_fallback, total_attempts,
            )
            return LLMTextResult(
                text=content, model_used=model, requested_model=settings.llm_model,
                attempts=total_attempts, used_fallback=is_fallback,
            )
        except ModelUnavailableError as e:
            last_error = e
            if model_index == 0:
                # Primary model 404/not-found is very likely an operator
                # configuration error (a typo, or a model ID that's since
                # been deprecated) -- fail loudly rather than silently
                # routing around it.
                logger.error(
                    "llm_call operation=%s PRIMARY model=%s is unavailable "
                    "(likely a misconfigured/deprecated model id) -- "
                    "failing, not rotating to a fallback model: %s",
                    operation, model, e,
                )
                raise
            logger.warning(
                "llm_call operation=%s FALLBACK model=%s is unavailable -- "
                "skipping to the next configured model: %s",
                operation, model, e,
            )
            continue
        except _FAILOVER_ELIGIBLE_TRANSIENT as e:
            last_error = e
            logger.warning(
                "llm_call operation=%s model=%s outcome=failed reason=%s fallback_eligible=true -- trying next model",
                operation, model, type(e).__name__,
            )
            continue
        except LLMProviderError as e:
            # Authentication / validation / malformed-output-with-no-format:
            # not eligible for model rotation, stop here.
            logger.error("llm_call operation=%s model=%s outcome=failed reason=%s fallback_eligible=false", operation, model, type(e).__name__)
            raise

    logger.error("llm_call operation=%s all models exhausted models=%s total_attempts=%d", operation, _models_to_try(), total_attempts)
    raise last_error


def call_llm_json(
    system_prompt: str,
    human_prompt: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    json_schema: Optional[dict] = None,
    operation: str = "unknown",
) -> LLMJsonResult:
    """
    JSON completion with strict-schema-then-json_object-then-lenient
    fallback PER MODEL, and model-to-model fallback across
    settings.llm_model + settings.llm_fallback_models_list for
    transient failures. See the routing diagram in this module's
    comments above call_llm_text for the exact decision flow.

    `json_schema`: a strict JSON Schema dict (see llm/schemas.py) to use
    when the selected model supports strict structured output
    (ModelCapabilities.supports_strict_json_schema). Only ever sent to a
    model whose capability entry says it's supported -- never sent
    blindly, and never used to make the schema MORE restrictive than
    the caller's actual domain model (see llm/schemas.py's own
    docstring for the specific enum-vs-plain-string design choice this
    depends on).

    `operation`: a short label ("classifier"/"planner"/"responder") for
    log correlation -- not sent to the provider, purely observability.

    Raises LLMProviderError subclasses on failure -- never silently
    returns a wrong/partial result.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]
    temp = temperature if temperature is not None else settings.llm_temperature
    tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

    total_attempts = 0
    last_error: Optional[LLMProviderError] = None

    for model_index, model in enumerate(_models_to_try()):
        is_fallback = model_index > 0
        caps = get_model_capabilities(model)
        reasoning_effort = _reasoning_kwargs(model).get("reasoning_effort")

        attempt_plan: list[tuple[str, Optional[dict]]] = []
        if json_schema is not None and caps.supports_strict_json_schema:
            attempt_plan.append(("strict_schema", {"type": "json_schema", "json_schema": json_schema}))
        attempt_plan.append(("json_object", {"type": "json_object"}))
        attempt_plan.append(("lenient", None))

        for mode_name, response_format in attempt_plan:
            if total_attempts >= _MAX_TOTAL_ATTEMPTS:
                logger.error("llm_call operation=%s hit max total attempts (%d) -- stopping", operation, _MAX_TOTAL_ATTEMPTS)
                raise last_error or LLMProviderError("Max LLM attempts exhausted", ClassificationFailureReason.UNKNOWN_ERROR, retryable=False)

            total_attempts += 1
            try:
                content = _chat_with_retry(
                    messages, model=model, temperature=temp, max_tokens=tokens,
                    response_format=response_format, reasoning_effort=reasoning_effort,
                )
                data = _parse_json(content)
                logger.info(
                    "llm_call operation=%s model=%s requested_model=%s fallback=%s attempts=%d mode=%s outcome=success",
                    operation, model, settings.llm_model, is_fallback, total_attempts, mode_name,
                )
                return LLMJsonResult(
                    data=data, model_used=model, requested_model=settings.llm_model,
                    attempts=total_attempts, used_fallback=is_fallback, response_mode=mode_name,
                )
            except MalformedOutputError as e:
                # This response mode couldn't produce valid JSON on this
                # model -- try the next, cheaper response mode on the
                # SAME model before giving up on it entirely.
                last_error = e
                logger.info(
                    "llm_call operation=%s model=%s mode=%s outcome=malformed -- trying next response mode on same model: %s",
                    operation, model, mode_name, e,
                )
                continue
            except ModelUnavailableError as e:
                last_error = e
                if model_index == 0:
                    logger.error(
                        "llm_call operation=%s PRIMARY model=%s is unavailable "
                        "(likely a misconfigured/deprecated model id) -- "
                        "failing, not rotating to a fallback model: %s",
                        operation, model, e,
                    )
                    raise
                logger.warning(
                    "llm_call operation=%s FALLBACK model=%s is unavailable -- "
                    "skipping to the next configured model: %s",
                    operation, model, e,
                )
                break  # out of the mode loop, continue the model loop
            except _FAILOVER_ELIGIBLE_TRANSIENT as e:
                # Transient/capacity issue -- trying a different
                # response_format on the SAME model wastes a call
                # against the same rate limit/outage. Go straight to
                # the next model.
                last_error = e
                logger.warning(
                    "llm_call operation=%s model=%s mode=%s outcome=failed reason=%s fallback_eligible=true -- trying next model",
                    operation, model, mode_name, type(e).__name__,
                )
                break  # out of the mode loop, continue the model loop
            except LLMProviderError as e:
                # Authentication or genuine request/schema validation
                # error -- never eligible for model rotation.
                logger.error(
                    "llm_call operation=%s model=%s mode=%s outcome=failed reason=%s fallback_eligible=false",
                    operation, model, mode_name, type(e).__name__,
                )
                raise
        else:
            # Every response mode on this model ended in
            # MalformedOutputError (the `continue` path, never hit
            # `break`) -- NOW it's reasonable to try a different model,
            # since this one couldn't produce valid structured output
            # in any format we offered it.
            continue

    logger.error(
        "llm_call operation=%s all models/modes exhausted models=%s total_attempts=%d last_error=%s",
        operation, _models_to_try(), total_attempts, last_error,
    )
    raise last_error or LLMProviderError("No models configured", ClassificationFailureReason.UNKNOWN_ERROR, retryable=False)


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
