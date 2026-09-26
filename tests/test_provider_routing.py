"""
Model routing / fallback tests for llm/provider_v2.py's call_llm_json
and call_llm_text.

Uses the same fake-Groq-client-at-the-_get_client()-layer approach as
tests/test_provider_failures.py (see that file's module docstring for
why this layer, not FakeLLMProvider, is the right one for testing retry/
routing policy itself) -- but where that file deliberately pins
llm_fallback_models="" to keep model rotation OUT of scope, this file's
entire purpose IS model rotation, so each test sets
llm_model/llm_fallback_models explicitly via _set_models().

All 17 scenarios from the routing spec are covered below, each in its
own test function named after the scenario.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from mailer_agent.llm import provider_v2
from mailer_agent.llm.provider_v2 import (
    AuthenticationError,
    ModelUnavailableError,
    ProviderUnavailableError,
    RateLimitError,
    ValidationError,
    call_llm_json,
    call_llm_text,
)
from mailer_agent.llm.schemas import CLASSIFIER_JSON_SCHEMA


class _FakeCompletions:
    def __init__(self, queue: list):
        self._queue = list(queue)
        self.calls: list[dict] = []  # full kwargs of every create() call, in order

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._queue:
            raise AssertionError(
                f"FakeGroqClient.chat.completions.create called more times "
                f"({len(self.calls)}) than fixtures were queued."
            )
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=item))])

    @property
    def call_count(self) -> int:
        return len(self.calls)


class FakeGroqClient:
    """Stand-in for the real groq.Groq client that _get_client() builds."""

    def __init__(self, queue: list):
        self._completions = _FakeCompletions(queue)
        self.chat = SimpleNamespace(completions=self._completions)

    @property
    def call_count(self) -> int:
        return self._completions.call_count

    @property
    def calls(self) -> list[dict]:
        return self._completions.calls


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(provider_v2._chat_with_retry.retry, "sleep", lambda seconds: None)


def _install_fake_client(monkeypatch, queue: list) -> FakeGroqClient:
    fake_client = FakeGroqClient(queue)
    monkeypatch.setattr(provider_v2, "_get_client", lambda: fake_client)
    return fake_client


def _set_models(monkeypatch, primary: str, fallbacks: list[str]) -> None:
    monkeypatch.setattr(provider_v2.settings, "llm_model", primary)
    monkeypatch.setattr(provider_v2.settings, "llm_fallback_models", ",".join(fallbacks))


# ---------------------------------------------------------------------------
# 1. primary model success -> no fallback
# ---------------------------------------------------------------------------

def test_primary_success_no_fallback(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, ['{"ok": true}'])

    result = call_llm_json("sys", "human")

    assert result.data == {"ok": True}
    assert result.model_used == "openai/gpt-oss-120b"
    assert result.used_fallback is False
    assert fake.call_count == 1


# ---------------------------------------------------------------------------
# 2. primary 429 -> retries according to policy (same model, bounded)
# ---------------------------------------------------------------------------

def test_primary_429_retries_same_model_then_succeeds(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", [])
    fake = _install_fake_client(monkeypatch, [
        Exception("Error code: 429 - rate_limit_exceeded"),
        '{"ok": true}',
    ])

    result = call_llm_json("sys", "human")

    assert result.data == {"ok": True}
    assert result.model_used == "openai/gpt-oss-120b"
    assert fake.call_count == 2
    assert all(c["model"] == "openai/gpt-oss-120b" for c in fake.calls)


# ---------------------------------------------------------------------------
# 3. primary 429 exhausted -> fallback model
# ---------------------------------------------------------------------------

def test_primary_429_exhausted_falls_over_to_next_model(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, [
        Exception("Error code: 429 - rate_limit_exceeded"),
        Exception("Error code: 429 - rate_limit_exceeded"),
        Exception("Error code: 429 - rate_limit_exceeded"),  # primary exhausted (3 tenacity attempts)
        '{"ok": true}',  # fallback model succeeds on first try
    ])

    result = call_llm_json("sys", "human")

    assert result.data == {"ok": True}
    assert result.model_used == "openai/gpt-oss-20b"
    assert result.used_fallback is True
    assert fake.call_count == 4
    assert [c["model"] for c in fake.calls] == ["openai/gpt-oss-120b"] * 3 + ["openai/gpt-oss-20b"]


# ---------------------------------------------------------------------------
# 4. fallback succeeds -> returns fallback model identity
# ---------------------------------------------------------------------------

def test_fallback_model_identity_is_reported_correctly(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"])
    fake = _install_fake_client(monkeypatch, [
        Exception("503 Service Unavailable"),
        Exception("503 Service Unavailable"),
        Exception("503 Service Unavailable"),
        '{"ok": true}',
    ])

    result = call_llm_json("sys", "human")

    assert result.model_used == "openai/gpt-oss-20b"  # first fallback, not second
    assert result.requested_model == "openai/gpt-oss-120b"


# ---------------------------------------------------------------------------
# 5. all models rate-limited -> safe failure
# ---------------------------------------------------------------------------

def test_all_models_rate_limited_fails_safely(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, [Exception("429 rate_limit_exceeded")] * 6)

    with pytest.raises(RateLimitError):
        call_llm_json("sys", "human")

    assert fake.call_count == 6  # 3 attempts x 2 models -- bounded, not infinite


# ---------------------------------------------------------------------------
# 6. authentication failure -> no fallback
# ---------------------------------------------------------------------------

def test_authentication_failure_no_fallback(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, [
        Exception("401 Unauthorized: invalid api key"),
    ])

    with pytest.raises(AuthenticationError):
        call_llm_json("sys", "human")

    assert fake.call_count == 1
    assert fake.calls[0]["model"] == "openai/gpt-oss-120b"  # fallback was never even tried


# ---------------------------------------------------------------------------
# 7. bad request/schema validation -> no model rotation
# ---------------------------------------------------------------------------

def test_validation_error_no_model_rotation(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, [
        Exception("400 Bad Request: invalid parameter"),
    ])

    with pytest.raises(ValidationError):
        call_llm_json("sys", "human")

    assert fake.call_count == 1


# ---------------------------------------------------------------------------
# 8. malformed model output -> appropriate parsing/retry behavior
# ---------------------------------------------------------------------------

def test_malformed_output_exhausts_response_modes_then_fails_over(monkeypatch):
    """All 3 response modes (strict_schema, json_object, lenient) fail on
    the primary model before the router tries the fallback model."""
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, [
        "not json",       # strict_schema
        "still not json", # json_object
        "nope",           # lenient
        '{"ok": true}',   # fallback model, strict_schema succeeds
    ])

    result = call_llm_json("sys", "human", json_schema=CLASSIFIER_JSON_SCHEMA)

    assert result.model_used == "openai/gpt-oss-20b"
    assert fake.call_count == 4


# ---------------------------------------------------------------------------
# 9. timeout -> eligible failover
# ---------------------------------------------------------------------------

def test_timeout_eligible_failover(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, [
        Exception("Request timed out"), Exception("Request timed out"), Exception("Request timed out"),
        '{"ok": true}',
    ])

    result = call_llm_json("sys", "human")

    assert result.model_used == "openai/gpt-oss-20b"


# ---------------------------------------------------------------------------
# 10. provider 5xx -> eligible failover
# ---------------------------------------------------------------------------

def test_5xx_eligible_failover(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, [
        Exception("502 Bad Gateway"), Exception("502 Bad Gateway"), Exception("502 Bad Gateway"),
        '{"ok": true}',
    ])

    result = call_llm_json("sys", "human")

    assert result.model_used == "openai/gpt-oss-20b"


# ---------------------------------------------------------------------------
# 11. retry-after is honored/bounded
# ---------------------------------------------------------------------------

def test_retry_after_hint_is_used_as_wait_time(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(provider_v2._chat_with_retry.retry, "sleep", lambda seconds: waits.append(seconds))
    _set_models(monkeypatch, "openai/gpt-oss-120b", [])
    fake = _install_fake_client(monkeypatch, [
        Exception("Error code: 429 - {'error': {'message': 'Rate limit... Please try again in 5.5s.'}}"),
        '{"ok": true}',
    ])

    call_llm_json("sys", "human")

    assert len(waits) == 1
    assert 5.0 < waits[0] < 7.0  # 5.5 + safety margin -- not the fixed ~1-12s exponential backoff


def test_retry_after_hint_is_bounded_even_if_provider_suggests_a_long_wait(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(provider_v2._chat_with_retry.retry, "sleep", lambda seconds: waits.append(seconds))
    _set_models(monkeypatch, "openai/gpt-oss-120b", [])
    fake = _install_fake_client(monkeypatch, [
        Exception("Error code: 429 - please try again in 999s"),
        '{"ok": true}',
    ])

    call_llm_json("sys", "human")

    assert waits[0] <= 30.5  # _parse_retry_after_from_body's own cap


# ---------------------------------------------------------------------------
# 12. reasoning effort is sent only to models that support it
# ---------------------------------------------------------------------------

def test_reasoning_effort_sent_for_gpt_oss(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", [])
    monkeypatch.setattr(provider_v2.settings, "llm_reasoning_effort", "low")
    fake = _install_fake_client(monkeypatch, ['{"ok": true}'])

    call_llm_json("sys", "human")

    assert fake.calls[0].get("reasoning_effort") == "low"


def test_reasoning_effort_omitted_for_model_without_capability(monkeypatch):
    _set_models(monkeypatch, "some/unregistered-model", [])
    monkeypatch.setattr(provider_v2.settings, "llm_reasoning_effort", "low")
    fake = _install_fake_client(monkeypatch, ['{"ok": true}'])

    call_llm_json("sys", "human")

    assert "reasoning_effort" not in fake.calls[0]


def test_reasoning_effort_omitted_when_value_invalid_for_model(monkeypatch, caplog):
    """gpt-oss models reject 'none' outright with a 400 -- confirmed
    against a real request during live testing. If LLM_REASONING_EFFORT
    is misconfigured to a value invalid for the selected model, omit
    the parameter rather than send one guaranteed to fail."""
    _set_models(monkeypatch, "openai/gpt-oss-120b", [])
    monkeypatch.setattr(provider_v2.settings, "llm_reasoning_effort", "none")
    fake = _install_fake_client(monkeypatch, ['{"ok": true}'])

    with caplog.at_level(logging.WARNING):
        call_llm_json("sys", "human")

    assert "reasoning_effort" not in fake.calls[0]
    assert any("not valid for model" in r.message for r in caplog.records)


def test_reasoning_effort_value_that_differs_per_model_is_respected(monkeypatch):
    """qwen/qwen3.8-27b additionally accepts 'none' (its own default) --
    proves capability checking is per-model, not a single global
    allow-list."""
    _set_models(monkeypatch, "qwen/qwen3.8-27b", [])
    monkeypatch.setattr(provider_v2.settings, "llm_reasoning_effort", "none")
    fake = _install_fake_client(monkeypatch, ['{"ok": true}'])

    call_llm_json("sys", "human")

    assert fake.calls[0].get("reasoning_effort") == "none"


# ---------------------------------------------------------------------------
# 13. strict JSON Schema is used for supported models
# ---------------------------------------------------------------------------

def test_strict_schema_used_when_model_supports_it_and_schema_provided(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", [])
    fake = _install_fake_client(monkeypatch, ['{"ok": true}'])

    result = call_llm_json("sys", "human", json_schema=CLASSIFIER_JSON_SCHEMA)

    assert fake.calls[0]["response_format"]["type"] == "json_schema"
    assert result.response_mode == "strict_schema"


# ---------------------------------------------------------------------------
# 14. compatibility mode is selected only when necessary
# ---------------------------------------------------------------------------

def test_compatibility_mode_when_model_lacks_strict_schema_support(monkeypatch):
    _set_models(monkeypatch, "some/unregistered-model", [])
    fake = _install_fake_client(monkeypatch, ['{"ok": true}'])

    result = call_llm_json("sys", "human", json_schema=CLASSIFIER_JSON_SCHEMA)

    # Model doesn't support strict schema -- falls straight to
    # json_object, never sends a schema shape it can't handle.
    assert fake.calls[0]["response_format"]["type"] == "json_object"
    assert result.response_mode == "json_object"


def test_no_schema_requested_uses_json_object_even_on_capable_model(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", [])
    fake = _install_fake_client(monkeypatch, ['{"ok": true}'])

    result = call_llm_json("sys", "human")  # no json_schema passed

    assert fake.calls[0]["response_format"]["type"] == "json_object"
    assert result.response_mode == "json_object"


# ---------------------------------------------------------------------------
# 15. primary model is not permanently mutated by one failure
# ---------------------------------------------------------------------------

def test_primary_model_setting_unchanged_after_fallback(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, [
        Exception("503"), Exception("503"), Exception("503"),
        '{"ok": true}',
    ])

    call_llm_json("sys", "human")

    assert provider_v2.settings.llm_model == "openai/gpt-oss-120b"  # never mutated by the failure

    # And the NEXT independent call goes back to trying the primary first.
    fake2 = _install_fake_client(monkeypatch, ['{"second": true}'])
    result2 = call_llm_json("sys", "human")
    assert result2.model_used == "openai/gpt-oss-120b"


# ---------------------------------------------------------------------------
# 16. no infinite fallback loops
# ---------------------------------------------------------------------------

def test_bounded_total_attempts_even_with_many_fallback_models(monkeypatch):
    _set_models(monkeypatch, "m0", ["m1", "m2", "m3", "m4", "m5"])
    fake = _install_fake_client(monkeypatch, [Exception("503")] * 50)

    with pytest.raises(ProviderUnavailableError):
        call_llm_json("sys", "human")

    # True worst-case bound: _MAX_MODELS_TO_TRY models x up to 3 tenacity
    # attempts each (a 503 skips the other response modes and moves
    # straight to the next model, so response-mode count doesn't
    # multiply in for this specific all-transient-failure scenario).
    max_real_http_calls = provider_v2._MAX_MODELS_TO_TRY * 3
    assert fake.call_count <= max_real_http_calls, (
        f"Router must never exceed the true worst-case HTTP call bound "
        f"({max_real_http_calls}) regardless of how many fallback models "
        f"are configured -- got {fake.call_count} calls. (Six models were "
        f"configured; only the first {provider_v2._MAX_MODELS_TO_TRY} "
        f"should ever be tried.)"
    )
    models_actually_tried = {c["model"] for c in fake.calls}
    assert len(models_actually_tried) <= provider_v2._MAX_MODELS_TO_TRY


def test_models_to_try_is_truncated_to_the_cap(monkeypatch, caplog):
    _set_models(monkeypatch, "m0", ["m1", "m2", "m3", "m4", "m5"])

    with caplog.at_level(logging.WARNING):
        models = provider_v2._models_to_try()

    assert len(models) == provider_v2._MAX_MODELS_TO_TRY
    assert models == ["m0", "m1", "m2"][:provider_v2._MAX_MODELS_TO_TRY]
    assert any("only the first" in r.message for r in caplog.records)


def test_models_to_try_deduplicates_preserving_order(monkeypatch):
    """
    Config like LLM_MODEL=openai/gpt-oss-120b + LLM_FALLBACK_MODELS
    listing the same model again (a plausible operator mistake, or the
    primary appearing a second time further down a hand-edited list)
    must never cause the router to try the same model twice while a
    different configured model goes untried.
    """
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])

    models = provider_v2._models_to_try()

    assert models == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert len(models) == len(set(models)), "must never contain a duplicate model"


def test_models_to_try_deduplicates_a_repeated_fallback(monkeypatch):
    _set_models(monkeypatch, "m0", ["m1", "m1", "m2"])

    models = provider_v2._models_to_try()

    assert models == ["m0", "m1", "m2"]


def test_deduplication_actually_prevents_a_wasted_duplicate_call(monkeypatch):
    """End-to-end proof, not just a unit check on _models_to_try: with
    the primary duplicated in the fallback list, a failure on the
    primary must escalate straight to the genuinely-different fallback
    model, never retry the primary a second time under the guise of
    'trying the next model'."""
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, [
        Exception("503 service unavailable"),
        Exception("503 service unavailable"),
        Exception("503 service unavailable"),
        '{"ok": true}',
    ])

    result = call_llm_json("sys", "human")

    assert result.model_used == "openai/gpt-oss-20b"
    models_tried = {c["model"] for c in fake.calls}
    assert models_tried == {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}


# ---------------------------------------------------------------------------
# Model-not-found (404) handling: primary fails hard (likely a
# misconfiguration), a fallback model is skipped in favor of the next one.
# ---------------------------------------------------------------------------

def test_primary_model_not_found_fails_immediately_no_rotation(monkeypatch):
    """A 404 on the PRIMARY model is very likely an operator
    configuration error (a typo, or a model ID that's since been
    deprecated/renamed) -- surface it loudly rather than silently
    routing around it, even though other models are configured."""
    _set_models(monkeypatch, "openai/gpt-oss-999b-typo", ["openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, [
        Exception("Error code: 404 - model_not_found: the model does not exist"),
    ])

    with pytest.raises(ModelUnavailableError):
        call_llm_json("sys", "human")

    assert fake.call_count == 1
    assert fake.calls[0]["model"] == "openai/gpt-oss-999b-typo"  # fallback never even tried


def test_fallback_model_not_found_is_skipped_to_next_model(monkeypatch):
    """A 404 on a FALLBACK model is far more forgivable -- it may have
    been deprecated/renamed since it was configured -- skip it and
    continue to the next configured model rather than failing the whole
    request."""
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-999b-typo", "openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, [
        Exception("Error code: 503 - service unavailable"),
        Exception("Error code: 503 - service unavailable"),
        Exception("Error code: 503 - service unavailable"),
        Exception("Error code: 404 - model_not_found"),  # fallback #1: unavailable, skip it
        '{"ok": true}',  # fallback #2: succeeds
    ])

    result = call_llm_json("sys", "human")

    assert result.model_used == "openai/gpt-oss-20b"
    models_tried = [c["model"] for c in fake.calls]
    assert models_tried == (
        ["openai/gpt-oss-120b"] * 3 + ["openai/gpt-oss-999b-typo", "openai/gpt-oss-20b"]
    )


def test_fallback_model_not_found_via_call_llm_text_also_skips(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-999b-typo", "openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, [
        Exception("Error code: 503 - service unavailable"),
        Exception("Error code: 503 - service unavailable"),
        Exception("Error code: 503 - service unavailable"),
        Exception("Error code: 404 - model_not_found"),
        "a plain text summary",
    ])

    result = call_llm_text("sys", "human")

    assert result.model_used == "openai/gpt-oss-20b"
    assert result.text == "a plain text summary"


# ---------------------------------------------------------------------------
# call_llm_text routing (minimal -- the JSON tests above already prove the
# shared routing/classification/retry-after machinery; these two just
# confirm call_llm_text's own thin wrapper around it actually works).
# ---------------------------------------------------------------------------

def test_call_llm_text_primary_success_no_fallback(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, ["a plain text summary"])

    result = call_llm_text("sys", "human")

    assert result.text == "a plain text summary"
    assert result.model_used == "openai/gpt-oss-120b"
    assert result.used_fallback is False
    assert fake.call_count == 1


def test_call_llm_text_transient_primary_failure_falls_back(monkeypatch):
    _set_models(monkeypatch, "openai/gpt-oss-120b", ["openai/gpt-oss-20b"])
    fake = _install_fake_client(monkeypatch, [
        Exception("503 Service Unavailable"),
        Exception("503 Service Unavailable"),
        Exception("503 Service Unavailable"),
        "fallback model's summary",
    ])

    result = call_llm_text("sys", "human")

    assert result.text == "fallback model's summary"
    assert result.model_used == "openai/gpt-oss-20b"
    assert result.used_fallback is True


# ---------------------------------------------------------------------------
# 17. no secrets in logs
# ---------------------------------------------------------------------------

def test_no_api_key_in_log_output(monkeypatch, caplog):
    """
    Defense in depth: even in the (real-world-unlikely) case that a
    provider error message happened to echo back the configured API
    key, _redact_secrets must strip it before it reaches a log line.
    """
    _set_models(monkeypatch, "openai/gpt-oss-120b", [])
    secret = "gsk_super_secret_key_value_12345"
    monkeypatch.setattr(provider_v2.settings, "groq_api_key", secret)
    fake = _install_fake_client(monkeypatch, [
        Exception(f"401 Unauthorized: invalid api key {secret}"),
    ])

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(AuthenticationError):
            call_llm_json("sys", "human")

    for record in caplog.records:
        assert secret not in record.getMessage(), (
            "The real API key value must never appear in log output, even "
            "when it happens to be echoed in a provider error message."
        )
