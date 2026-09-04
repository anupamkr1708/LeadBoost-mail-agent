"""
Provider-failure tests (spec sections 15-16): exercise mailer_agent.llm.
provider_v2 directly -- not through the semantic classifier -- so these
prove the retry/classification policy itself, independent of anything
downstream that consumes it.

This uses a fake Groq *client* (a stand-in for the `groq` SDK object
`_get_client()` normally constructs), not the FakeLLMProvider fixture
used elsewhere in this test suite -- FakeLLMProvider fakes
provider_v2.call_llm_json/call_llm_text themselves (the interface
consumers use), which is the right level for testing consumers, but
would bypass all the retry/classification logic this file exists to
test. Here we fake one layer lower: the thing _chat_with_retry calls.

Tenacity's real wait strategy (exponential backoff + jitter) sleeps for
real between attempts by default. The autouse fixture below patches the
decorated function's `.retry.sleep` hook to a no-op so these tests run
in milliseconds and stay deterministic regardless of jitter, without
changing what's actually being tested (the number and classification of
attempts).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mailer_agent.llm import provider_v2
from mailer_agent.llm.provider_v2 import (
    AuthenticationError,
    MalformedOutputError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError as ProviderTimeoutError,
    ValidationError,
    call_llm_json,
    call_llm_text,
)


class _FakeCompletions:
    def __init__(self, queue: list):
        self._queue = list(queue)
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        if not self._queue:
            raise AssertionError(
                f"FakeGroqClient.chat.completions.create called more times "
                f"({self.call_count}) than fixtures were queued -- this "
                f"usually means a retry policy retried more than expected."
            )
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=item))])


class FakeGroqClient:
    """Stand-in for the real groq.Groq client that _get_client() builds."""

    def __init__(self, queue: list):
        self._completions = _FakeCompletions(queue)
        self.chat = SimpleNamespace(completions=self._completions)

    @property
    def call_count(self) -> int:
        return self._completions.call_count


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """
    Skip tenacity's real backoff sleep so retry tests run fast and
    deterministically.

    Tenacity attaches the Retrying instance itself to the decorated
    function as `.retry` specifically so callers can reconfigure it --
    overwriting its `.sleep` attribute (which __call__ invokes between
    attempts) is tenacity's own documented pattern for this, and is more
    reliable here than patching the tenacity.nap.sleep module function,
    since BaseRetrying captures its `sleep` callable once at construction
    time (i.e. once, at provider_v2.py's import time) rather than looking
    it up fresh on every call -- a module-level patch applied later
    wouldn't reach an already-bound instance attribute.
    """
    monkeypatch.setattr(provider_v2._chat_with_retry.retry, "sleep", lambda seconds: None)


def _install_fake_client(monkeypatch, queue: list) -> FakeGroqClient:
    fake_client = FakeGroqClient(queue)
    # Same instance returned on every call, so a shared queue is
    # correctly consumed across repeated retry attempts within one test
    # -- a fresh instance per call would silently reset the queue and
    # defeat the "N failures then success" scenarios below.
    monkeypatch.setattr(provider_v2, "_get_client", lambda: fake_client)
    return fake_client


# ---------------------------------------------------------------------------
# Retryable errors: bounded retry, eventually succeeds if it recovers
# ---------------------------------------------------------------------------

def test_rate_limit_is_retried_and_recovers(monkeypatch):
    fake_client = _install_fake_client(monkeypatch, [
        Exception("429 Too Many Requests"),
        Exception("429 Too Many Requests"),
        '{"ok": true}',
    ])

    result = call_llm_json("system", "human")

    assert result == {"ok": True}
    assert fake_client.call_count == 3


def test_persistent_rate_limit_is_bounded_not_infinite(monkeypatch):
    fake_client = _install_fake_client(monkeypatch, [
        Exception("429 Too Many Requests"),
        Exception("429 Too Many Requests"),
        Exception("429 Too Many Requests"),
    ])

    with pytest.raises(RateLimitError):
        call_llm_json("system", "human")

    assert fake_client.call_count == 3, (
        "Retry must be bounded at exactly 3 attempts (stop_after_attempt(3)), "
        f"not retried indefinitely. Got {fake_client.call_count} attempts."
    )


def test_transient_5xx_is_retried_and_recovers(monkeypatch):
    fake_client = _install_fake_client(monkeypatch, [
        Exception("503 Service Unavailable"),
        '{"ok": true}',
    ])

    result = call_llm_json("system", "human")

    assert result == {"ok": True}
    assert fake_client.call_count == 2


def test_persistent_5xx_is_bounded(monkeypatch):
    fake_client = _install_fake_client(monkeypatch, [
        Exception("502 Bad Gateway"),
        Exception("503 Service Unavailable"),
        Exception("504 Gateway Timeout"),
    ])

    with pytest.raises(ProviderUnavailableError):
        call_llm_json("system", "human")

    assert fake_client.call_count == 3


def test_timeout_is_retried_and_recovers(monkeypatch):
    fake_client = _install_fake_client(monkeypatch, [
        Exception("Request timed out"),
        '{"ok": true}',
    ])

    result = call_llm_json("system", "human")

    assert result == {"ok": True}
    assert fake_client.call_count == 2


# ---------------------------------------------------------------------------
# Non-retryable errors: exactly one attempt, fail fast
# ---------------------------------------------------------------------------

def test_authentication_error_is_not_retried(monkeypatch):
    """
    Regression test for a real bug found and fixed this session:
    _classify_error used to map 401/403 to ProviderUnavailableError,
    which IS in _chat_with_retry's retryable exception tuple -- so an
    invalid/expired API key (which fails identically on every attempt)
    was being retried 3 times, tripling the latency of a guaranteed
    failure for zero benefit. Authentication failures must fail on the
    first attempt.
    """
    fake_client = _install_fake_client(monkeypatch, [
        Exception("401 Unauthorized: invalid api key"),
    ])

    with pytest.raises(AuthenticationError):
        call_llm_json("system", "human")

    assert fake_client.call_count == 1, (
        "Authentication failures must not be retried -- got "
        f"{fake_client.call_count} attempts."
    )


def test_forbidden_error_is_not_retried(monkeypatch):
    fake_client = _install_fake_client(monkeypatch, [
        Exception("403 Forbidden"),
    ])

    with pytest.raises(AuthenticationError):
        call_llm_json("system", "human")

    assert fake_client.call_count == 1


def test_bad_request_validation_error_is_not_retried(monkeypatch):
    fake_client = _install_fake_client(monkeypatch, [
        Exception("400 Bad Request: invalid parameter"),
    ])

    with pytest.raises(ValidationError):
        call_llm_json("system", "human")

    assert fake_client.call_count == 1


def test_empty_content_within_single_attempt_is_not_retried(monkeypatch):
    """
    _chat_with_retry raises MalformedOutputError itself when the model
    returns empty content. MalformedOutputError is deliberately not in
    the retryable tuple (a model returning nothing is a model/prompt
    issue, not a transient condition retrying would fix) -- so this
    should be exactly one attempt at the _chat_with_retry layer.
    call_llm_json's own strict-then-lenient fallback (tested separately
    below) is a distinct, explicitly bounded, non-tenacity mechanism
    layered on top -- not additional tenacity retries.
    """
    fake_client = _install_fake_client(monkeypatch, ["   "])  # whitespace-only

    with pytest.raises(MalformedOutputError):
        call_llm_text("system", "human")

    assert fake_client.call_count == 1


# ---------------------------------------------------------------------------
# call_llm_json's strict -> lenient fallback: bounded at 2, not stacked with retry
# ---------------------------------------------------------------------------

def test_malformed_json_falls_back_from_strict_to_lenient_and_recovers(monkeypatch):
    fake_client = _install_fake_client(monkeypatch, [
        "not valid json at all",      # strict attempt: unparseable
        '{"ok": true}',                 # lenient attempt: succeeds
    ])

    result = call_llm_json("system", "human")

    assert result == {"ok": True}
    assert fake_client.call_count == 2


def test_malformed_json_persistent_is_bounded_at_two_not_six(monkeypatch):
    """
    If both the strict and lenient attempts return unparseable content,
    the total call count must be exactly 2 (one strict + one lenient),
    not 2*3=6 -- proving the strict/lenient fallback and tenacity's
    per-attempt retry aren't accidentally stacked together for this
    non-retryable failure mode.
    """
    fake_client = _install_fake_client(monkeypatch, [
        "not valid json at all",
        "still not valid json",
    ])

    with pytest.raises(MalformedOutputError):
        call_llm_json("system", "human")

    assert fake_client.call_count == 2, (
        f"Expected exactly 2 attempts (strict + lenient), got {fake_client.call_count} "
        "-- malformed-output fallback must not be stacked with tenacity retry."
    )


def test_call_llm_json_parses_markdown_fenced_output():
    """Pure parsing regression, no client needed -- protects a real Groq quirk."""
    from mailer_agent.llm.provider_v2 import _parse_json
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json('Here is the result: {"a": 1} -- hope that helps!') == {"a": 1}
    assert _parse_json('{"a": 1}') == {"a": 1}


# ---------------------------------------------------------------------------
# Idempotent classification -- an already-classified error isn't reclassified
# ---------------------------------------------------------------------------

def test_already_classified_error_is_not_reclassified(monkeypatch):
    """
    call_llm_json's lenient-fallback branch calls _classify_error() on
    whatever exception the lenient attempt raised. If that exception is
    already a classified LLMProviderError (e.g. a MalformedOutputError
    raised directly by _parse_json), re-running string-matching against
    its message could accidentally misclassify it a second time. This
    confirms already-classified errors pass through unchanged.
    """
    from mailer_agent.llm.provider_v2 import _classify_error
    original = MalformedOutputError("No JSON object found in output: 503 something")
    reclassified = _classify_error(original)
    assert reclassified is original, (
        "An already-classified LLMProviderError must pass through "
        "_classify_error unchanged, not be re-derived from its message "
        "text (which could coincidentally contain misleading substrings "
        "like '503')."
    )
