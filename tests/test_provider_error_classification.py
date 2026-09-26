"""
Regression tests for llm/provider_v2.py's error classification, built
directly from real Groq responses captured during live testing (not
hypothetical inputs).

Two real bugs found this way (Revision 1):

1. A 429 rate-limit response was misclassified as an authentication
   failure, because Groq's error body includes token-count fields like
   "Requested 4016" -- and a bare `"401" in error_str` substring check
   matched the "401" inside "4016". Fixed two ways, both covered below:
   (a) prefer the SDK's own structured status_code attribute over text
   matching entirely, and (b) even in the text-only fallback path (no
   status_code available), use word-boundary token regexes (\\b401\\b)
   instead of bare substring checks, so "4016" no longer matches "401"
   at all -- Revision 1 shipped with this fallback path still known to
   misclassify (documented as a "known limitation" test), which was
   itself flagged as unacceptable in review: a passing test must never
   encode wrong behavior as expected. Fixed and reasserted correctly
   below.

2. Groq's json_validate_failed error (a structured-output validation
   failure) is always reported as a 400 Bad Request. The original
   _classify_error checked the generic "400" branch before the specific
   "json_validate_failed" branch, so it always returned ValidationError
   -- never MalformedOutputError -- and call_llm_json's strict-then-
   lenient fallback (which only catches MalformedOutputError) never
   actually ran for exactly the case it exists to handle.

Revision 2 additions:

3. json_validate_failed must NOT be treated as "the model's output was
   bad" when the error actually indicates the SCHEMA/REQUEST itself was
   rejected (a bug in our own llm/schemas.py, or an unsupported JSON
   Schema construct) -- that must stay a ValidationError with no model
   rotation, or a real schema bug would be silently papered over by
   falling back to json_object/lenient mode forever instead of ever
   surfacing.

4. Retry-after must prefer the structured HTTP `Retry-After` response
   header over parsing Groq's human-readable body text.

5. A model-not-found/404 condition gets its own classification
   (ModelUnavailableError), distinct from a generic 5xx or 429.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mailer_agent.llm.provider_v2 import (
    AuthenticationError,
    MalformedOutputError,
    ModelUnavailableError,
    ProviderUnavailableError,
    RateLimitError,
    ValidationError,
    _classify_error,
    _extract_retry_after_seconds,
    _parse_retry_after_from_body,
    _wait_strategy,
)


class _FakeAPIStatusError(Exception):
    """Stands in for groq.APIStatusError/BadRequestError/RateLimitError/etc,
    which all expose a real `.status_code: int` from the HTTP response,
    and (via `.response`) the underlying httpx.Response with headers."""
    def __init__(self, message: str, status_code: int, headers: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        if headers is not None:
            self.response = SimpleNamespace(status_code=status_code, headers=headers)


# The exact real 429 body captured during live testing (org id/numbers
# unchanged from the actual observed failure).
REAL_RATE_LIMIT_MESSAGE = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-120b` in organization `org_01m1951zpxe269h88hg54y90gk` "
    "service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used "
    "7231, Requested 4016. Please try again in 24.3525s. Need more tokens? "
    "Upgrade to Dev Tier today at https://console.groq.com/settings/billing', "
    "'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)

# The exact real 400 body captured during live testing.
REAL_JSON_VALIDATE_FAILED_MESSAGE = (
    "Error code: 400 - {'error': {'message': \"Failed to validate JSON. "
    "Please adjust your prompt. See 'failed_generation' for more details.\", "
    "'type': 'invalid_request_error', 'code': 'json_validate_failed', "
    "'failed_generation': ''}}"
)


class TestRealRateLimitMisclassification:
    """The core regression: '401' as a substring of '4016' must never
    cause a rate limit to be classified as an auth failure -- in EITHER
    the structured status_code path or the text-only fallback path."""

    def test_string_contains_401_as_a_substring_of_an_unrelated_number(self):
        # Sanity check proving the collision is real, not hypothetical.
        assert "401" in REAL_RATE_LIMIT_MESSAGE.lower()

    def test_structured_status_code_429_classifies_as_rate_limit_not_auth(self):
        error = _FakeAPIStatusError(REAL_RATE_LIMIT_MESSAGE, status_code=429)

        classified = _classify_error(error)

        assert isinstance(classified, RateLimitError), (
            f"A real 429 with status_code=429 must classify as RateLimitError, "
            f"got {type(classified).__name__} -- likely misled by '401' "
            f"appearing inside 'Requested 4016' in the error text."
        )
        assert not isinstance(classified, AuthenticationError)

    def test_text_only_fallback_now_correctly_classifies_as_rate_limit(self):
        """
        Revision 2: the text-only fallback path (no structured
        status_code at all) now ALSO gets this right, via word-boundary
        token matching (\\b401\\b) instead of bare substring matching.
        "4016" no longer matches the "401" token. This test previously
        asserted the WRONG behavior (AuthenticationError) as a
        documented "known limitation" -- that was flagged in review as
        unacceptable (a passing test must never encode incorrect
        behavior as expected), and is fixed both in the implementation
        and in this assertion.
        """
        error = Exception(REAL_RATE_LIMIT_MESSAGE)  # no status_code attribute at all

        classified = _classify_error(error)

        assert isinstance(classified, RateLimitError), (
            f"Even without a structured status_code, '401' inside '4016' "
            f"must not cause a rate limit to misclassify as an "
            f"authentication failure -- got {type(classified).__name__}."
        )
        assert not isinstance(classified, AuthenticationError)

    def test_word_boundary_still_correctly_detects_a_real_401(self):
        """Proves the token-boundary fix doesn't overcorrect -- a
        genuine, unambiguous 401 in text (no structured status_code)
        must still classify as AuthenticationError."""
        error = Exception("Error code: 401 - invalid api key provided")

        classified = _classify_error(error)

        assert isinstance(classified, AuthenticationError)


class TestJsonValidateFailedFallback:
    """The core regression: json_validate_failed (always a 400) must
    classify as MalformedOutputError, not ValidationError, so
    call_llm_json's strict-then-lenient fallback actually triggers --
    UNLESS the error indicates the schema/request itself was rejected
    (see TestSchemaRequestRejectionNeverDowngrades below), which must
    never be downgraded this way."""

    def test_structured_400_with_json_validate_failed_is_malformed_output(self):
        error = _FakeAPIStatusError(REAL_JSON_VALIDATE_FAILED_MESSAGE, status_code=400)

        classified = _classify_error(error)

        assert isinstance(classified, MalformedOutputError), (
            f"json_validate_failed must classify as MalformedOutputError "
            f"(so call_llm_json's lenient fallback triggers), got "
            f"{type(classified).__name__} instead -- the generic '400' "
            f"branch is shadowing the more specific json_validate_failed "
            f"check."
        )

    def test_structured_400_without_json_validate_failed_is_still_plain_validation_error(self):
        """A genuinely generic 400 (not a JSON-validation failure) must
        still classify as ValidationError -- proves the fix is specific
        to json_validate_failed, not a blanket reclassification of every
        400."""
        error = _FakeAPIStatusError("Error code: 400 - some other bad request", status_code=400)

        classified = _classify_error(error)

        assert isinstance(classified, ValidationError)
        assert not isinstance(classified, MalformedOutputError)

    def test_text_only_fallback_also_gets_this_right(self):
        error = Exception(REAL_JSON_VALIDATE_FAILED_MESSAGE)  # no status_code attribute

        classified = _classify_error(error)

        assert isinstance(classified, MalformedOutputError)


class TestSchemaRequestRejectionNeverDowngrades:
    """
    Revision 2: distinguish "the MODEL's generated output failed to
    validate" (MalformedOutputError -- legitimately retryable with a
    different response format) from "the SCHEMA/REQUEST itself was
    rejected by the provider" (ValidationError -- must never be
    downgraded to a retry, or a real bug in our own llm/schemas.py
    would be silently masked forever instead of ever surfacing).
    """

    @pytest.mark.parametrize("message", [
        "Error code: 400 - Invalid schema for response_format: schema must be a JSON object with 'type': 'object'",
        "Error code: 400 - invalid 'response_format': the provided json_schema is malformed",
        "Error code: 400 - request rejected: does not match JSON Schema specification",
    ])
    def test_schema_rejection_is_validation_error_not_malformed_output(self, message):
        error = _FakeAPIStatusError(message, status_code=400)

        classified = _classify_error(error)

        assert isinstance(classified, ValidationError), (
            f"A rejected schema/request must classify as ValidationError "
            f"(never eligible for model rotation or response-format "
            f"downgrade), got {type(classified).__name__} for: {message!r}"
        )
        assert not isinstance(classified, MalformedOutputError)

    def test_schema_rejection_takes_precedence_even_if_message_also_mentions_json(self):
        """A schema-rejection message could plausibly also contain the
        word 'json' somewhere -- the schema-rejection signal must win,
        not fall through to the json_validate_failed branch."""
        error = _FakeAPIStatusError(
            "Error code: 400 - invalid json_schema: schema must be a valid JSON Schema object",
            status_code=400,
        )

        classified = _classify_error(error)

        assert isinstance(classified, ValidationError)
        assert not isinstance(classified, MalformedOutputError)

    def test_text_only_fallback_also_never_downgrades_schema_rejection(self):
        error = Exception(
            "Bad request: invalid response_format -- schema must be an object with additionalProperties"
        )

        classified = _classify_error(error)

        assert isinstance(classified, ValidationError)
        assert not isinstance(classified, MalformedOutputError)


class TestModelUnavailable:
    """A 404 / model-not-found condition is its own classification,
    distinct from a 5xx provider outage or a 429 rate limit -- retrying
    the same model can never succeed here, and the router (see
    tests/test_provider_routing.py) treats primary vs. fallback
    differently for this specific error."""

    def test_structured_404_is_model_unavailable(self):
        error = _FakeAPIStatusError("Error code: 404 - model not found", status_code=404)

        classified = _classify_error(error)

        assert isinstance(classified, ModelUnavailableError)

    @pytest.mark.parametrize("message", [
        "The model `openai/gpt-oss-999b` does not exist",
        "model_not_found: no such model",
    ])
    def test_text_only_model_not_found_patterns(self, message):
        error = Exception(message)

        classified = _classify_error(error)

        assert isinstance(classified, ModelUnavailableError)

    def test_generic_5xx_is_not_model_unavailable(self):
        """Sanity: a plain provider outage must not be misclassified as
        model-unavailable just because both are 'the model didn't
        answer'."""
        error = _FakeAPIStatusError("Error code: 503 - service unavailable", status_code=503)

        classified = _classify_error(error)

        assert isinstance(classified, ProviderUnavailableError)
        assert not isinstance(classified, ModelUnavailableError)


class TestRetryAfterHonored:
    """Groq's rate-limit response includes a concrete wait-time signal
    -- honoring it (bounded) avoids retrying straight back into the
    same still-active rate limit window, which is what the fixed
    ~1-12s exponential backoff was observed to do in practice against
    real 13-25s required waits. The HTTP Retry-After header (a
    structured, standard signal) is preferred over parsing Groq's
    human-readable body text."""

    def test_header_is_preferred_over_body_text(self):
        """The header says 10s, the body text says a different number
        (24.35s) -- the header must win."""
        error = _FakeAPIStatusError(
            REAL_RATE_LIMIT_MESSAGE, status_code=429,
            headers={"retry-after": "10"},
        )

        seconds = _extract_retry_after_seconds(error, REAL_RATE_LIMIT_MESSAGE.lower())

        assert 9.5 < seconds < 11.0, (
            f"Expected the Retry-After header value (10s + margin) to win "
            f"over the body text's 24.3525s hint, got {seconds}"
        )

    def test_body_text_used_when_no_header_present(self):
        error = _FakeAPIStatusError(REAL_RATE_LIMIT_MESSAGE, status_code=429)  # no headers at all

        seconds = _extract_retry_after_seconds(error, REAL_RATE_LIMIT_MESSAGE.lower())

        assert seconds is not None
        assert 24.0 < seconds < 26.0  # 24.3525 + the small safety margin

    def test_falls_back_to_none_when_neither_header_nor_body_has_a_hint(self):
        error = _FakeAPIStatusError("Error code: 429 - rate_limit_exceeded", status_code=429)

        seconds = _extract_retry_after_seconds(error, "error code: 429 - rate_limit_exceeded")

        assert seconds is None

    def test_parses_the_real_observed_body_hint(self):
        seconds = _parse_retry_after_from_body(REAL_RATE_LIMIT_MESSAGE.lower())
        assert seconds is not None
        assert 24.0 < seconds < 26.0  # 24.3525 + the small safety margin

    def test_bounded_against_a_malformed_or_extreme_hint(self):
        assert _parse_retry_after_from_body("please try again in 999999s") <= 30.5
        assert _parse_retry_after_from_body("no hint here at all") is None

    def test_rate_limit_error_carries_the_parsed_hint_via_header(self):
        error = _FakeAPIStatusError(
            REAL_RATE_LIMIT_MESSAGE, status_code=429,
            headers={"retry-after": "10"},
        )
        classified = _classify_error(error)
        assert isinstance(classified, RateLimitError)
        assert classified.retry_after_seconds is not None
        assert 9.5 < classified.retry_after_seconds < 11.0

    def test_wait_strategy_honors_the_hint_over_fixed_backoff(self):
        class _FakeOutcome:
            def __init__(self, exc):
                self._exc = exc
            def exception(self):
                return self._exc

        class _FakeRetryState:
            def __init__(self, exc):
                self.outcome = _FakeOutcome(exc)

        rate_limit_error = RateLimitError("rate limited", retry_after_seconds=24.85)
        wait_time = _wait_strategy(_FakeRetryState(rate_limit_error))
        assert wait_time == 24.85

    def test_wait_strategy_falls_back_to_exponential_when_no_hint(self):
        class _FakeOutcome:
            def exception(self):
                return RateLimitError("rate limited", retry_after_seconds=None)

        class _FakeRetryState:
            outcome = _FakeOutcome()
            attempt_number = 1

        wait_time = _wait_strategy(_FakeRetryState())
        assert 0 < wait_time <= 12  # exponential(min=1,max=10) + jitter(0-2)
