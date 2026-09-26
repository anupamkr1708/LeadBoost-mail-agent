"""
Regression test for llm/provider.py's compatibility-wrapper exception
mapping.

Real bug: ModelUnavailableError (provider_v2.py's classification for a
404/model-not-found response -- see tests/test_provider_error_classification.py
and tests/test_provider_routing.py for how it's raised) is a subclass of
LLMProviderError, but not of ProviderUnavailableError, AuthenticationError,
or MalformedOutputError. The wrapper's except clauses only explicitly
handled that first pair before mapping to LLMUnavailableError -- so
ModelUnavailableError fell through to the generic
`except LLMProviderError: raise LLMOutputError(...)` branch, mislabeling
a "the configured model doesn't exist" failure as malformed model
output. Fixed by adding ModelUnavailableError to the
ProviderUnavailableError/AuthenticationError tuple in both
call_llm_text and call_llm_json.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mailer_agent.llm.provider import LLMOutputError, LLMUnavailableError, call_llm_json, call_llm_text
from mailer_agent.llm.provider_v2 import ModelUnavailableError


def test_model_unavailable_maps_to_llm_unavailable_not_llm_output_json():
    with patch(
        "mailer_agent.llm.provider_v2.call_llm_json",
        side_effect=ModelUnavailableError("Error code: 404 - model_not_found"),
    ):
        with pytest.raises(LLMUnavailableError) as exc_info:
            call_llm_json("sys", "human")

    assert not isinstance(exc_info.value, LLMOutputError), (
        "A model-not-found failure must map to LLMUnavailableError "
        "(provider/model unavailable), not LLMOutputError (malformed "
        "output) -- these mean different things to callers like "
        "agent.py's draft_message, and to anyone reading the logs."
    )


def test_model_unavailable_maps_to_llm_unavailable_not_llm_output_text():
    """Same mapping, for call_llm_text's wrapper -- both entry points
    share the same except-clause bug and the same fix."""
    with patch(
        "mailer_agent.llm.provider_v2.call_llm_text",
        side_effect=ModelUnavailableError("Error code: 404 - model_not_found"),
    ):
        with pytest.raises(LLMUnavailableError) as exc_info:
            call_llm_text("sys", "human")

    assert not isinstance(exc_info.value, LLMOutputError)
