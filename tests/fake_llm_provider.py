"""
Fake LLM provider for deterministic testing.

Design
------
This is a deterministic response *fixture* provider, not a second
semantic engine. It does not try to infer what a scenario "means" from
the prompt text -- no keyword lists, no substring matching, nothing that
could itself drift out of sync with the real classifier's prompts. Each
test explicitly queues the exact structured result it wants the LLM
"provider" to return, then exercises the real application code
(classify_prospect_reply, draft_message, etc), and asserts on how the
real code processed that canned response.

That split matters: these tests are regression tests for
mailer_agent's parsing/policy/grounding logic, not for the LLM's
semantic judgment (there is no LLM here to judge anything -- that's the
live-provider smoke suite's job, tests/test_business_context.py,
@pytest.mark.integration).

Usage in a test:

    def test_meeting_request_routes_to_coordinate_meeting(fake_llm, ...):
        fake_llm.queue_response({
            "intents": ["meeting_request"],
            "sentiment": "positive",
            "buying_stage": "evaluating",
            "urgency": "near_term",
            "requested_timing": "next week",
            "has_pricing_question": False,
            "has_budget_signal": False,
            "questions_asked": ["When can we meet?"],
            "objections_raised": [],
            "requires_human_review": False,
            "confidence": 0.9,
            "reasoning": "Prospect explicitly requested a meeting",
        })
        result = classify_prospect_reply(...)
        assert IntentType.MEETING_REQUEST in result.intents
        ...

If a test exercises a code path that calls the LLM without first
queuing a response, call_llm_json/call_llm_text raise a clear
AssertionError rather than silently guessing -- a missing fixture is a
test bug, and failing loudly is much cheaper to debug than a test that
"passes" against an unintended default.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class FakeLLMProvider:
    """
    Deterministic response-fixture provider for tests.

    Responses are consumed in FIFO order from an explicit queue. There is
    no prompt inspection and no pattern matching: what you queue is
    exactly what comes back, in the order you queued it.
    """

    def __init__(self) -> None:
        self._queue: list[dict | str | BaseException] = []
        self._default: dict | str | None = None
        self.call_count = 0
        self.last_prompts: list[tuple[str, str]] = []

    # -- fixture setup ----------------------------------------------------

    def queue_response(self, response: dict | str) -> None:
        """Queue one predetermined response for the next LLM call."""
        self._queue.append(response)

    def queue_responses(self, responses: list[dict | str]) -> None:
        """Queue several predetermined responses, consumed in order."""
        self._queue.extend(responses)

    def queue_error(self, error: BaseException) -> None:
        """
        Queue an exception instance to be raised (not returned) on the
        next call -- for testing provider-failure handling (rate limits,
        timeouts, malformed output) without needing a real provider
        outage. e.g. fake_llm.queue_error(RateLimitError("429")).
        """
        self._queue.append(error)

    def set_default_response(self, response: dict | str | None) -> None:
        """
        Optional: set a response returned whenever the queue is empty,
        for tests that need many identical calls (e.g. a loop over many
        contacts) and don't want to queue one entry per call. Most tests
        should prefer queue_response for an explicit, scenario-scoped
        fixture instead.
        """
        self._default = response

    def reset(self) -> None:
        self._queue = []
        self._default = None
        self.call_count = 0
        self.last_prompts = []

    # -- provider interface -----------------------------------------------

    def call_llm_json(
        self,
        system_prompt: str,
        human_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        self.call_count += 1
        self.last_prompts.append((system_prompt, human_prompt))
        response = self._next_response()
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, str):
            return json.loads(response)
        return response

    def call_llm_text(
        self,
        system_prompt: str,
        human_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.call_count += 1
        self.last_prompts.append((system_prompt, human_prompt))
        response = self._next_response()
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, dict):
            return json.dumps(response)
        return response

    def _next_response(self) -> dict | str | BaseException:
        if self._queue:
            return self._queue.pop(0)
        if self._default is not None:
            return self._default
        raise AssertionError(
            "FakeLLMProvider was called with no fixture response queued. "
            "Call fake_llm.queue_response({...}) with the exact structured "
            "result this scenario needs before exercising code that calls "
            "the LLM provider. (This fake never infers a response from "
            "prompt content -- see tests/fake_llm_provider.py.)"
        )


def install_fake_llm_provider(monkeypatch) -> FakeLLMProvider:
    """
    Patch mailer_agent.llm.provider_v2 to use a fresh FakeLLMProvider
    instance and report the LLM as available (so callers take the LLM
    code path rather than their deterministic/keyword fallback path --
    tests that want to exercise the fallback path instead should force
    is_llm_available() to return False, e.g. via GROQ_API_KEY="", the
    way tests/test_grounding.py does).
    """
    import mailer_agent.llm.provider_v2 as provider_module

    fake = FakeLLMProvider()
    monkeypatch.setattr(provider_module, "call_llm_json", fake.call_llm_json)
    monkeypatch.setattr(provider_module, "call_llm_text", fake.call_llm_text)
    monkeypatch.setattr(provider_module, "is_llm_available", lambda: True)
    return fake
