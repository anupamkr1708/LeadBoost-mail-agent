"""
Pytest configuration and shared fixtures.
"""

import pytest

from tests.fake_llm_provider import FakeLLMProvider, install_fake_llm_provider


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch, request):
    """
    Fixture that provides a fake LLM provider for deterministic testing.
    
    Automatically applied to ALL tests unless explicitly disabled with 
    @pytest.mark.integration marker.
    
    This prevents tests from calling the live Groq API, avoiding:
    - Rate limits
    - API costs
    - Flaky network-dependent tests
    - Non-deterministic LLM outputs

    The fake never infers a response from prompt content -- it only
    returns whatever was explicitly queued via fake_llm.queue_response(...).
    A test that calls code which reaches the LLM provider without queuing
    a response first will get a clear AssertionError, not a guess.

    Usage:
        def test_something(fake_llm):
            fake_llm.queue_response({...})
            result = classify_prospect_reply(...)
            ...

    Usage (explicitly disable for integration test):
        @pytest.mark.integration
        def test_with_real_llm():
            # This test uses actual Groq API
    """
    # Skip patching for integration tests
    if "integration" in request.keywords:
        yield None
        return

    yield install_fake_llm_provider(monkeypatch)


@pytest.fixture
def use_real_llm(monkeypatch):
    """
    Fixture to explicitly enable real LLM for integration tests.
    
    Usage:
        @pytest.mark.integration
        def test_with_real_llm(use_real_llm):
            # This test will use actual Groq API
    """
    # Don't patch anything - use real implementation
    yield
