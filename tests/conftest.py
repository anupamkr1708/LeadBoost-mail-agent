"""
Pytest configuration and shared fixtures.
"""

import pytest
from tests.fake_llm_provider import FakeLLMProvider


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
    
    Usage (automatic):
        def test_something():
            # fake_llm is automatically active
            
    Usage (explicitly disable for integration test):
        @pytest.mark.integration
        def test_with_real_llm():
            # This test uses actual Groq API
    """
    # Skip patching for integration tests
    if "integration" in request.keywords:
        yield None
        return
    
    import mailer_agent.llm.provider_v2 as provider_module
    
    fake = FakeLLMProvider()
    
    # Patch provider_v2 functions
    monkeypatch.setattr(provider_module, "call_llm_json", fake.call_llm_json)
    monkeypatch.setattr(provider_module, "call_llm_text", fake.call_llm_text)
    monkeypatch.setattr(provider_module, "is_llm_available", lambda: True)
    
    yield fake


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
