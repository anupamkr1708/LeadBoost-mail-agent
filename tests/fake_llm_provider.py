"""
Fake LLM provider for deterministic testing.

This allows semantic regression tests to run without calling the live Groq API,
preventing rate limits and ensuring consistent test results.

Usage in tests:

    from tests.fake_llm_provider import FakeLLMProvider
    import mailer_agent.llm.provider_v2 as provider_module
    
    @pytest.fixture
    def fake_llm(monkeypatch):
        fake = FakeLLMProvider({
            "meeting": {...},  # Predetermined JSON response
            "pricing": {...},
            "not interested": {...},
        })
        monkeypatch.setattr(provider_module, "call_llm_json", fake.call_llm_json)
        monkeypatch.setattr(provider_module, "call_llm_text", fake.call_llm_text)
        return fake
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class FakeLLMProvider:
    """
    Deterministic LLM provider for testing.
    
    Returns predetermined responses based on pattern matching in prompts.
    """
    
    def __init__(self, responses: dict[str, dict | str] = None):
        """
        Initialize with predetermined responses.
        
        Args:
            responses: Dict mapping prompt keywords to responses
                      e.g., {"meeting": {"intent": "meeting_request", ...}}
        """
        self.responses = responses or self._default_responses()
        self.call_count = 0
        self.last_prompts = []
    
    def call_llm_text(
        self,
        system_prompt: str,
        human_prompt: str,
        *,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Return predetermined text response.
        """
        self.call_count += 1
        self.last_prompts.append((system_prompt, human_prompt))
        
        # Match prompt to response
        prompt_lower = (system_prompt + " " + human_prompt).lower()
        
        for keyword, response in self.responses.items():
            if keyword.lower() in prompt_lower:
                if isinstance(response, dict):
                    # If response is dict, convert to string
                    return json.dumps(response)
                return response
        
        # Default response
        return self._default_text_response(human_prompt)
    
    def call_llm_json(
        self,
        system_prompt: str,
        human_prompt: str,
        *,
        temperature: float = None,
        max_tokens: int = None
    ) -> dict[str, Any]:
        """
        Return predetermined JSON response.
        """
        self.call_count += 1
        self.last_prompts.append((system_prompt, human_prompt))
        
        # Match prompt to response
        prompt_lower = (system_prompt + " " + human_prompt).lower()
        
        # Sort responses by number of keywords (more specific matches first)
        sorted_responses = sorted(
            self.responses.items(),
            key=lambda x: len(x[0].split()),
            reverse=True
        )
        
        for keyword, response in sorted_responses:
            # For multi-word keys, ALL words must be present
            keywords = keyword.lower().split()
            if all(kw in prompt_lower for kw in keywords):
                if isinstance(response, str):
                    # If response is string, try to parse as JSON
                    try:
                        return json.loads(response)
                    except json.JSONDecodeError:
                        return {"text": response}
                return response
        
        # Default JSON response
        return self._default_json_response(human_prompt)
    
    def _default_text_response(self, prompt: str) -> str:
        """Generate default text response."""
        return "Thank you for your interest. I'll follow up with more details."
    
    def _default_json_response(self, prompt: str) -> dict:
        """Generate default JSON response."""
        return {
            "intents": ["neutral"],
            "sentiment": "neutral",
            "buying_stage": "awareness",
            "urgency": "none",
            "confidence": 0.5,
            "reasoning": "Default fake LLM response"
        }
    
    def _default_responses(self) -> dict[str, dict]:
        """
        Default predetermined responses for common scenarios.
        """
        return {
            # Combined meeting + pricing (check this FIRST before individual keywords)
            "meeting pricing cost": {
                "intents": ["meeting_request", "pricing_request", "positive_interest"],
                "sentiment": "positive",
                "buying_stage": "evaluating",
                "urgency": "near_term",  # "next week" indicates near-term urgency
                "requested_timing": "next week",
                "has_pricing_question": True,
                "has_budget_signal": True,  # Asking about team size shows budget consideration
                "questions_asked": ["When can we meet?", "What does this cost?"],
                "objections_raised": [],
                "requires_human_review": True,  # Pricing always requires review
                "confidence": 0.85,
                "reasoning": "Prospect requested meeting and asked about pricing"
            },
            
            # Meeting requests
            "meeting": {
                "intents": ["meeting_request", "positive_interest"],
                "sentiment": "positive",
                "buying_stage": "evaluating",
                "urgency": "medium",
                "requested_timing": "next week",
                "has_pricing_question": False,
                "has_budget_signal": False,
                "questions_asked": ["When can we meet?"],
                "objections_raised": [],
                "requires_human_review": False,
                "confidence": 0.9,
                "reasoning": "Prospect explicitly requested a meeting"
            },
            
            # Pricing questions
            "pricing": {
                "intents": ["pricing_request", "positive_interest"],
                "sentiment": "neutral",
                "buying_stage": "evaluating",
                "urgency": "medium",
                "requested_timing": None,
                "has_pricing_question": True,
                "has_budget_signal": False,
                "questions_asked": ["What does this cost?"],
                "objections_raised": [],
                "requires_human_review": True,
                "confidence": 0.85,
                "reasoning": "Prospect asked about pricing"
            },
            
            # Not interested
            "not interested": {
                "intents": ["not_interested"],
                "sentiment": "negative",
                "buying_stage": "awareness",
                "urgency": "none",
                "requested_timing": None,
                "has_pricing_question": False,
                "has_budget_signal": False,
                "questions_asked": [],
                "objections_raised": ["Not a good fit"],
                "requires_human_review": False,
                "confidence": 0.95,
                "reasoning": "Prospect explicitly declined interest"
            },
            
            # Unsubscribe
            "unsubscribe": {
                "intents": ["unsubscribe"],
                "sentiment": "negative",
                "buying_stage": "awareness",
                "urgency": "none",
                "requested_timing": None,
                "has_pricing_question": False,
                "has_budget_signal": False,
                "questions_asked": [],
                "objections_raised": [],
                "requires_human_review": False,
                "confidence": 1.0,
                "reasoning": "Unsubscribe keyword detected"
            },
            
            # Positive interest with timing
            "interested": {
                "intents": ["positive_interest", "timing_constraint"],
                "sentiment": "positive",
                "buying_stage": "interest",
                "urgency": "medium",
                "requested_timing": "next quarter",
                "has_pricing_question": False,
                "has_budget_signal": False,
                "questions_asked": [],
                "objections_raised": [],
                "requires_human_review": False,
                "confidence": 0.8,
                "reasoning": "Prospect showed interest with future timing"
            },
            
            # Out of office
            "out of office": {
                "intents": ["out_of_office"],
                "sentiment": "neutral",
                "buying_stage": "awareness",
                "urgency": "none",
                "requested_timing": None,
                "has_pricing_question": False,
                "has_budget_signal": False,
                "questions_asked": [],
                "objections_raised": [],
                "requires_human_review": False,
                "confidence": 1.0,
                "reasoning": "Automatic out-of-office reply"
            },
            
            # Question
            "question": {
                "intents": ["information_request"],
                "sentiment": "neutral",
                "buying_stage": "interest",
                "urgency": "low",
                "requested_timing": None,
                "has_pricing_question": False,
                "has_budget_signal": False,
                "questions_asked": ["Can you tell me more?"],
                "objections_raised": [],
                "requires_human_review": False,
                "confidence": 0.7,
                "reasoning": "Prospect asked for more information"
            },
            
            # Objection
            "objection": {
                "intents": ["objection"],
                "sentiment": "neutral",
                "buying_stage": "evaluating",
                "urgency": "medium",
                "requested_timing": None,
                "has_pricing_question": False,
                "has_budget_signal": False,
                "questions_asked": [],
                "objections_raised": ["Too expensive", "Already have a solution"],
                "requires_human_review": True,
                "confidence": 0.75,
                "reasoning": "Prospect raised concerns"
            },
        }
    
    def reset(self):
        """Reset call tracking."""
        self.call_count = 0
        self.last_prompts = []


def create_fake_provider_fixture():
    """
    Create a pytest fixture for the fake LLM provider.
    
    Usage in conftest.py:
    
        @pytest.fixture
        def fake_llm(monkeypatch):
            return create_fake_provider_fixture()(monkeypatch)
    """
    def fixture(monkeypatch):
        import mailer_agent.llm.provider_v2 as provider_module
        
        fake = FakeLLMProvider()
        monkeypatch.setattr(provider_module, "call_llm_json", fake.call_llm_json)
        monkeypatch.setattr(provider_module, "call_llm_text", fake.call_llm_text)
        monkeypatch.setattr(provider_module, "is_llm_available", lambda: True)
        
        return fake
    
    return fixture
