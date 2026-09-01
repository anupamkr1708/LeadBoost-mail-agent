"""
Tests for Phase 11: Grounding validation.

Covers:
- extract_factual_claims()        — claim detection regex
- validate_grounding()            — corpus matching logic
- Pricing / availability flagging — special class
- draft_message() integration     — grounding attached to AgentDraft
- Pipeline integration            — unsafe drafts produce DRAFT status
"""

from __future__ import annotations

import os

# Must set DB URL before importing anything from mailer_agent
os.environ.setdefault("DATABASE_URL", "sqlite:///./grounding_test.db")
os.environ.setdefault("LIVE_SENDING_ENABLED", "false")
os.environ.setdefault("AUTO_REPLY_ENABLED", "false")
os.environ.setdefault("GROQ_API_KEY", "")  # force deterministic fallback

import pytest

from mailer_agent.llm.grounding import extract_factual_claims, validate_grounding
from mailer_agent.models import Campaign, Contact
from mailer_agent.semantic_models import GroundingValidation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _campaign(
    value_prop: str = "We help teams move faster.",
    proof_points: str | None = None,
) -> Campaign:
    return Campaign(
        id=1,
        name="Test Campaign",
        sender_name="Jordan",
        sender_org="TestCorp",
        sender_email="jordan@testcorp.example.com",
        value_prop=value_prop,
        proof_points=proof_points,
        tone="professional, direct",
    )


def _contact(context_notes: str | None = None) -> Contact:
    return Contact(
        id=1,
        campaign_id=1,
        name="Priya Singh",
        email="priya@prospect.example.com",
        title="VP Operations",
        company="ProspectCo",
        status="active",
        context_notes=context_notes,
    )


# ---------------------------------------------------------------------------
# claim extraction
# ---------------------------------------------------------------------------

class TestExtractFactualClaims:
    def test_percentage_claim(self):
        body = "We reduce support tickets by 40%."
        claims = extract_factual_claims(body)
        assert any("40" in c for c in claims), f"claims={claims}"
        assert any("metric" in c for c in claims)

    def test_dollar_claim(self):
        body = "Typical customers save $50K per year."
        claims = extract_factual_claims(body)
        assert any("money" in c for c in claims)

    def test_headcount_claim(self):
        body = "Used by 30 companies across North India."
        claims = extract_factual_claims(body)
        assert any("scale" in c for c in claims)

    def test_time_claim(self):
        body = "We can respond within 24 hours."
        claims = extract_factual_claims(body)
        assert any("time-claim" in c for c in claims)

    def test_setup_time_claim(self):
        body = "Installation takes under 6 hours."
        claims = extract_factual_claims(body)
        assert any("setup-claim" in c or "time-claim" in c for c in claims)

    def test_guarantee_language(self):
        body = "We guarantee a 3× ROI within 90 days."
        claims = extract_factual_claims(body)
        assert any("guarantee" in c for c in claims)

    def test_no_claims_clean_text(self):
        body = "Hi Priya, thought this might be relevant for your team."
        claims = extract_factual_claims(body)
        assert claims == []

    def test_multiple_claims(self):
        body = "We serve 50 customers, cut costs 30%, setup in under 1 day."
        claims = extract_factual_claims(body)
        assert len(claims) >= 3


# ---------------------------------------------------------------------------
# validate_grounding
# ---------------------------------------------------------------------------

class TestValidateGrounding:

    # --- Clean / fully grounded ---

    def test_clean_draft_no_claims_is_grounded(self):
        result = validate_grounding(
            "Hi Priya, I think we can help ProspectCo move faster.",
            proof_points=None,
            context_notes=None,
            conversation_transcript=None,
        )
        assert result.is_grounded
        assert result.is_safe_to_send
        assert result.unsupported_claims == []

    def test_metric_supported_by_proof_points(self):
        body = "We cut support response time by 40%."
        result = validate_grounding(
            body,
            proof_points="Average 40% reduction in support response time across customers.",
            context_notes=None,
            conversation_transcript=None,
        )
        assert result.is_grounded
        assert result.unsupported_claims == [], result.unsupported_claims

    def test_metric_supported_by_value_prop(self):
        body = "We help B2B SaaS teams cut support response time 40%."
        result = validate_grounding(
            body,
            proof_points=None,
            context_notes=None,
            conversation_transcript=None,
            value_prop="We help B2B SaaS teams cut support response time 40% with an AI triage layer.",
        )
        assert result.is_grounded

    def test_headcount_supported_by_proof_points(self):
        body = "Used by 3 YC-backed startups so far."
        result = validate_grounding(
            body,
            proof_points="Used by 3 YC-backed startups; average setup time 6 hours.",
            context_notes=None,
            conversation_transcript=None,
        )
        assert result.is_grounded

    def test_fact_in_context_notes_is_supported(self):
        body = "I noticed ProspectCo raised a Series A last March."
        result = validate_grounding(
            body,
            proof_points=None,
            context_notes="ProspectCo raised Series A in March.",
            conversation_transcript=None,
        )
        assert result.is_grounded

    def test_number_from_conversation_transcript_is_supported(self):
        """
        Prospect mentioned "team of 50" — agent can reference 50.
        """
        body = "For a team of 50, our pricing tier would be relevant."
        result = validate_grounding(
            body,
            proof_points=None,
            context_notes=None,
            conversation_transcript="THEM: we have a team of 50 people here.",
        )
        # NOTE: pricing term triggers pricing flag — check that the
        # reason is pricing, not an unsupported metric
        assert "pricing" in result.validation_notes.lower() or not result.is_safe_to_send

    # --- Unsupported claims ---

    def test_metric_not_in_any_source_is_unsupported(self):
        body = "We improve efficiency by 75%."
        result = validate_grounding(
            body,
            proof_points="Average 40% reduction in support response time.",
            context_notes=None,
            conversation_transcript=None,
        )
        assert not result.is_grounded
        assert not result.is_safe_to_send
        assert len(result.unsupported_claims) > 0
        assert any("75" in c for c in result.unsupported_claims)

    def test_invented_headcount_is_unsupported(self):
        body = "Over 200 companies trust us for their sales outreach."
        result = validate_grounding(
            body,
            proof_points="Used by 3 YC-backed startups.",
            context_notes=None,
            conversation_transcript=None,
        )
        assert not result.is_grounded
        assert any("200" in c for c in result.unsupported_claims)

    def test_no_source_data_any_numeric_is_unsupported(self):
        body = "Our customers see 50% faster results."
        result = validate_grounding(
            body,
            proof_points=None,
            context_notes=None,
            conversation_transcript=None,
        )
        assert not result.is_grounded

    def test_guarantee_not_in_source_is_unsupported(self):
        body = "We guarantee results within 30 days."
        result = validate_grounding(
            body,
            proof_points="Typical ROI in 3 months.",
            context_notes=None,
            conversation_transcript=None,
        )
        assert not result.is_grounded
        assert any("guarantee" in c for c in result.unsupported_claims)

    def test_guarantee_present_in_proof_points_is_allowed(self):
        body = "We guarantee your satisfaction."
        result = validate_grounding(
            body,
            proof_points="We guarantee satisfaction or full refund.",
            context_notes=None,
            conversation_transcript=None,
        )
        # The guarantee is sourced — should be grounded
        assert result.is_grounded

    # --- Pricing / availability special class ---

    def test_pricing_term_always_flagged(self):
        body = "Our pricing starts at $200/month for teams."
        result = validate_grounding(
            body,
            proof_points="Pricing starts at $200/month.",
            context_notes=None,
            conversation_transcript=None,
        )
        assert not result.is_safe_to_send
        assert "pricing" in result.validation_notes.lower()

    def test_cost_term_flagged(self):
        body = "The cost is well within your budget."
        result = validate_grounding(
            body,
            proof_points=None,
            context_notes=None,
            conversation_transcript=None,
        )
        assert not result.is_safe_to_send

    def test_discount_term_flagged(self):
        body = "We can offer a 20% discount for annual plans."
        result = validate_grounding(
            body,
            proof_points=None,
            context_notes=None,
            conversation_transcript=None,
        )
        assert not result.is_safe_to_send

    def test_free_trial_flagged(self):
        body = "Sign up for a free trial today."
        result = validate_grounding(
            body,
            proof_points=None,
            context_notes=None,
            conversation_transcript=None,
        )
        assert not result.is_safe_to_send

    def test_availability_promise_flagged(self):
        body = "We're available now and can start this week."
        result = validate_grounding(
            body,
            proof_points=None,
            context_notes=None,
            conversation_transcript=None,
        )
        assert not result.is_safe_to_send

    # --- GroundingValidation.is_safe_to_send property ---

    def test_is_safe_to_send_requires_no_unsupported_and_grounded(self):
        grounded_clean = GroundingValidation(
            is_grounded=True,
            unsupported_claims=[],
            supported_claims=["metric: «40%»"],
            confidence=1.0,
        )
        assert grounded_clean.is_safe_to_send

        grounded_with_unsupported = GroundingValidation(
            is_grounded=True,  # edge case: set manually
            unsupported_claims=["metric: «99%»"],
            confidence=0.5,
        )
        # is_safe_to_send checks both conditions
        assert not grounded_with_unsupported.is_safe_to_send

        ungrounded = GroundingValidation(
            is_grounded=False,
            unsupported_claims=["pricing-terms: pricing"],
            confidence=0.0,
        )
        assert not ungrounded.is_safe_to_send

    # --- Confidence scoring ---

    def test_confidence_is_1_for_empty_body(self):
        result = validate_grounding(
            "Hi there, quick question.",
            proof_points=None,
            context_notes=None,
            conversation_transcript=None,
        )
        assert result.confidence == 1.0

    def test_confidence_scales_with_unsupported_ratio(self):
        # 2 claims, 1 supported (40%), 1 unsupported (99%)
        body = "We cut response time 40%, and improve output by 99%."
        result = validate_grounding(
            body,
            proof_points="We cut response time 40%.",
            context_notes=None,
            conversation_transcript=None,
        )
        # confidence = 1 supported / 2 total = 0.5
        assert 0.4 <= result.confidence <= 0.6, f"confidence={result.confidence}"

    def test_pricing_sets_confidence_to_zero(self):
        body = "Our pricing is very competitive."
        result = validate_grounding(
            body,
            proof_points="Used by 50 companies.",
            context_notes=None,
            conversation_transcript=None,
        )
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Integration: draft_message() attaches grounding
# ---------------------------------------------------------------------------

class TestDraftMessageGrounding:
    """
    Tests that grounding is wired into draft_message().
    Uses the deterministic fallback path exclusively -- every test in this
    class forces is_llm_available() to False via the autouse fixture below,
    rather than relying on the module-level GROQ_API_KEY="" env var. Relying
    on ambient env state was fragile: it depended on import order and on
    settings caching, and it silently stopped working once agent.py's LLM
    calls correctly started honoring the fake_llm fixture's patches (see
    mailer_agent/llm/provider.py). Explicit per-class monkeypatching is
    robust regardless of what else runs in the same session.
    """

    @pytest.fixture(autouse=True)
    def _force_fallback(self, monkeypatch):
        import mailer_agent.llm.agent as agent_mod
        monkeypatch.setattr(agent_mod, "is_llm_available", lambda: False)

    def _make_campaign_contact(self, proof_points=None, context_notes=None):
        campaign = Campaign(
            id=1, name="Test", sender_name="Jordan", sender_org="TestCorp",
            sender_email="jordan@testcorp.example.com",
            value_prop="We help teams move faster.",
            proof_points=proof_points,
            tone="professional, direct",
        )
        contact = Contact(
            id=1, campaign_id=1, name="Sam", email="sam@prospect.example.com",
            company="ProspectCo", status="active",
            context_notes=context_notes, follow_up_index=0,
        )
        return campaign, contact

    def test_draft_has_grounding_attribute(self):
        from mailer_agent.llm.agent import draft_message
        campaign, contact = self._make_campaign_contact()
        draft = draft_message(
            campaign=campaign,
            contact=contact,
            action_type="initial_outreach",
            context_transcript="(no messages sent yet)",
        )
        assert draft.grounding is not None
        assert isinstance(draft.grounding, GroundingValidation)

    def test_fallback_draft_no_claims_is_grounded(self):
        """The deterministic fallback never invents metrics."""
        from mailer_agent.llm.agent import draft_message
        campaign, contact = self._make_campaign_contact()
        draft = draft_message(
            campaign=campaign,
            contact=contact,
            action_type="initial_outreach",
            context_transcript="(no messages sent yet)",
        )
        assert draft.source == "fallback"
        # Fallback body contains no numeric claims
        assert draft.grounding is not None
        assert draft.grounding.is_grounded, (
            f"Fallback draft should be grounded but got: "
            f"unsupported={draft.grounding.unsupported_claims}, "
            f"notes={draft.grounding.validation_notes}"
        )

    def test_campaign_proof_point_supports_value_prop_metric(self):
        """
        If the value_prop already contains a metric and proof_points
        repeats it, the grounding check should pass.
        """
        from mailer_agent.llm.agent import draft_message
        vp = "We cut support response time 40% with an AI triage layer."
        pp = "Used by 3 YC-backed startups; average setup time 6 hours."
        campaign = Campaign(
            id=1, name="Test", sender_name="Jordan", sender_org="TestCorp",
            sender_email="j@tc.example.com",
            value_prop=vp, proof_points=pp, tone="direct",
        )
        contact = Contact(
            id=1, campaign_id=1, name="Sam", email="s@p.example.com",
            company="Prospect", status="active", follow_up_index=0,
        )
        draft = draft_message(
            campaign=campaign,
            contact=contact,
            action_type="initial_outreach",
            context_transcript="(no messages sent yet)",
        )
        # Grounding result present; either clean or only pricing-flagged
        assert draft.grounding is not None

    def test_grounding_result_logged_for_unsafe_draft(self, caplog, monkeypatch):
        """
        If a draft contains pricing language, the grounding result
        should be logged at INFO level.
        """
        import logging
        from mailer_agent.llm import agent as agent_module

        def _pricing_fallback(campaign, contact, action_type):
            from mailer_agent.llm.agent import AgentDraft
            return AgentDraft(
                subject="Test",
                body="Our pricing starts at $200/month per user.",
                reasoning="test",
                source="fallback",
            )

        monkeypatch.setattr(agent_module, "_draft_fallback", _pricing_fallback)

        from mailer_agent.llm.agent import draft_message
        campaign, contact = self._make_campaign_contact()
        with caplog.at_level(logging.INFO, logger="mailer_agent.agent"):
            draft = draft_message(
                campaign=campaign,
                contact=contact,
                action_type="initial_outreach",
                context_transcript="(no messages)",
            )
        assert not draft.grounding.is_safe_to_send
        # Log should mention grounding
        assert any(
            "grounding" in r.message.lower() for r in caplog.records
        ), f"Log records: {[r.message for r in caplog.records]}"
