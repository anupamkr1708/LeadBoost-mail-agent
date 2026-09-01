"""
Semantic regression tests for the classification pipeline's own logic.

What these tests are -- and are not
------------------------------------
These are regression tests for mailer_agent.semantic.classifier's
*plumbing*: prompt -> LLM response -> SemanticIntent -> ClassificationResult.
They are NOT a test of whether a real LLM understands B2B sales email
correctly -- there is no real LLM here, only tests/fake_llm_provider.py, a
deterministic fixture provider that returns exactly what each test queues
for it and infers nothing from prompt content.

Each test explicitly queues the structured JSON a real LLM call would be
expected to return for that scenario, then calls classify_prospect_reply()
for real and asserts the classifier correctly:
  - parses each field into the right enum / list / bool
  - preserves unknown/invalid enum values as safe defaults instead of
    crashing
  - tags the result with the right `source`
  - propagates explicit provider failures as failures, never silently
    as a "neutral" semantic result (see test_provider_failure_is_not_neutral)

Two scenarios below (unsubscribe, out-of-office) never reach the LLM at
all: they're caught by classify_prospect_reply's rule-based safety net
(_check_critical_keywords) before any provider call, so no fixture is
queued for them -- if the classifier ever changes to send these to the
LLM instead, the tests still catch a regression because the fake would
raise AssertionError for a missing fixture.

Live-LLM behavioral quality (does a real model actually understand these
scenarios well) belongs in a separate, explicitly-marked
@pytest.mark.integration suite that calls the real provider -- not here.
"""

from __future__ import annotations

import pytest

from mailer_agent.llm.provider_v2 import MalformedOutputError, RateLimitError
from mailer_agent.models import Campaign, Contact
from mailer_agent.semantic.classifier import classify_prospect_reply
from mailer_agent.semantic_models import BuyingStage, IntentType, SentimentType, UrgencyLevel


@pytest.fixture
def test_campaign():
    return Campaign(
        id=1,
        name="Test Campaign",
        sender_name="Jordan",
        sender_org="TestCorp",
        sender_email="jordan@testcorp.example.com",
        value_prop="We help B2B companies streamline operations",
        proof_points="Used by 50+ companies, average ROI 3x",
        tone="professional, direct",
    )


@pytest.fixture
def test_contact():
    return Contact(
        id=1,
        campaign_id=1,
        name="Priya Singh",
        email="priya@prospect.example.com",
        title="VP Operations",
        company="ProspectCo",
        status="active",
    )


def _classify(campaign, contact, body, context="(no prior messages)"):
    return classify_prospect_reply(
        campaign=campaign,
        contact=contact,
        inbound_body=body,
        conversation_context=context,
    )


# ---------------------------------------------------------------------------
# LLM-path scenarios: fixture queued explicitly, plumbing asserted
# ---------------------------------------------------------------------------

def test_meeting_and_pricing_multi_intent_is_parsed_correctly(fake_llm, test_campaign, test_contact):
    """
    A structured response with several intents, list fields, and
    commercial-signal booleans should come through classify_prospect_reply
    with every field intact and correctly enum-typed.
    """
    fake_llm.queue_response({
        "intents": ["positive_interest", "meeting_request", "pricing_request", "question"],
        "sentiment": "positive",
        "buying_stage": "evaluating",
        "urgency": "near_term",
        "requested_timing": "next week",
        "has_pricing_question": True,
        "has_budget_signal": True,
        "has_decision_maker_signal": False,
        "has_commitment_signal": False,
        "requested_information": ["pricing for 50-person team"],
        "objections_raised": [],
        "questions_asked": ["pricing for team of 50", "availability for call next week"],
        "confidence": 0.9,
        "reasoning": "Clear positive interest with specific meeting request and pricing question.",
        "requires_human_review": False,
        "human_review_reason": None,
    })

    result = _classify(
        test_campaign, test_contact,
        "This looks interesting -- can we do a call next week? "
        "Also, what does pricing look like for a team of 50?",
    )

    assert result.success
    assert result.source == "llm"
    intent = result.semantic_intent

    assert intent.intents == [
        IntentType.POSITIVE_INTEREST,
        IntentType.MEETING_REQUEST,
        IntentType.PRICING_REQUEST,
        IntentType.QUESTION,
    ]
    assert intent.sentiment == SentimentType.POSITIVE
    assert intent.buying_stage == BuyingStage.EVALUATING
    assert intent.urgency == UrgencyLevel.NEAR_TERM
    assert intent.requested_timing == "next week"
    assert intent.has_pricing_question is True
    assert intent.has_budget_signal is True
    assert intent.requested_information == ["pricing for 50-person team"]
    assert intent.questions_asked == ["pricing for team of 50", "availability for call next week"]
    assert intent.confidence == pytest.approx(0.9)
    assert intent.requires_human_review is False


def test_timing_objection_is_parsed_correctly(fake_llm, test_campaign, test_contact):
    """Mixed sentiment + objection + long-horizon timing, all fields preserved."""
    fake_llm.queue_response({
        "intents": ["positive_interest", "objection", "timing_constraint"],
        "sentiment": "mixed",
        "buying_stage": "nurture",
        "urgency": "long_term",
        "requested_timing": "18 months",
        "has_pricing_question": False,
        "has_budget_signal": False,
        "objections_raised": ["locked into current vendor for 18 months"],
        "questions_asked": [],
        "confidence": 0.85,
        "reasoning": "Genuine interest but strong timing objection.",
        "requires_human_review": False,
    })

    result = _classify(
        test_campaign, test_contact,
        "Interesting idea, but we're locked into our current vendor for "
        "the next 18 months. Maybe revisit then?",
    )

    assert result.success
    intent = result.semantic_intent
    assert IntentType.OBJECTION in intent.intents
    assert IntentType.TIMING_CONSTRAINT in intent.intents
    assert intent.sentiment == SentimentType.MIXED
    assert intent.buying_stage == BuyingStage.NURTURE
    assert intent.urgency == UrgencyLevel.LONG_TERM
    assert intent.requested_timing == "18 months"
    assert intent.objections_raised == ["locked into current vendor for 18 months"]


def test_price_objection_is_parsed_correctly(fake_llm, test_campaign, test_contact):
    fake_llm.queue_response({
        "intents": ["positive_interest", "objection"],
        "sentiment": "neutral",
        "buying_stage": "evaluating",
        "urgency": "no_timeline",
        "has_pricing_question": True,
        "objections_raised": ["pricing seems high compared to alternatives"],
        "questions_asked": [],
        "confidence": 0.8,
        "reasoning": "Interest with a price objection.",
        "requires_human_review": True,
        "human_review_reason": "Pricing objection needs a tailored response",
    })

    result = _classify(
        test_campaign, test_contact,
        "We like the idea, but your pricing seems high compared to alternatives.",
    )

    assert result.success
    intent = result.semantic_intent
    assert IntentType.OBJECTION in intent.intents
    assert intent.has_pricing_question is True
    assert intent.objections_raised == ["pricing seems high compared to alternatives"]
    assert intent.requires_human_review is True
    assert intent.human_review_reason == "Pricing objection needs a tailored response"


def test_information_request_is_parsed_correctly(fake_llm, test_campaign, test_contact):
    fake_llm.queue_response({
        "intents": ["information_request"],
        "sentiment": "neutral",
        "buying_stage": "considering",
        "urgency": "no_timeline",
        "requested_information": ["case studies", "product demo video"],
        "questions_asked": [],
        "confidence": 0.75,
        "reasoning": "Requesting specific materials.",
        "requires_human_review": False,
    })

    result = _classify(
        test_campaign, test_contact,
        "Can you send me your case studies and a product demo video?",
    )

    assert result.success
    intent = result.semantic_intent
    assert IntentType.INFORMATION_REQUEST in intent.intents
    assert intent.requested_information == ["case studies", "product demo video"]


def test_referral_is_parsed_correctly(fake_llm, test_campaign, test_contact):
    fake_llm.queue_response({
        "intents": ["referral"],
        "sentiment": "neutral",
        "buying_stage": "unaware",
        "urgency": "no_timeline",
        "questions_asked": [],
        "confidence": 0.7,
        "reasoning": "Pointed to procurement team.",
        "requires_human_review": True,
        "human_review_reason": "Needs re-routing to the referred contact",
    })

    result = _classify(
        test_campaign, test_contact,
        "I'm not the right person for this. Please contact our procurement "
        "team at procurement@company.com.",
    )

    assert result.success
    intent = result.semantic_intent
    assert IntentType.REFERRAL in intent.intents
    assert intent.requires_human_review is True


def test_complex_multi_question_is_parsed_correctly(fake_llm, test_campaign, test_contact):
    fake_llm.queue_response({
        "intents": ["question", "information_request"],
        "sentiment": "neutral",
        "buying_stage": "considering",
        "urgency": "no_timeline",
        "questions_asked": [
            "How does this integrate with Salesforce?",
            "What's the implementation timeline?",
            "Do you offer training for our team?",
            "Can we see a demo with our actual data?",
        ],
        "confidence": 0.8,
        "reasoning": "Multiple concrete evaluation questions.",
        "requires_human_review": False,
    })

    result = _classify(
        test_campaign, test_contact,
        "Thanks for reaching out. A few questions:\n"
        "1. How does this integrate with Salesforce?\n"
        "2. What's the implementation timeline?\n"
        "3. Do you offer training for our team?\n"
        "4. Can we see a demo with our actual data?",
    )

    assert result.success
    intent = result.semantic_intent
    assert len(intent.questions_asked) == 4


def test_budget_and_timeline_is_parsed_correctly(fake_llm, test_campaign, test_contact):
    fake_llm.queue_response({
        "intents": ["positive_interest", "timing_constraint"],
        "sentiment": "positive",
        "buying_stage": "deciding",
        "urgency": "immediate",
        "requested_timing": "end of July (Q3)",
        "has_budget_signal": True,
        "has_decision_maker_signal": True,
        "questions_asked": ["Can you work with that timeline?"],
        "confidence": 0.9,
        "reasoning": "Explicit budget figure and decision deadline.",
        "requires_human_review": False,
    })

    result = _classify(
        test_campaign, test_contact,
        "We have $50K budgeted for this in Q3. Our team wants to make a "
        "decision by end of July. Can you work with that timeline?",
    )

    assert result.success
    intent = result.semantic_intent
    assert intent.has_budget_signal is True
    assert intent.has_decision_maker_signal is True
    assert intent.buying_stage == BuyingStage.DECIDING
    assert intent.urgency == UrgencyLevel.IMMEDIATE


def test_competitor_comparison_is_parsed_correctly(fake_llm, test_campaign, test_contact):
    fake_llm.queue_response({
        "intents": ["question", "information_request"],
        "sentiment": "neutral",
        "buying_stage": "evaluating",
        "urgency": "no_timeline",
        "questions_asked": ["How do you compare to CompetitorX?"],
        "confidence": 0.75,
        "reasoning": "Active competitive evaluation.",
        "requires_human_review": False,
    })

    result = _classify(
        test_campaign, test_contact,
        "How do you compare to CompetitorX? They're offering something similar.",
    )

    assert result.success
    intent = result.semantic_intent
    assert intent.buying_stage == BuyingStage.EVALUATING
    assert len(intent.questions_asked) == 1


def test_shoe_business_multi_item_request_is_parsed_correctly(fake_llm):
    """
    Same classifier, a different (footwear) business context -- proves the
    plumbing has no per-industry branches, only the campaign/contact data
    passed in changes.
    """
    campaign = Campaign(
        id=1,
        name="Footwear Distribution Partnership",
        sender_name="Raj Kumar",
        sender_org="Kumar Footwear Distributors",
        sender_email="raj@kumarfootwear.example.com",
        value_prop="We supply premium footwear brands to established retail chains across North India",
        proof_points="Partnered with 50+ retail chains, 15 years in business",
        tone="professional, business-focused",
    )
    contact = Contact(
        id=1, campaign_id=1, name="Priya Sharma", email="priya@mochishoes.example.com",
        title="Procurement Manager", company="Mochi Shoes", status="active",
    )

    fake_llm.queue_response({
        "intents": ["positive_interest", "information_request"],
        "sentiment": "positive",
        "buying_stage": "considering",
        "urgency": "no_timeline",
        "requested_information": [
            "wholesale catalog", "minimum order quantities", "payment terms", "delivery timelines",
        ],
        "questions_asked": [],
        "confidence": 0.85,
        "reasoning": "Prospect requested catalog and commercial terms.",
        "requires_human_review": False,
    })

    result = _classify(
        campaign, contact,
        "Thanks for reaching out. We're always looking at new supplier "
        "partnerships. Can you send over your wholesale catalog and terms? "
        "We'd specifically need: current product line, minimum order "
        "quantities, payment terms, delivery timelines for Mumbai region.",
    )

    assert result.success
    intent = result.semantic_intent
    assert IntentType.POSITIVE_INTEREST in intent.intents
    assert len(intent.requested_information) == 4


# ---------------------------------------------------------------------------
# Rule-based safety net -- never reaches the LLM/fake, no fixture queued
# ---------------------------------------------------------------------------

def test_unsubscribe_keyword_never_calls_llm(fake_llm, test_campaign, test_contact):
    """
    Unsubscribe must be caught by the rule-based safety net before any
    provider call -- verified by NOT queuing a fixture: if this ever
    silently fell through to the LLM, the fake would raise
    AssertionError for the missing fixture and this test would fail loudly.
    """
    result = _classify(test_campaign, test_contact, "Please unsubscribe me from this list.")

    assert result.success
    assert result.source == "rule_based"
    assert fake_llm.call_count == 0, "Unsubscribe should never reach the LLM provider"
    intent = result.semantic_intent
    assert IntentType.UNSUBSCRIBE in intent.intents
    assert intent.confidence >= 0.9


def test_out_of_office_never_calls_llm(fake_llm, test_campaign, test_contact):
    result = _classify(
        test_campaign, test_contact,
        "Thank you for your email. I am out of the office until Monday, "
        "March 15th with limited access to email.",
    )

    assert result.success
    assert result.source == "rule_based"
    assert fake_llm.call_count == 0, "OOO auto-replies should never reach the LLM provider"
    intent = result.semantic_intent
    assert IntentType.OUT_OF_OFFICE in intent.intents
    assert intent.sentiment == SentimentType.NEUTRAL
    assert not intent.requires_human_review


# ---------------------------------------------------------------------------
# Defensive parsing and explicit failure states
# ---------------------------------------------------------------------------

def test_unknown_intent_value_is_dropped_not_crashed(fake_llm, test_campaign, test_contact):
    """
    If a response contains an intent string outside IntentType's known
    values (a future prompt/schema drift, or a provider glitch), parsing
    must drop it and fall back to a safe default rather than raising.
    """
    fake_llm.queue_response({
        "intents": ["totally_unknown_future_intent"],
        "sentiment": "positive",
        "buying_stage": "aware",
        "urgency": "no_timeline",
        "confidence": 0.5,
        "reasoning": "test",
        "requires_human_review": False,
    })

    result = _classify(test_campaign, test_contact, "Some reply text.")

    assert result.success
    # Unknown intent dropped; classifier falls back to NEUTRAL rather than
    # an empty intents list or a crash.
    assert result.semantic_intent.intents == [IntentType.NEUTRAL]


def test_unknown_enum_values_fall_back_to_safe_defaults(fake_llm, test_campaign, test_contact):
    """
    Same defensive-parsing guarantee for sentiment/buying_stage/urgency:
    an invalid value degrades to a safe default instead of raising.
    """
    fake_llm.queue_response({
        "intents": ["question"],
        "sentiment": "extremely_thrilled",   # not a real SentimentType
        "buying_stage": "somewhere_in_the_funnel",  # not a real BuyingStage
        "urgency": "medium",   # not a real UrgencyLevel (this was the old fake's bug)
        "confidence": 0.5,
        "reasoning": "test",
        "requires_human_review": False,
    })

    result = _classify(test_campaign, test_contact, "Some reply text.")

    assert result.success
    intent = result.semantic_intent
    assert intent.sentiment == SentimentType.NEUTRAL
    assert intent.buying_stage == BuyingStage.UNAWARE
    assert intent.urgency == UrgencyLevel.NO_TIMELINE


def test_provider_failure_is_not_neutral(fake_llm, test_campaign, test_contact):
    """
    Critical invariant: when the provider call itself fails (rate limit,
    timeout, malformed output), classify_prospect_reply must return an
    explicit failure (success=False, failure_reason set) -- never a
    success=True result with intent defaulted to "neutral". Silently
    downgrading a provider outage to "the prospect said nothing
    interesting" would hide real failures from monitoring and from the
    policy layer, which treats "neutral" as a legitimate semantic
    reading rather than "we don't actually know".
    """
    fake_llm.queue_error(RateLimitError("429 from provider"))

    result = _classify(test_campaign, test_contact, "Some reply text.")

    assert result.success is False
    assert result.failure_reason is not None
    assert result.semantic_intent is None


def test_malformed_provider_output_is_not_neutral(fake_llm, test_campaign, test_contact):
    fake_llm.queue_error(MalformedOutputError("could not parse provider response"))

    result = _classify(test_campaign, test_contact, "Some reply text.")

    assert result.success is False
    assert result.failure_reason is not None


def test_llm_unavailable_returns_low_confidence_fallback_requiring_review(test_campaign, test_contact, monkeypatch):
    """
    When the provider is not configured at all (no API key), the
    classifier should not attempt a call -- it should go straight to the
    explicit, low-confidence fallback that always routes to human review.
    """
    import mailer_agent.llm.provider_v2 as provider_module
    monkeypatch.setattr(provider_module, "is_llm_available", lambda: False)

    result = _classify(test_campaign, test_contact, "Some reply text.")

    assert result.success
    assert result.source == "fallback"
    intent = result.semantic_intent
    assert intent.confidence < 0.5
    assert intent.requires_human_review is True
