"""
Deterministic semantic evaluation suite (spec §28-30).

What this file proves, and what it deliberately cannot
---------------------------------------------------------
Every scenario here uses tests/fake_llm_provider.py -- a deterministic
fixture provider that returns exactly what's queued and infers nothing
from prompt content. That means this suite CANNOT demonstrate that a
real LLM actually understands paraphrased or ambiguous prospect
language; there is no real language understanding happening here at
all. What it DOES prove, honestly: given a particular structured
interpretation (the kind a real LLM would be expected to produce for a
given scenario), does the rest of the pipeline -- state inference,
planning, guardrail authorization, drafting -- handle it correctly and
consistently? That is a real, valuable, and previously mostly-untested
property independent of whether the LLM itself is good.

The genuine "does the model actually understand this" claim belongs to
tests/evaluation/test_live_llm_semantic_quality.py, a separate,
explicitly @pytest.mark.integration suite that calls the real provider.
It is NOT part of ordinary CI and has NOT been executed in this
sandbox (no network access) -- see docs/FINAL_PRODUCTION_READINESS.md.
This file's job is everything downstream of interpretation.

Dimensions evaluated (each as its own test, not folded into one giant
assertion, so a regression in one dimension doesn't hide behind an
unrelated one passing):
  1. intent understanding (classifier parsing -- see also
     test_semantic_regression.py, which covers this in more depth)
  2. prospect-goal understanding (user_goal reaches the planner)
  3. objection understanding (routes to address_objection / requires review)
  4. fact extraction (current_solution/new_facts survive to planner context)
  5. timing interpretation (structured TimingSignal, not string parsing)
  6. state consistency (state machine event inference from intent)
  7. uncertainty handling (low confidence blocks auto-send)
  8. action quality / policy compliance (guardrails produce the right
     auto-send decision for a given proposal)
  9. grounding (unsupported claims block send)
  10. semantic generalization (multiple DIFFERENT structured
      interpretations that a real LLM might produce for differently-
      worded-but-equivalent emails all lead to the SAME correct
      downstream handling -- this is the honest, fake-LLM-compatible
      version of "converges semantically": it tests the pipeline's
      consistency across equivalent inputs, not the LLM's ability to
      recognize the inputs as equivalent in the first place)
"""

from __future__ import annotations

import pytest

from mailer_agent.policy.guardrails import authorize_action
from mailer_agent.policy.next_action import ActionType, plan_next_action
from mailer_agent.semantic.classifier import classify_prospect_reply
from mailer_agent.state_machine import infer_event_from_semantic_intent, StateTransitionEvent
from tests.test_semantic_regression import test_campaign, test_contact  # noqa: F401 -- reuse fixtures


def _classify(campaign, contact, body):
    return classify_prospect_reply(
        campaign=campaign, contact=contact, inbound_body=body,
        conversation_context="(prior conversation)",
    )


# ---------------------------------------------------------------------------
# 1-2. Intent + prospect-goal understanding
# ---------------------------------------------------------------------------

def test_dimension_prospect_goal_reaches_planner(fake_llm, test_campaign, test_contact):
    fake_llm.queue_response({
        "intents": ["objection"],
        "sentiment": "mixed",
        "buying_stage": "evaluating",
        "urgency": "no_timeline",
        "user_goal": "determine whether migration risk outweighs the benefit",
        "objections_raised": ["switching systems would be painful"],
        "confidence": 0.85,
        "reasoning": "test",
        "requires_human_review": False,
    })
    result = _classify(test_campaign, test_contact, "Switching would be a real pain for us.")
    assert result.semantic_intent.user_goal == "determine whether migration risk outweighs the benefit"

    fake_llm.queue_response({
        "action_type": "address_objection",
        "objective": "Directly address the migration-risk concern with a concrete, honest answer.",
        "reason": "Their stated goal is specifically about migration risk, not general interest.",
        "confidence": 0.8,
        "requires_human_review": False,
    })
    proposal = plan_next_action(intent=result.semantic_intent, context_transcript="(history)")
    assert proposal.action_type == ActionType.ADDRESS_OBJECTION
    assert "migration" in proposal.objective.lower()


# ---------------------------------------------------------------------------
# 3. Objection understanding -> never auto-sent
# ---------------------------------------------------------------------------

def test_dimension_objection_always_requires_review():
    from mailer_agent.policy.next_action import NextActionProposal
    proposal = NextActionProposal(
        action_type=ActionType.ADDRESS_OBJECTION,
        objective="test", reason="test", confidence=1.0, requires_human_review=False,
    )
    from mailer_agent.semantic_models import SemanticIntent
    authorized = authorize_action(
        proposal, intent=SemanticIntent(confidence=1.0, requires_human_review=False), auto_reply_enabled=True,
    )
    assert authorized.can_auto_send is False


# ---------------------------------------------------------------------------
# 4. Fact extraction survives into planner context
# ---------------------------------------------------------------------------

def test_dimension_fact_extraction_with_provenance(fake_llm, test_campaign, test_contact):
    fake_llm.queue_response({
        "intents": ["positive_interest"],
        "sentiment": "positive",
        "buying_stage": "considering",
        "urgency": "no_timeline",
        "current_solution": {"value": "Salesforce", "certainty": "explicit", "evidence": "we use Salesforce today"},
        "confidence": 0.9,
        "reasoning": "test",
        "requires_human_review": False,
    })
    result = _classify(test_campaign, test_contact, "We use Salesforce today but are open to options.")
    assert result.semantic_intent.current_solution.value == "Salesforce"
    assert result.semantic_intent.current_solution.certainty.value == "explicit"
    # Explicit vs inferred distinction is preserved as data, not collapsed
    # into a plain string the rest of the system might treat as equally
    # certain regardless of source.
    assert result.semantic_intent.current_solution.explicit is True


def test_dimension_inferred_fact_is_not_marked_explicit(fake_llm, test_campaign, test_contact):
    fake_llm.queue_response({
        "intents": ["objection"],
        "sentiment": "neutral",
        "buying_stage": "evaluating",
        "urgency": "no_timeline",
        "new_facts": [{
            "value": "migration effort is likely their primary blocker",
            "certainty": "weakly_inferred",
            "evidence": "switching systems would be painful",
        }],
        "confidence": 0.7,
        "reasoning": "test",
        "requires_human_review": False,
    })
    result = _classify(test_campaign, test_contact, "Switching systems would be a real pain.")
    fact = result.semantic_intent.new_facts[0]
    assert fact.certainty.value == "weakly_inferred"
    assert fact.explicit is False, "An inferred conclusion must never be marked explicit=True"


# ---------------------------------------------------------------------------
# 5. Timing interpretation: structured, not regex-parsed; unknown stays unknown
# ---------------------------------------------------------------------------

def test_dimension_vague_timing_does_not_get_a_fabricated_date(fake_llm, test_campaign, test_contact):
    """
    Regression test for the removed 30-day hardcoded fallback (spec §15):
    "maybe sometime next quarter" must not become a specific scheduled
    date invented by deterministic Python.
    """
    fake_llm.queue_response({
        "intents": ["timing_constraint"],
        "sentiment": "neutral",
        "buying_stage": "nurture",
        "urgency": "long_term",
        "timing": {
            "expression": "maybe sometime next quarter, depending",
            "kind": "vague",
            "normalized_target": None,
            "certainty": "weakly_inferred",
            "commitment_strength": "weak",
            "requires_clarification": True,
        },
        "confidence": 0.6,
        "reasoning": "test",
        "requires_human_review": False,
    })
    result = _classify(test_campaign, test_contact, "Maybe sometime next quarter, depending on how things go.")

    from mailer_agent.followup.conversation_aware import FollowUpScheduler
    scheduler = FollowUpScheduler()  # no __init__ args needed; _extract_requested_timing is a pure function
    resolved = scheduler._extract_requested_timing(result.semantic_intent)
    assert resolved is None, (
        "Vague timing with requires_clarification=True must not resolve to "
        "any specific date -- the old behavior invented a 30-day fallback here."
    )


def test_dimension_specific_timing_resolves_to_normalized_target(fake_llm, test_campaign, test_contact):
    fake_llm.queue_response({
        "intents": ["timing_constraint"],
        "sentiment": "neutral",
        "buying_stage": "considering",
        "urgency": "near_term",
        "timing": {
            "expression": "March 15th",
            "kind": "specific_date",
            "normalized_target": "2026-03-15T00:00:00+00:00",
            "certainty": "explicit",
            "commitment_strength": "moderate",
            "requires_clarification": False,
        },
        "confidence": 0.9,
        "reasoning": "test",
        "requires_human_review": False,
    })
    result = _classify(test_campaign, test_contact, "Let's revisit this on March 15th.")

    from mailer_agent.followup.conversation_aware import FollowUpScheduler
    scheduler = FollowUpScheduler()
    resolved = scheduler._extract_requested_timing(result.semantic_intent)
    assert resolved is not None
    assert resolved.year == 2026 and resolved.month == 3 and resolved.day == 15


# ---------------------------------------------------------------------------
# 6. State consistency: event inference has no compound semantic assumptions
# ---------------------------------------------------------------------------

def test_dimension_state_machine_maps_directly_not_by_inference():
    """
    Regression test for the removed "positive_interest + evaluating ->
    pricing_discussed" inference (spec §14): the state machine must map
    directly from what was communicated, never combine two signals to
    invent a third meaning.
    """
    from mailer_agent.semantic_models import BuyingStage, IntentType, SemanticIntent

    # Positive interest + evaluating stage, but NO pricing-related intent
    # at all -- the old code would have inferred PRICING_DISCUSSED here.
    intent = SemanticIntent(
        intents=[IntentType.POSITIVE_INTEREST],
        buying_stage=BuyingStage.EVALUATING,
        confidence=0.9,
    )
    event = infer_event_from_semantic_intent(intent)
    assert event == StateTransitionEvent.POSITIVE_INTEREST, (
        "buying_stage must not upgrade a plain positive-interest signal "
        "into a pricing-discussed event -- that was semantic inference "
        "that didn't belong in the state machine."
    )


# ---------------------------------------------------------------------------
# 7. Uncertainty handling: low confidence blocks auto-send
# ---------------------------------------------------------------------------

def test_dimension_low_confidence_blocks_auto_send():
    from mailer_agent.policy.next_action import NextActionProposal
    from mailer_agent.semantic_models import SemanticIntent

    proposal = NextActionProposal(
        action_type=ActionType.ANSWER, objective="test", reason="test",
        confidence=0.9, requires_human_review=False,
    )
    low_confidence_intent = SemanticIntent(confidence=0.35, requires_human_review=False)
    authorized = authorize_action(proposal, intent=low_confidence_intent, auto_reply_enabled=True)
    assert authorized.can_auto_send is False


# ---------------------------------------------------------------------------
# 9. Grounding: unsupported claims block send (delegates to test_grounding.py
#    for depth; this is the cross-dimension smoke check)
# ---------------------------------------------------------------------------

def test_dimension_grounding_blocks_unsupported_numeric_claim():
    from mailer_agent.llm.grounding import validate_grounding
    result = validate_grounding(
        "We've helped 500 companies just like yours.",
        proof_points="We've worked with 50 companies across the region.",
        context_notes=None,
        conversation_transcript="",
        value_prop="We help teams move faster.",
    )
    assert not result.is_safe_to_send


# ---------------------------------------------------------------------------
# 10. Semantic generalization (honest, fake-LLM-compatible version --
#     see module docstring for what this can/cannot prove)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phrasing_style", ["direct_statement", "soft_hedge", "negative_framing"])
def test_dimension_equivalent_meanings_converge_to_same_handling(fake_llm, test_campaign, test_contact, phrasing_style):
    """
    Three DIFFERENT structured interpretations -- representing what a
    real LLM might plausibly produce for three differently-worded but
    semantically equivalent prospect emails ("we already use Salesforce",
    "we've got something in place already", "we're not looking to
    replace our current stack right now") -- must all lead to the SAME
    downstream planner-eligible handling: current_solution recognized,
    objection-shaped response, human review still required for the
    objection itself. This does not test whether a real LLM would
    actually produce these three interpretations for these three
    inputs (that's the live-LLM suite's job) -- it tests that the
    pipeline treats equivalent structured meaning equivalently
    regardless of which surface wording it came from.
    """
    fixtures = {
        "direct_statement": {
            "intents": ["objection"], "sentiment": "neutral", "buying_stage": "nurture",
            "urgency": "no_timeline",
            "current_solution": {"value": "Salesforce", "certainty": "explicit", "evidence": "we already use Salesforce"},
            "objections_raised": ["already using a CRM"],
            "confidence": 0.85, "reasoning": "test", "requires_human_review": False,
        },
        "soft_hedge": {
            "intents": ["objection"], "sentiment": "neutral", "buying_stage": "nurture",
            "urgency": "no_timeline",
            "current_solution": {"value": "an existing CRM (unspecified)", "certainty": "strongly_inferred", "evidence": "we've got something in place already"},
            "objections_raised": ["already have a solution in place"],
            "confidence": 0.75, "reasoning": "test", "requires_human_review": False,
        },
        "negative_framing": {
            "intents": ["objection"], "sentiment": "neutral", "buying_stage": "nurture",
            "urgency": "no_timeline",
            "current_solution": {"value": "current stack (unspecified)", "certainty": "strongly_inferred", "evidence": "not looking to replace our current stack right now"},
            "objections_raised": ["not looking to replace current stack right now"],
            "confidence": 0.75, "reasoning": "test", "requires_human_review": False,
        },
    }
    fake_llm.queue_response(fixtures[phrasing_style])
    result = _classify(test_campaign, test_contact, "(phrasing varies per scenario)")

    assert result.semantic_intent.current_solution is not None
    from mailer_agent.semantic_models import IntentType
    assert IntentType.OBJECTION in result.semantic_intent.intents

    event = infer_event_from_semantic_intent(result.semantic_intent)
    assert event == StateTransitionEvent.OBJECTION_RAISED

    fake_llm.queue_response({
        "action_type": "address_objection",
        "objective": "Acknowledge their existing solution without pushing a switch.",
        "reason": "test", "confidence": 0.8, "requires_human_review": False,
    })
    proposal = plan_next_action(intent=result.semantic_intent, context_transcript="(history)")
    authorized = authorize_action(proposal, intent=result.semantic_intent, auto_reply_enabled=True)

    assert proposal.action_type == ActionType.ADDRESS_OBJECTION
    assert authorized.can_auto_send is False, "Objections always require human review regardless of phrasing"
