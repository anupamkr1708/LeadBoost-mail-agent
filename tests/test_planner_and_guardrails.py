"""
Unit tests for policy/next_action.py (the Planner) and
policy/guardrails.py (deterministic authorization of the planner's
proposal).

These are deliberately separate from tests/test_semantic_regression.py:
that file proves the classifier's parsing is correct; this file proves
the planner's parsing is correct and that guardrails make the right
auto-send/review call for a given proposal -- two distinct reasoning
steps, tested independently, matching how they're structured in
production (see mail/reply_handler_v2.py::_draft_and_maybe_send_reply).
"""

from __future__ import annotations

import pytest

from mailer_agent.policy.guardrails import MIN_AUTO_SEND_CONFIDENCE, authorize_action
from mailer_agent.policy.next_action import ActionType, NextActionProposal, plan_next_action
from mailer_agent.semantic_models import BuyingStage, IntentType, SemanticIntent, SentimentType


def _intent(**overrides) -> SemanticIntent:
    base = dict(
        intents=[IntentType.QUESTION],
        sentiment=SentimentType.NEUTRAL,
        buying_stage=BuyingStage.CONSIDERING,
        confidence=0.85,
        requires_human_review=False,
    )
    base.update(overrides)
    return SemanticIntent(**base)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def test_planner_parses_llm_proposal(fake_llm):
    fake_llm.queue_response({
        "action_type": "ask_targeted_question",
        "objective": "Find out what specifically concerns them about migration.",
        "reason": "They raised a vague switching-risk objection without specifics.",
        "required_information": [],
        "confidence": 0.8,
        "requires_human_review": False,
    })

    proposal = plan_next_action(
        intent=_intent(intents=[IntentType.OBJECTION], objections_raised=["switching risk"]),
        context_transcript="(conversation so far)",
    )

    assert proposal.action_type == ActionType.ASK_TARGETED_QUESTION
    assert proposal.objective == "Find out what specifically concerns them about migration."
    assert proposal.confidence == pytest.approx(0.8)
    assert proposal.requires_human_review is False
    assert proposal.source == "llm"


def test_planner_unknown_action_type_escalates(fake_llm):
    """A planner response outside the known ActionType vocabulary must
    not crash the pipeline -- it degrades to the conservative escalate
    fallback, same as any other malformed-output case in this codebase."""
    fake_llm.queue_response({
        "action_type": "book_meeting_immediately",  # not a real ActionType
        "objective": "test",
        "reason": "test",
        "confidence": 0.9,
        "requires_human_review": False,
    })

    proposal = plan_next_action(intent=_intent(), context_transcript="(none)")

    assert proposal.action_type == ActionType.ESCALATE
    assert proposal.requires_human_review is True
    assert proposal.source == "fallback"


def test_planner_response_missing_action_type_key_entirely_escalates(fake_llm):
    """
    Compatibility-mode validation (spec: 'do not treat merely
    syntactically valid JSON as equivalent to schema-valid output'):
    if the router falls back from strict_schema to json_object/lenient
    mode, the result is only guaranteed to be *parseable* JSON, not
    necessarily the expected shape. A response missing action_type
    entirely (not just an unrecognized value, but the key absent) must
    still degrade safely via the same ActionType(None) -> ValueError ->
    fallback path -- no separate validation framework needed, the
    existing per-field handling already covers this.
    """
    fake_llm.queue_response({
        "unrelated_key": "a completely different shape",
        "confidence": 0.9,
    })

    proposal = plan_next_action(intent=_intent(), context_transcript="(none)")

    assert proposal.action_type == ActionType.ESCALATE
    assert proposal.source == "fallback"


def test_planner_empty_objective_escalates(fake_llm):
    fake_llm.queue_response({
        "action_type": "answer",
        "objective": "",  # empty -- nothing for the responder to act on
        "reason": "test",
        "confidence": 0.9,
        "requires_human_review": False,
    })

    proposal = plan_next_action(intent=_intent(), context_transcript="(none)")

    assert proposal.action_type == ActionType.ESCALATE
    assert proposal.source == "fallback"


def test_planner_llm_unavailable_escalates(monkeypatch):
    import mailer_agent.llm.provider_v2 as provider_module
    monkeypatch.setattr(provider_module, "is_llm_available", lambda: False)

    proposal = plan_next_action(intent=_intent(), context_transcript="(none)")

    assert proposal.action_type == ActionType.ESCALATE
    assert proposal.requires_human_review is True
    assert proposal.confidence == 0.0
    assert proposal.source == "fallback"


def test_planner_provider_failure_escalates(fake_llm):
    from mailer_agent.llm.provider_v2 import RateLimitError
    fake_llm.queue_error(RateLimitError("429"))

    proposal = plan_next_action(intent=_intent(), context_transcript="(none)")

    assert proposal.action_type == ActionType.ESCALATE
    assert proposal.source == "fallback"


# ---------------------------------------------------------------------------
# Planner context completeness (spec sections 19, 49, 50): the planner can
# only weigh constraints, pain points, and commercial signals it actually
# SEES. SemanticIntent already carries all of these -- this proves they
# reach the planner's own prompt, not just that they exist on the dataclass.
# ---------------------------------------------------------------------------

def test_planner_prompt_includes_constraints(fake_llm):
    """The literal spec section 49 scenario: 'prospect interested but
    contract locked for 12 months' must actually be visible to the
    planner LLM, not silently dropped before the prompt is built."""
    fake_llm.queue_response({
        "action_type": "nurture", "objective": "test", "reason": "test",
        "confidence": 0.8, "requires_human_review": False,
    })

    plan_next_action(
        intent=_intent(
            intents=[IntentType.POSITIVE_INTEREST, IntentType.TIMING_CONSTRAINT],
            constraints=["locked into current vendor contract for 12 months"],
        ),
        context_transcript="(conversation so far)",
    )

    _, human_prompt = fake_llm.last_prompts[-1]
    assert "locked into current vendor contract for 12 months" in human_prompt


def test_planner_prompt_includes_pain_points_and_commitments(fake_llm):
    fake_llm.queue_response({
        "action_type": "answer", "objective": "test", "reason": "test",
        "confidence": 0.8, "requires_human_review": False,
    })

    plan_next_action(
        intent=_intent(
            pain_points=["manual data entry across three tools"],
            commitments_made=["I'll loop in our IT lead"],
        ),
        context_transcript="(conversation so far)",
    )

    _, human_prompt = fake_llm.last_prompts[-1]
    assert "manual data entry across three tools" in human_prompt
    assert "I'll loop in our IT lead" in human_prompt


def test_planner_prompt_includes_commercial_signals(fake_llm):
    """has_pricing_question was already surfaced; budget/decision-maker/
    commitment/procurement signals were not -- all five are 'commercial
    signals' per spec section 19 and must be equally visible."""
    fake_llm.queue_response({
        "action_type": "answer", "objective": "test", "reason": "test",
        "confidence": 0.8, "requires_human_review": False,
    })

    plan_next_action(
        intent=_intent(
            has_budget_signal=True,
            has_decision_maker_signal=True,
            has_commitment_signal=True,
            procurement_signal=True,
        ),
        context_transcript="(conversation so far)",
    )

    _, human_prompt = fake_llm.last_prompts[-1]
    assert "has_budget_signal" in human_prompt
    assert "has_decision_maker_signal" in human_prompt
    assert "has_commitment_signal" in human_prompt
    assert "procurement_signal" in human_prompt


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def _proposal(**overrides) -> NextActionProposal:
    base = dict(
        action_type=ActionType.ANSWER,
        objective="Answer their question.",
        reason="test",
        confidence=0.9,
        requires_human_review=False,
    )
    base.update(overrides)
    return NextActionProposal(**base)


def test_guardrail_authorizes_high_confidence_answer():
    authorized = authorize_action(_proposal(), intent=_intent(confidence=0.9), auto_reply_enabled=True)
    assert authorized.can_auto_send is True
    assert authorized.review_reason is None
    assert authorized.draft_action_type == "reply"


def test_guardrail_blocks_when_auto_reply_disabled():
    authorized = authorize_action(_proposal(), intent=_intent(), auto_reply_enabled=False)
    assert authorized.can_auto_send is False
    assert "AUTO_REPLY_ENABLED" in authorized.review_reason


def test_guardrail_blocks_low_classifier_confidence():
    authorized = authorize_action(
        _proposal(), intent=_intent(confidence=0.3), auto_reply_enabled=True
    )
    assert authorized.can_auto_send is False


def test_guardrail_blocks_low_planner_confidence():
    authorized = authorize_action(
        _proposal(confidence=0.2), intent=_intent(confidence=0.9), auto_reply_enabled=True
    )
    assert authorized.can_auto_send is False


def test_guardrail_respects_classifier_requires_human_review():
    authorized = authorize_action(
        _proposal(),
        intent=_intent(confidence=0.9, requires_human_review=True, human_review_reason="ambiguous"),
        auto_reply_enabled=True,
    )
    assert authorized.can_auto_send is False
    assert "ambiguous" in authorized.review_reason


def test_guardrail_respects_planner_requires_human_review():
    authorized = authorize_action(
        _proposal(requires_human_review=True, review_reason="needs judgment call"),
        intent=_intent(confidence=0.9),
        auto_reply_enabled=True,
    )
    assert authorized.can_auto_send is False
    assert "needs judgment call" in authorized.review_reason


@pytest.mark.parametrize("action_type", [ActionType.ESCALATE, ActionType.ADDRESS_OBJECTION])
def test_guardrail_never_auto_sends_certain_action_types(action_type):
    """Regression test for the direct replacement of the old
    ALWAYS_REQUIRE_APPROVAL_INTENTS set: these action categories are
    never eligible for auto-send, no matter how confident either the
    classifier or planner are."""
    authorized = authorize_action(
        _proposal(action_type=action_type, confidence=1.0),
        intent=_intent(confidence=1.0),
        auto_reply_enabled=True,
    )
    assert authorized.can_auto_send is False


def test_guardrail_never_auto_sends_pricing_answers():
    """Regression test: PRICING_REQUEST used to be hardcoded into
    ALWAYS_REQUIRE_APPROVAL_INTENTS directly; now it's the guardrail
    checking the classifier's has_pricing_question signal against
    whatever action the planner proposed -- same safety outcome, driven
    by the richer signal instead of raw intent membership."""
    authorized = authorize_action(
        _proposal(action_type=ActionType.PROVIDE_REQUESTED_INFORMATION, confidence=1.0),
        intent=_intent(confidence=1.0, has_pricing_question=True),
        auto_reply_enabled=True,
    )
    assert authorized.can_auto_send is False
    assert "pricing" in authorized.review_reason.lower()


def test_guardrail_maps_propose_next_step_to_closing_draft_type():
    authorized = authorize_action(
        _proposal(action_type=ActionType.PROPOSE_NEXT_STEP),
        intent=_intent(confidence=0.9),
        auto_reply_enabled=True,
    )
    assert authorized.draft_action_type == "closing"


def test_guardrail_maps_everything_else_to_reply_draft_type():
    for action_type in ActionType:
        if action_type == ActionType.PROPOSE_NEXT_STEP:
            continue
        authorized = authorize_action(
            _proposal(action_type=action_type),
            intent=_intent(confidence=0.9),
            auto_reply_enabled=True,
        )
        assert authorized.draft_action_type == "reply", action_type
