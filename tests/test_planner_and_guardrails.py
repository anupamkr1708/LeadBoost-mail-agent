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
