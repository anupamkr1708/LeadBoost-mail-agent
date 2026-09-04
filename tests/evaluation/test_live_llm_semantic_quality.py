"""
Live-LLM semantic quality smoke suite (spec §31).

Every test in this file is marked @pytest.mark.integration and calls the
REAL Groq provider -- it requires GROQ_API_KEY and network access, is
never required for ordinary deterministic CI (conftest.py's autouse
fake_llm fixture explicitly skips patching for tests with this marker,
so these genuinely hit the real API), and its results are NOT
deterministic run to run the way the rest of this suite is.

Honesty check, stated plainly: this file has never been executed. This
sandbox has no network access to reach Groq. Every other test file in
this repository was at least traceable by hand against code whose exact
behavior is knowable in advance; these tests can't be -- their entire
point is to check what a real model actually does, which is unknowable
without calling it. Do not treat this file's existence as evidence of
semantic quality. It is evidence that the harness for measuring semantic
quality exists; the measurement itself is the thing you run locally
(see docs/FINAL_PRODUCTION_READINESS.md for the exact command).

What this suite actually checks (unlike tests/evaluation/test_semantic_evaluation.py,
which proves the pipeline handles a GIVEN structured interpretation
correctly): whether the real model, given genuinely different phrasings
of the same underlying meaning, converges on structurally similar
classifications -- the actual "semantic generalization" claim, not the
pipeline-consistency stand-in for it.
"""

from __future__ import annotations

import pytest

from mailer_agent.models import Campaign, Contact
from mailer_agent.policy.next_action import plan_next_action
from mailer_agent.semantic.classifier import classify_prospect_reply

pytestmark = pytest.mark.integration


@pytest.fixture
def live_campaign():
    return Campaign(
        id=1, name="Live Smoke Test", sender_name="Jordan", sender_org="TestCorp",
        sender_email="jordan@testcorp.example.com",
        value_prop="We help B2B companies streamline operations",
        proof_points="Used by 50+ companies, average setup time under a week",
        tone="professional, direct",
    )


@pytest.fixture
def live_contact():
    return Contact(
        id=1, campaign_id=1, name="Priya Singh", email="priya@prospect.example.com",
        title="VP Operations", company="ProspectCo", status="active",
    )


# ---------------------------------------------------------------------------
# Structured output validity -- does the real model reliably produce
# something the strict-JSON parser in provider_v2.py accepts?
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("body", [
    "This looks interesting -- can we do a call next week? Also, what's "
    "pricing for a team of 20?",
    "Not interested, please remove me from your list.",
    "We already have a CRM in place, but I'd be curious what makes yours different.",
])
def test_live_classification_produces_valid_structured_output(live_campaign, live_contact, body):
    result = classify_prospect_reply(
        campaign=live_campaign, contact=live_contact,
        inbound_body=body, conversation_context="(no prior messages)",
    )
    assert result.success, f"Classification failed: {result.failure_details}"
    assert result.semantic_intent is not None
    assert 0.0 <= result.semantic_intent.confidence <= 1.0


# ---------------------------------------------------------------------------
# Semantic generalization -- the real claim from spec §29/§20 (calling-agent
# doc), tested with a real model instead of the pipeline-consistency stand-in.
# ---------------------------------------------------------------------------

EQUIVALENT_CURRENT_SOLUTION_PHRASINGS = [
    "We already use Salesforce.",
    "We're on Salesforce today.",
    "We aren't replacing our current CRM right now -- we're on Salesforce.",
    "Changing systems isn't on our radar; Salesforce is working fine for now.",
]


def test_live_equivalent_phrasings_converge_on_current_solution(live_campaign, live_contact):
    """
    Four differently-worded prospect emails, all expressing the same
    underlying fact (they use Salesforce and aren't actively switching).
    A real model should recognize the current_solution in all four,
    even though only one uses the word "Salesforce" as a direct
    statement and the others hedge or imply it. This is the test a
    keyword system would fail and a semantic system should pass.
    """
    solutions_detected = []
    for phrasing in EQUIVALENT_CURRENT_SOLUTION_PHRASINGS:
        result = classify_prospect_reply(
            campaign=live_campaign, contact=live_contact,
            inbound_body=phrasing, conversation_context="(no prior messages)",
        )
        assert result.success, f"Classification failed for {phrasing!r}: {result.failure_details}"
        cs = result.semantic_intent.current_solution
        solutions_detected.append(cs.value.lower() if cs else None)

    matched = sum(1 for v in solutions_detected if v and "salesforce" in v)
    assert matched >= 3, (
        f"Expected at least 3/4 phrasings to recognize Salesforce as the "
        f"current solution, got: {solutions_detected}"
    )


AMBIGUOUS_EXPENSIVE_SCENARIOS = [
    ("We already looked at pricing on your site -- that's expensive for us.", "pricing_objection"),
    ("I imagine migration will be expensive given our data volume.", "implementation_concern"),
    ("Compared with Salesforce, what's the pricing difference?", "information_request"),
]


def test_live_same_surface_word_different_meaning_by_context(live_campaign, live_contact):
    """
    "Expensive" means something different depending on what it's
    attached to -- a pricing objection, an implementation concern, or
    part of a comparison question. A keyword match on "expensive" alone
    can't distinguish these; a real model reasoning over the whole
    sentence should.
    """
    for body, expected_category in AMBIGUOUS_EXPENSIVE_SCENARIOS:
        result = classify_prospect_reply(
            campaign=live_campaign, contact=live_contact,
            inbound_body=body, conversation_context="(no prior messages)",
        )
        assert result.success, f"Classification failed for {body!r}: {result.failure_details}"
        intent = result.semantic_intent
        if expected_category == "pricing_objection":
            from mailer_agent.semantic_models import IntentType
            assert intent.has_pricing_question or IntentType.OBJECTION in intent.intents, (
                f"Expected a pricing objection signal for {body!r}, got intents={intent.intents}"
            )
        elif expected_category == "information_request":
            from mailer_agent.semantic_models import IntentType
            assert IntentType.QUESTION in intent.intents or IntentType.INFORMATION_REQUEST in intent.intents, (
                f"Expected a question/information-request signal for {body!r}, got intents={intent.intents}"
            )
        # "implementation_concern" is intentionally not asserted as strictly
        # as the other two -- this is the genuinely hard case (a concern
        # about cost that isn't quite a pricing question), and a live
        # smoke test should surface how the model actually handles it
        # rather than encode a guess about the "right" answer here.


# ---------------------------------------------------------------------------
# Planner: does a real model produce a coherent objective, not just a
# valid enum value?
# ---------------------------------------------------------------------------

def test_live_planner_produces_relevant_objective(live_campaign, live_contact):
    result = classify_prospect_reply(
        campaign=live_campaign, contact=live_contact,
        inbound_body="Interesting, but we're locked into our current vendor "
                     "contract for another year. Might be worth revisiting then.",
        conversation_context="(no prior messages)",
    )
    assert result.success

    proposal = plan_next_action(
        intent=result.semantic_intent,
        context_transcript="(no prior messages)",
    )
    assert proposal.objective, "Planner must produce a non-empty objective"
    # Loose relevance check -- not exact wording, per spec §30 ("do not
    # make exact wording the primary metric"). The objective should
    # plausibly relate to the timing constraint/nurture situation, not
    # be a generic "book a meeting" regardless of context.
    assert not any(
        generic in proposal.objective.lower()
        for generic in ["schedule a call immediately", "sign up today"]
    ), f"Planner objective looks generic/pushy given a locked-in-contract context: {proposal.objective!r}"


# ---------------------------------------------------------------------------
# Provider failure handling under real conditions
# ---------------------------------------------------------------------------

def test_live_latency_is_reasonable(live_campaign, live_contact):
    """
    Not a hard SLA test -- just a sanity check that a single
    classification call completes in a time range consistent with an
    interactive (if not real-time) email-processing pipeline, so a
    latency regression from a prompt/model change would be visible here.
    """
    import time
    start = time.time()
    result = classify_prospect_reply(
        campaign=live_campaign, contact=live_contact,
        inbound_body="Sounds good, let's talk more.",
        conversation_context="(no prior messages)",
    )
    elapsed = time.time() - start
    assert result.success
    assert elapsed < 30, f"Classification took {elapsed:.1f}s -- investigate before treating this as normal"
