"""
Minimal live smoke test for the provider-routing changes (spec: "LIVE
TESTING -- Do not immediately run the entire expensive live semantic
suite. First create/run a very small live smoke test").

Three calls total: one classifier call, one planner call, one
structured-output call reusing the classifier's real strict schema.
That's it -- this is deliberately NOT
tests/evaluation/test_live_llm_semantic_quality.py (7 tests, several
scenarios each), which should only be re-run once this smoke test is
stable, per the same instruction.

Cannot be executed in the sandbox this patch was developed in (no
network egress to api.groq.com there -- confirmed via direct curl
earlier in this project). Run it yourself:

    python -m pytest -o addopts="" tests/evaluation/test_live_smoke.py -v -m integration

Uses whatever LLM_MODEL/LLM_FALLBACK_MODELS/LLM_REASONING_EFFORT are
set in your real .env -- it does not override them, so this also
implicitly smoke-tests your actual configured provider setup, not a
hardcoded one.
"""

from __future__ import annotations

import pytest

from mailer_agent.llm.provider_v2 import is_llm_available
from mailer_agent.semantic.classifier import classify_prospect_reply
from mailer_agent.models import Campaign, Contact

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not is_llm_available(),
        reason=(
            "Live smoke test requires a configured GROQ_API_KEY. "
            "is_llm_available() reads it through the application's own "
            "settings layer (mailer_agent.config.get_settings(), which "
            "loads .env) -- not os.environ directly, since a key set only "
            "in .env (not exported into the shell) is a real, correctly-"
            "configured setup that a raw os.environ.get check would "
            "incorrectly skip."
        ),
    ),
]


@pytest.fixture
def smoke_campaign():
    return Campaign(
        id=1, name="Smoke Test", organization_id="default",
        sender_name="Jordan", sender_org="TestCorp", sender_email="jordan@testcorp.example.com",
        value_prop="We help B2B SaaS teams cut support response time 40% with an AI triage layer.",
        proof_points="Used by 3 YC-backed startups; average setup time 6 hours.",
    )


@pytest.fixture
def smoke_contact(smoke_campaign):
    return Contact(id=1, campaign_id=smoke_campaign.id, name="Sam", email="sam@prospect.example.com")


def test_smoke_provider_is_configured():
    """If this fails, nothing else in this file will work either --
    check GROQ_API_KEY is actually set, not just present-but-empty."""
    assert is_llm_available(), (
        "is_llm_available() returned False even though GROQ_API_KEY is "
        "set in the environment -- check mailer_agent/config.py's "
        "Settings picks it up (e.g. .env not loaded, wrong variable name)."
    )


def test_smoke_one_classifier_call_succeeds(smoke_campaign, smoke_contact):
    """
    The single most important smoke check: does a real classification
    call, through the FULL router (strict schema -> json_object ->
    lenient, with model fallback), actually come back successful with
    this session's model/schema/reasoning_effort changes -- not whether
    the semantics are good (that's the separate, more expensive live
    suite's job).
    """
    result = classify_prospect_reply(
        campaign=smoke_campaign, contact=smoke_contact,
        inbound_body="This looks interesting -- can we do a call next week?",
        conversation_context="(no prior messages)",
    )

    assert result.success, f"Classification failed: {result.failure_details}"
    assert result.semantic_intent is not None
    assert result.model_used, "model_used must be populated (not left empty/misleading)"
    print(f"\n[smoke] classifier: model_used={result.model_used} "
          f"used_fallback={result.used_fallback} response_mode={result.response_mode} "
          f"attempts={result.attempts}")


def test_smoke_one_planner_call_succeeds(smoke_campaign, smoke_contact):
    from mailer_agent.policy.next_action import plan_next_action
    from mailer_agent.semantic_models import IntentType, SemanticIntent

    intent = SemanticIntent(
        intents=[IntentType.MEETING_REQUEST],
        user_goal="schedule a call",
        confidence=0.9,
    )
    proposal = plan_next_action(intent=intent, context_transcript="(smoke test conversation)")

    assert proposal.source == "llm", (
        f"Expected the planner to actually reach the LLM (source='llm'), got "
        f"source={proposal.source!r} -- likely fell back to the deterministic "
        f"escalate path, meaning the live call itself failed. "
        f"reason={proposal.reason!r}"
    )
    print(f"\n[smoke] planner: model_used={proposal.model_used} "
          f"used_fallback={proposal.used_fallback} response_mode={proposal.response_mode}")


def test_smoke_structured_output_call_with_strict_schema(smoke_campaign, smoke_contact):
    """
    Explicitly re-exercises the classifier's real CLASSIFIER_JSON_SCHEMA
    (the biggest, most failure-prone of the three schemas) end to end,
    separately from test_smoke_one_classifier_call_succeeds, so a
    strict-schema-specific failure is distinguishable from a general
    classification failure in the pytest output.
    """
    result = classify_prospect_reply(
        campaign=smoke_campaign, contact=smoke_contact,
        inbound_body="We already have a CRM in place, but I'd be curious what makes yours different.",
        conversation_context="(no prior messages)",
    )

    assert result.success, f"Structured-output call failed: {result.failure_details}"
    print(f"\n[smoke] structured output: response_mode={result.response_mode} "
          f"(expect 'strict_schema' if the primary model handled it cleanly, "
          f"'json_object' or 'lenient' if it had to fall back)")
