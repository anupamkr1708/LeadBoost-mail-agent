"""
Strict JSON Schema definitions for Groq's structured-output mode
(response_format={"type": "json_schema", "json_schema": {...}}).

These are NOT a second, independently-invented schema. Each one mirrors
the JSON shape already documented in the corresponding system prompt --
semantic/classifier.py's SEMANTIC_CLASSIFIER_SYSTEM_PROMPT,
llm/prompts.py's PLANNER_SYSTEM_PROMPT, and AGENT_SYSTEM_PROMPT -- which
remain the single source of truth for WHAT fields mean. If you change a
field in one of those prompts, update the matching schema here in the
same change, or the strict-mode request will reject (or silently
truncate) exactly the field you just added.

Design choices, and why:

1. Enum-like string fields (sentiment, buying_stage, intents entries,
   certainty, etc.) are typed as plain "string", NOT constrained with a
   JSON Schema "enum" list. This is deliberate: the domain-model parsing
   layer (semantic_models.py / semantic/classifier.py's _parse_certainty
   etc.) already handles an unrecognized value gracefully -- it drops it
   to a safe default and logs a warning, rather than crashing. Encoding
   the current enum set as a hard "enum" constraint in the strict schema
   would upgrade "the model used a slightly novel phrasing, or we add a
   new IntentType next month" from a graceful degrade into an outright
   400 rejection of the entire response -- strictly LESS robust than
   what the application already does. That would be making the schema
   MORE restrictive than the application's actual domain model, which
   we were explicitly told not to do.

2. Every object has "additionalProperties": false and lists every one
   of its properties in "required" (Groq/OpenAI-style strict mode
   requires this -- optionality is expressed via a ["type", "null"]
   union on the property itself, never by omitting it from "required").

3. Nested "fact" objects (current_solution, each entry of
   competitors_mentioned/new_facts) are inlined rather than shared via
   $ref/$defs -- Groq's strict-schema support for JSON Schema
   references is not something we've verified works, so this avoids
   depending on an untested feature for a production request path.
"""

from __future__ import annotations

_FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string"},
        "certainty": {"type": "string"},
        "evidence": {"type": ["string", "null"]},
        "supersedes": {"type": ["string", "null"]},
    },
    "required": ["value", "certainty", "evidence", "supersedes"],
    "additionalProperties": False,
}

_FACT_SCHEMA_NO_SUPERSEDES = {
    # competitors_mentioned entries don't carry `supersedes` in the
    # documented prompt shape (only current_solution/new_facts do).
    "type": "object",
    "properties": {
        "value": {"type": "string"},
        "certainty": {"type": "string"},
        "evidence": {"type": ["string", "null"]},
    },
    "required": ["value", "certainty", "evidence"],
    "additionalProperties": False,
}

_TIMING_SCHEMA = {
    "type": "object",
    "properties": {
        "expression": {"type": "string"},
        "kind": {"type": "string"},
        "normalized_target": {"type": ["string", "null"]},
        "certainty": {"type": "string"},
        "commitment_strength": {"type": "string"},
        "requires_clarification": {"type": "boolean"},
    },
    "required": [
        "expression", "kind", "normalized_target", "certainty",
        "commitment_strength", "requires_clarification",
    ],
    "additionalProperties": False,
}

CLASSIFIER_JSON_SCHEMA = {
    "name": "semantic_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "intents": {"type": "array", "items": {"type": "string"}},
            "speech_act": {"type": "string"},
            "sentiment": {"type": "string"},
            "buying_stage": {"type": "string"},
            "user_goal": {"type": ["string", "null"]},
            "pain_points": {"type": "array", "items": {"type": "string"}},
            "objections_raised": {"type": "array", "items": {"type": "string"}},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "urgency": {"type": "string"},
            "has_pricing_question": {"type": "boolean"},
            "has_budget_signal": {"type": "boolean"},
            "has_decision_maker_signal": {"type": "boolean"},
            "has_commitment_signal": {"type": "boolean"},
            "procurement_signal": {"type": "boolean"},
            "requested_information": {"type": "array", "items": {"type": "string"}},
            "questions_asked": {"type": "array", "items": {"type": "string"}},
            "commitments_made": {"type": "array", "items": {"type": "string"}},
            "current_solution": {"type": ["object", "null"], **{k: v for k, v in _FACT_SCHEMA.items() if k != "type"}},
            "competitors_mentioned": {"type": "array", "items": _FACT_SCHEMA_NO_SUPERSEDES},
            "new_facts": {"type": "array", "items": _FACT_SCHEMA},
            "contradicted_facts": {"type": "array", "items": {"type": "string"}},
            "unresolved_items": {"type": "array", "items": {"type": "string"}},
            "timing": {"type": ["object", "null"], **{k: v for k, v in _TIMING_SCHEMA.items() if k != "type"}},
            "confidence": {"type": "number"},
            "uncertain_aspects": {"type": "array", "items": {"type": "string"}},
            "reasoning": {"type": "string"},
            "requires_human_review": {"type": "boolean"},
            "human_review_reason": {"type": ["string", "null"]},
        },
        "required": [
            "intents", "speech_act", "sentiment", "buying_stage", "user_goal",
            "pain_points", "objections_raised", "constraints", "urgency",
            "has_pricing_question", "has_budget_signal", "has_decision_maker_signal",
            "has_commitment_signal", "procurement_signal", "requested_information",
            "questions_asked", "commitments_made", "current_solution",
            "competitors_mentioned", "new_facts", "contradicted_facts",
            "unresolved_items", "timing", "confidence", "uncertain_aspects",
            "reasoning", "requires_human_review", "human_review_reason",
        ],
        "additionalProperties": False,
    },
}

PLANNER_JSON_SCHEMA = {
    "name": "next_action_proposal",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action_type": {"type": "string"},
            "objective": {"type": "string"},
            "reason": {"type": "string"},
            "required_information": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "requires_human_review": {"type": "boolean"},
            "review_reason": {"type": ["string", "null"]},
        },
        "required": [
            "action_type", "objective", "reason", "required_information",
            "confidence", "requires_human_review", "review_reason",
        ],
        "additionalProperties": False,
    },
}

RESPONDER_JSON_SCHEMA = {
    "name": "drafted_email",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["subject", "body", "reasoning"],
        "additionalProperties": False,
    },
}
