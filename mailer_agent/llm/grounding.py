"""
Grounding validation for LLM-generated outreach content.

Ensures the LLM does not invent facts that were never supplied in the
campaign or contact context.  The check is intentionally non-LLM:
regex + keyword heuristics run synchronously in microseconds and have
no external dependencies, so they cannot be the bottleneck or failure
mode in the drafting pipeline.

Design constraints
------------------
* No LLM calls — too slow and circular.
* No network calls — must work offline.
* Conservative flagging: false-positives (holding a good draft for
  review) are cheaper than false-negatives (sending a fabricated claim).
* The validator never blocks a send by itself; it marks the
  GroundingValidation result and the caller decides what to do with it.

Validation layers
-----------------
1. Claim extraction  — pull numbered/bulleted facts, percentage claims,
   "$" figures, quoted company names, and "guarantee/promise" language.
2. Proof-point check — each extracted claim must be traceable to at
   least one of: campaign.proof_points, contact.context_notes, or the
   conversation transcript.
3. Pricing/availability special class — any mention of price, cost,
   budget, discount, availability, or timeline is flagged as requiring
   explicit human review regardless of whether it's in the source data,
   because these are contractually sensitive.
4. Fabricated metric detection — bare numeric percentages, dollar
   amounts, headcount claims, or time spans that do not appear in any
   approved source text are always unsupported claims.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from mailer_agent.semantic_models import GroundingValidation

logger = logging.getLogger("mailer_agent.llm.grounding")

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Percentage claims: "40%", "up to 50 %", "3x faster", "2× ROI"
_PCT_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:%|percent|x faster|x roi|x return|×)",
    re.IGNORECASE,
)

# Dollar/currency claims: "$50K", "$1,200/mo", "£500", "€1000"
_DOLLAR_PATTERN = re.compile(
    r"[\$£€¥]\s*\d[\d,\.]*(?:\s*[kKmMbB])?",
    re.IGNORECASE,
)

# Headcount/team-size claims: "50 companies", "200 customers", "3 clients"
_HEADCOUNT_PATTERN = re.compile(
    r"\b(\d+)\s+(?:companies|customers|clients|users|teams|organizations|businesses|startups|brands|retailers|chains)",
    re.IGNORECASE,
)

# Time claims that could be commitments: "within 24 hours", "in 6 hours", "under a day"
_TIME_CLAIM_PATTERN = re.compile(
    r"\b(?:within|in under|in less than|under|less than)\s+\d+\s*(?:hours?|days?|minutes?|weeks?)",
    re.IGNORECASE,
)

# Setup/delivery time claims: "installs in under a day", "setup time 6 hours"
_SETUP_TIME_PATTERN = re.compile(
    r"\b(?:setup|install(?:ation)?|deploy(?:ment)?|onboard(?:ing)?|implement(?:ation)?)\s+"
    r"(?:time\s+)?(?:is\s+)?(?:only\s+)?(?:takes?\s+)?(?:in\s+)?(?:under\s+|less\s+than\s+|within\s+)?"
    r"\d+\s*(?:hours?|days?|minutes?|weeks?)",
    re.IGNORECASE,
)

# Guarantee/warranty language — always needs human eyes
_GUARANTEE_PATTERN = re.compile(
    r"\b(?:guarantee(?:d)?|warrant(?:y|ied)?|promise(?:d)?|committed\s+to|assured?)\b",
    re.IGNORECASE,
)

# Pricing-related terms — special class, always flagged for review
_PRICING_TERMS = re.compile(
    r"\b(?:price|pricing|cost|costs|fee|fees|rate|rates|discount|discounts|"
    r"budget|budgets|afford|affordable|cheap|expensive|quote|proposal|contract|"
    r"subscription|per\s+(?:seat|user|month|year)|monthly|annually|"
    r"free\s+trial|trial\s+period|available(?:ility)?)\b",
    re.IGNORECASE,
)

# Availability / launch / timeline promises
_AVAILABILITY_PATTERN = re.compile(
    r"\b(?:available\s+(?:now|today|immediately|this\s+week)|"
    r"launch(?:ing|es)?\s+(?:next\s+(?:week|month)|soon|shortly)|"
    r"ship(?:ping|s)?\s+(?:next\s+(?:week|month)))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

def extract_factual_claims(body: str) -> list[str]:
    """
    Extract candidate factual claims from a draft email body.

    Returns a list of short strings describing each extracted claim,
    suitable for display in review UIs or log entries.  Empty list means
    no numeric / commitment / pricing language was found.
    """
    claims: list[str] = []

    for match in _PCT_PATTERN.finditer(body):
        # Grab a short snippet of surrounding context (up to 80 chars)
        start = max(0, match.start() - 20)
        end = min(len(body), match.end() + 20)
        snippet = body[start:end].replace("\n", " ").strip()
        claims.append(f"metric: «{snippet}»")

    for match in _DOLLAR_PATTERN.finditer(body):
        start = max(0, match.start() - 20)
        end = min(len(body), match.end() + 20)
        snippet = body[start:end].replace("\n", " ").strip()
        claims.append(f"money: «{snippet}»")

    for match in _HEADCOUNT_PATTERN.finditer(body):
        start = max(0, match.start() - 10)
        end = min(len(body), match.end() + 30)
        snippet = body[start:end].replace("\n", " ").strip()
        claims.append(f"scale: «{snippet}»")

    for match in _TIME_CLAIM_PATTERN.finditer(body):
        start = max(0, match.start() - 10)
        end = min(len(body), match.end() + 30)
        snippet = body[start:end].replace("\n", " ").strip()
        claims.append(f"time-claim: «{snippet}»")

    for match in _SETUP_TIME_PATTERN.finditer(body):
        start = max(0, match.start() - 5)
        end = min(len(body), match.end() + 30)
        snippet = body[start:end].replace("\n", " ").strip()
        claims.append(f"setup-claim: «{snippet}»")

    for match in _GUARANTEE_PATTERN.finditer(body):
        start = max(0, match.start() - 20)
        end = min(len(body), match.end() + 40)
        snippet = body[start:end].replace("\n", " ").strip()
        claims.append(f"guarantee: «{snippet}»")

    return claims


# ---------------------------------------------------------------------------
# Source-text lookup helpers
# ---------------------------------------------------------------------------

def _build_approved_corpus(*sources: str | None) -> str:
    """
    Concatenate all non-empty approved source strings into one lowercase
    blob for substring searching.
    """
    return " ".join(s.lower() for s in sources if s)


def _numeric_value_in_corpus(numeric_str: str, corpus: str) -> bool:
    """
    Return True if the raw numeric string (e.g. "40", "6") appears in the
    corpus.  We require the number to appear as a whole word/token to avoid
    "600" matching "40" via substring.
    """
    pattern = re.compile(r"\b" + re.escape(numeric_str) + r"\b")
    return bool(pattern.search(corpus))


def _extract_numeric(claim_snippet: str) -> list[str]:
    """Pull all digit sequences out of a claim snippet."""
    return re.findall(r"\d+(?:\.\d+)?", claim_snippet)


# ---------------------------------------------------------------------------
# Main grounding check
# ---------------------------------------------------------------------------

def validate_grounding(
    draft_body: str,
    *,
    proof_points: str | None,
    context_notes: str | None,
    conversation_transcript: str | None,
    value_prop: str | None = None,
) -> GroundingValidation:
    """
    Validate ``draft_body`` against all approved source material.

    Parameters
    ----------
    draft_body:
        The LLM-generated email body to check.
    proof_points:
        ``campaign.proof_points`` — facts the SDR is explicitly allowed to cite.
    context_notes:
        ``contact.context_notes`` — verified facts about this specific prospect.
    conversation_transcript:
        The full conversation history as a plain-text string (outbound +
        inbound) — things the *prospect* said count as verified facts too.
    value_prop:
        ``campaign.value_prop`` — the core offer text; numeric claims
        that appear here (e.g. "cut response time 40%") are allowed.

    Returns
    -------
    GroundingValidation
        ``is_grounded=True`` iff no unsupported claims were found AND no
        pricing/availability triggers were hit.  ``is_safe_to_send`` is
        ``True`` only when ``is_grounded`` is True and ``unsupported_claims``
        is empty.
    """
    corpus = _build_approved_corpus(proof_points, context_notes, conversation_transcript, value_prop)

    claims = extract_factual_claims(draft_body)

    unsupported: list[str] = []
    supported: list[str] = []
    notes: list[str] = []
    pricing_flagged = False

    # ----------------------------------------------------------------
    # 1. Pricing / availability special class — always flag for review
    # ----------------------------------------------------------------
    if _PRICING_TERMS.search(draft_body):
        pricing_flagged = True
        notes.append("Draft contains pricing/cost/availability language — requires human review.")
        # Extract which terms were found
        found_pricing = list({m.group(0).lower() for m in _PRICING_TERMS.finditer(draft_body)})
        unsupported.append(f"pricing-terms: {', '.join(sorted(found_pricing)[:5])}")

    if _AVAILABILITY_PATTERN.search(draft_body):
        pricing_flagged = True
        notes.append("Draft contains availability/launch timeline claim — requires human review.")
        unsupported.append("availability-claim detected")

    # ----------------------------------------------------------------
    # 2. Check each extracted claim against approved corpus
    # ----------------------------------------------------------------
    for claim in claims:
        # Extract numeric tokens from the claim snippet
        numerics = _extract_numeric(claim)

        claim_supported = True
        if numerics and corpus:
            # All numeric values in the claim must appear in the corpus
            for num in numerics:
                if not _numeric_value_in_corpus(num, corpus):
                    claim_supported = False
                    break
        elif not corpus:
            # No approved source data at all — any numeric claim is unsupported
            if numerics:
                claim_supported = False

        # Special: guarantee language is always unsupported unless it
        # literally appears in the approved source material
        if claim.startswith("guarantee:"):
            claim_lower = claim.lower()
            if "guarantee" not in corpus and "warrant" not in corpus:
                claim_supported = False

        if claim_supported:
            supported.append(claim)
        else:
            unsupported.append(claim)

    # ----------------------------------------------------------------
    # 3. Determine overall grounding status
    # ----------------------------------------------------------------
    is_grounded = (len(unsupported) == 0) and (not pricing_flagged)

    if notes:
        validation_notes = "; ".join(notes)
    elif unsupported:
        validation_notes = f"{len(unsupported)} unsupported claim(s) found"
    else:
        validation_notes = "All claims traceable to approved source material"

    # Confidence: 1.0 if clean, scaled down by unsupported count
    total = len(supported) + len(unsupported)
    if total == 0:
        confidence = 1.0  # nothing to check — trivially grounded
    else:
        confidence = len(supported) / total

    if pricing_flagged:
        confidence = min(confidence, 0.0)  # pricing always drops to 0

    logger.debug(
        "Grounding check: grounded=%s, unsupported=%d, supported=%d, pricing=%s",
        is_grounded,
        len(unsupported),
        len(supported),
        pricing_flagged,
    )

    return GroundingValidation(
        is_grounded=is_grounded,
        unsupported_claims=unsupported,
        supported_claims=supported,
        confidence=confidence,
        validation_notes=validation_notes,
    )
