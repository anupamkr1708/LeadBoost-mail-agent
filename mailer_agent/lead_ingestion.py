"""
Lead ingestion: turns "a lead in whatever shape the caller has it" into
the Contact fields this service actually needs (email/name/title/company/
context_notes).

This exists because the brief was explicit: the service needs to
understand what format a lead is arriving in, not assume every caller
already matches an exact schema. Concretely tuned for LeadBoost's own
`Lead` model fields (company_name, contact_name, contact_title, email,
about_text, industry, employees, revenue_band, founded_year,
qualification_label, score, website, linkedin_url) while still accepting
generic aliases (name/email/company/title) so any other caller works too.

Anything recognized as a *fact* (industry, employee band, a recent
signal in about_text, etc.) is folded into `context_notes` -- the exact
field the drafting agent treats as "verified, citable" -- rather than
discarded, since that's what keeps generated copy specific instead of
generic.
"""

from __future__ import annotations

from typing import Any

EMAIL_ALIASES = ["email", "contact_email", "lead_email", "to_email"]
NAME_ALIASES = ["contact_name", "name", "full_name", "lead_name"]
TITLE_ALIASES = ["contact_title", "title", "job_title", "role", "position"]
COMPANY_ALIASES = ["company_name", "company", "organization_name", "org_name", "org"]

# Fields folded into context_notes when present, in a fixed, readable order.
# Label -> list of accepted keys for that fact.
CONTEXT_FIELD_ALIASES: dict[str, list[str]] = {
    "Industry": ["industry"],
    "Company size": ["employees", "employee_count", "headcount"],
    "Revenue band": ["revenue_band", "revenue"],
    "Founded": ["founded_year"],
    "Website": ["website", "domain"],
    "About": ["about_text", "description", "company_description"],
    "LinkedIn": ["linkedin_url"],
    "Lead qualification": ["qualification_label"],
    "Lead score": ["score", "overall_score"],
}


class LeadIngestionError(ValueError):
    pass


def _first_present(payload: dict[str, Any], keys: list[str]) -> Any | None:
    for k in keys:
        v = payload.get(k)
        if v not in (None, ""):
            return v
    return None


def normalize_lead_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Returns {"email": str, "name": str|None, "title": str|None,
    "company": str|None, "context_notes": str|None}.

    Raises LeadIngestionError if no usable email could be found under
    any known alias -- a contact with no email can't be mailed, so this
    fails loudly at ingestion time rather than silently later.
    """
    email = _first_present(raw, EMAIL_ALIASES)
    if not email:
        raise LeadIngestionError(
            f"Could not find an email field. Looked for: {EMAIL_ALIASES}. "
            f"Payload keys received: {list(raw.keys())}"
        )

    name = _first_present(raw, NAME_ALIASES)
    title = _first_present(raw, TITLE_ALIASES)
    company = _first_present(raw, COMPANY_ALIASES)

    context_lines = []
    for label, keys in CONTEXT_FIELD_ALIASES.items():
        value = _first_present(raw, keys)
        if value is not None:
            context_lines.append(f"{label}: {value}")

    # Anything else the caller included that we don't explicitly recognize
    # -- still surfaced, clearly marked as unverified/raw, never silently
    # dropped, but kept separate so the agent doesn't treat it with the
    # same confidence as the recognized fields above.
    known_keys = set(EMAIL_ALIASES + NAME_ALIASES + TITLE_ALIASES + COMPANY_ALIASES)
    for keys in CONTEXT_FIELD_ALIASES.values():
        known_keys.update(keys)
    extra = {k: v for k, v in raw.items() if k not in known_keys and v not in (None, "")}
    if extra:
        context_lines.append(f"Other supplied fields: {extra}")

    return {
        "email": str(email).strip().lower(),
        "name": str(name).strip() if name else None,
        "title": str(title).strip() if title else None,
        "company": str(company).strip() if company else None,
        "context_notes": "\n".join(context_lines) if context_lines else None,
    }
