"""
LLM-free, sender-free suppression lookup for the exact-message dispatch path.

Why this module exists
----------------------
``followup.engine.is_suppressed`` is the project's suppression check, but
importing ``followup.engine`` loads ``llm.agent`` and ``mail.sender`` (the
whole drafting stack). The LeadBoost dispatch worker must evaluate
suppression without dragging either in, so this is the *minimum* extraction:
one pure database lookup and nothing else.

Semantics are IDENTICAL to ``followup.engine.is_suppressed`` (which is left
untouched -- legacy callers keep using it):

* exact ``email`` match against ``suppression_list`` (no case folding, no
  trimming -- the same as the legacy check and as POST /suppress, which
  stores the address as given);
* ``org_id`` given  -> only that organization's entries count;
* ``org_id=None``   -> entries from ANY organization count. This is the
  deliberate safe default the legacy scheduler/approval paths use: an address
  that unsubscribed from one org's campaign is not mailed by another.
  tests/test_external_dispatch_worker.py pins parity against the legacy
  function over a matrix so the two cannot silently drift.

Allowed imports (enforced by tests/test_external_dispatch_boundary.py):
``sqlalchemy.orm.Session`` and ``mailer_agent.models`` only.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from mailer_agent.models import SuppressionEntry


def is_email_suppressed(db: Session, email_addr: str, org_id: str | None = None) -> bool:
    query = db.query(SuppressionEntry).filter_by(email=email_addr)
    if org_id is not None:
        query = query.filter_by(organization_id=org_id)
    return query.first() is not None
