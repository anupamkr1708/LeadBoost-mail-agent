"""
Conversation memory.

The Message table *is* the memory -- nothing here duplicates it. This
module's job is just to turn "every Message row for this Contact" into
a bounded, LLM-ready context: recent messages verbatim, anything older
collapsed into a short rolling summary so a 15-message thread doesn't
blow the prompt budget or make the model lose track of what actually
matters (the last couple of exchanges).

Conversational evidence vs. every row in the table
-----------------------------------------------------
Not every Message row is something that actually happened in the
conversation. A DRAFT is a message the system considered sending but
hasn't (or was blocked from sending); a FAILED or UNKNOWN-outcome
outbound message may never have reached the prospect at all. None of
those are conversational EVENTS -- they're internal drafting/send-state
artifacts -- and including them in the context fed back into prompting
or grounding is not a neutral inclusion, it's actively dangerous: a
draft containing an unsupported claim would then appear as its own
"supporting evidence" the next time grounding checks that same draft
(or a similar one) against "the conversation so far", making the check
circular. `_conversational_evidence()` is the one place this filter is
defined, used by both functions below, so a future evidence source
doesn't have to remember to reapply it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from mailer_agent.llm.provider import LLMOutputError, LLMUnavailableError, call_llm_text
from mailer_agent.models import Contact, Message, MessageDirection, MessageStatus

logger = logging.getLogger("mailer_agent.memory")

# How many of the most recent messages to always include verbatim.
RECENT_MESSAGES_VERBATIM = 6
# Once a thread exceeds this many messages, summarize the overflow.
SUMMARIZE_THRESHOLD = 8


def _conversational_evidence(messages: list[Message]) -> list[Message]:
    """
    Filter every Message row down to only what actually happened in the
    conversation, preserving chronological order.

    - Inbound messages are always valid evidence (an inbound row only
      ever gets created for a message that was genuinely received --
      see mail/reply_handler_v2.py -- so there's no separate "inbound
      status" to check here, unlike outbound).
    - Outbound messages count only when status is exactly SENT. DRAFT
      (never sent), FAILED (definitely didn't reach the prospect), and
      UNKNOWN (ambiguous -- see mail/sender.py's SendOutcome docstring;
      may or may not have sent, and treating "might have sent" as
      "definitely sent history" is exactly the kind of unsafe optimism
      that outcome exists to prevent elsewhere in this codebase) are all
      excluded. APPROVED/PENDING/SENDING are reserved, currently-unset
      statuses (see MessageStatus's own docstring) -- not treated as
      evidence either, on the same "don't assume sent" principle, should
      they ever start being used.
    - No semantic/keyword logic here -- this is a status-field check
      against the actual MessageStatus/MessageDirection enums already
      defined in models.py, nothing more.
    """
    evidence = []
    for m in messages:
        if m.direction == MessageDirection.INBOUND.value:
            evidence.append(m)
        elif m.direction == MessageDirection.OUTBOUND.value and m.status == MessageStatus.SENT.value:
            evidence.append(m)
    return evidence


def build_conversation_context(db: Session, contact: Contact) -> str:
    """
    Returns a human-readable transcript string for prompting: an optional
    leading summary line for older history, then the recent messages
    verbatim in chronological order, each labeled by who sent it and when.

    Only actual conversational evidence is included (see
    _conversational_evidence) -- a DRAFT or FAILED/UNKNOWN-outcome
    outbound message never appears here, regardless of how recent it is.
    """
    messages: list[Message] = _conversational_evidence(list(contact.messages))

    if not messages:
        return "(no messages sent yet -- this is the first contact)"

    parts: list[str] = []

    if contact.memory_summary:
        parts.append(f"[Summary of earlier conversation]: {contact.memory_summary}")

    recent = messages[-RECENT_MESSAGES_VERBATIM:]
    for m in recent:
        # Every m here is either inbound or a confirmed-SENT outbound
        # (see _conversational_evidence), so this label is now always
        # accurate -- it used to read "US (sent)" for ANY outbound
        # message regardless of whether it actually was.
        who = "US (sent)" if m.direction == "outbound" else "THEM (received)"
        ts = m.created_at.strftime("%Y-%m-%d") if m.created_at else "?"
        subject = f" | Subject: {m.subject}" if m.subject else ""
        parts.append(f"[{ts}] {who}{subject}\n{m.body.strip()}")

    return "\n\n---\n\n".join(parts)


def build_known_facts_context(contact: Contact) -> str:
    """
    Facts for the planner prompt, formatted from reconcile_known_facts()
    (below) -- the actual State Reconciliation stage. See that function's
    docstring for the reconciliation rules; this just renders its output
    as text, clearly separating what's CURRENT from what's HISTORY
    (spec section 15: "history != current truth"), instead of silently
    dropping or conflating superseded values.
    """
    knowledge = reconcile_known_facts(contact)

    lines: list[str] = []
    if knowledge.current_solution:
        lines.append(
            f"current_solution: {knowledge.current_solution.value} "
            f"(certainty={knowledge.current_solution.certainty})"
        )
    for h in knowledge.current_solution_history:
        lines.append(f"current_solution (superseded, no longer current): {h.value}")

    current_facts = [f for f in knowledge.facts if f.status == "current"]
    superseded_facts = [f for f in knowledge.facts if f.status == "superseded"]
    for f in current_facts:
        lines.append(f"fact: {f.value} (certainty={f.certainty})")
    if superseded_facts:
        lines.append(
            "superseded facts (history, no longer current): "
            + ", ".join(f.value for f in superseded_facts)
        )

    if knowledge.contradicted_notes:
        lines.append(f"contradictions noted during conversation: {knowledge.contradicted_notes}")
    if knowledge.unresolved_items:
        lines.append(f"still unresolved as of latest message: {knowledge.unresolved_items}")

    return "\n".join(lines)


@dataclass
class ReconciledFact:
    """One fact as it stands after reconciliation -- see reconcile_known_facts."""
    value: str
    certainty: str = "unknown"
    status: str = "current"  # "current" | "superseded" -- see reconcile_known_facts
    evidence: Optional[str] = None
    source_message_id: Optional[int] = None
    observed_at: Optional[str] = None


@dataclass
class ReconciledKnowledge:
    """
    The canonical reconciled conversation state (spec section 12, "Stage
    D -- State Reconciliation" / section 15, "ConversationState"): the
    system's current understanding, with full history retained
    separately rather than overwritten. See reconcile_known_facts.
    """
    current_solution: Optional[ReconciledFact] = None
    # Prior current_solution values, oldest first, kept for auditability
    # even though only the newest is "current" -- current_solution is
    # inherently single-valued, so any new statement of it always
    # supersedes the previous one (spec section 13's exact example).
    current_solution_history: list[ReconciledFact] = field(default_factory=list)
    # ALL generic facts ever extracted, chronological, each tagged
    # current/superseded in place -- never deleted, matching section 13:
    # "the old evidence should not be erased from history."
    facts: list[ReconciledFact] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)
    contradicted_notes: list[str] = field(default_factory=list)


def reconcile_known_facts(contact: Contact) -> ReconciledKnowledge:
    """
    Stage D -- State Reconciliation. Walks every classified inbound
    message's stored semantic_analysis, in chronological order, and
    produces one canonical ReconciledKnowledge: not "every extraction is
    a new permanent fact" (the bug spec section 12 describes), but a
    reconciled view where a later statement can supersede an earlier one
    while the earlier one is kept, marked superseded, rather than
    deleted.

    How supersession is decided (deliberately narrow, matching this
    codebase's existing anti-heuristic principle -- see the module
    docstring and the removed dict-based predecessor of this function):
    - current_solution is single-valued by definition. Any new
      current_solution statement always supersedes whatever
      current_solution was current before it -- no guessing required,
      the shape of the data settles it.
    - Generic facts (new_facts) are NOT assumed to supersede each other
      just because they arrived later -- two different facts can both be
      true at once (e.g. "team is 50 people" and "they use Salesforce"
      are unrelated). A fact only marks an earlier one superseded when
      the LLM explicitly said so via `supersedes` (see
      semantic/classifier.py's SUPERSESSION rule), naming the prior
      fact's exact value. This is an exact string match against a value
      the LLM was instructed to copy verbatim from conversation history,
      not a fuzzy/semantic match computed here in deterministic code --
      recognizing that two differently-worded statements mean the same
      thing is exactly the kind of judgment this architecture reserves
      for the LLM (see llm/grounding.py and memory/store.py's own
      pre-existing "anything cleverer would drift toward guessing
      semantic equivalence outside the LLM" principle).
    - An exact-duplicate restatement of an already-current fact (same
      value, no supersedes) does not add a second line -- it's the same
      fact being reaffirmed, not new information.

    Provenance (source_message_id/observed_at) rides along on every
    ReconciledFact -- see semantic_models.attach_fact_provenance for
    where it's stamped, and semantic_models.SemanticFact's docstring for
    why it's never LLM-provided.
    """
    inbound = [m for m in contact.messages if m.direction == "inbound" and m.semantic_analysis]
    inbound = sorted(inbound, key=lambda m: (m.created_at is None, m.created_at, m.id or 0))

    def _mk(raw: dict) -> ReconciledFact:
        return ReconciledFact(
            value=raw.get("value", ""),
            certainty=raw.get("certainty") or "unknown",
            status="current",
            evidence=raw.get("evidence"),
            source_message_id=raw.get("source_message_id"),
            observed_at=raw.get("observed_at"),
        )

    current_solution: Optional[ReconciledFact] = None
    current_solution_history: list[ReconciledFact] = []
    facts: list[ReconciledFact] = []
    unresolved: list[str] = []
    contradicted: list[str] = []

    for m in inbound:
        analysis = m.semantic_analysis
        if not isinstance(analysis, dict):
            continue

        cs = analysis.get("current_solution")
        if cs and cs.get("value"):
            if current_solution is not None:
                current_solution.status = "superseded"
                current_solution_history.append(current_solution)
            current_solution = _mk(cs)

        for raw_fact in analysis.get("new_facts") or []:
            if not raw_fact or not raw_fact.get("value"):
                continue
            value = raw_fact["value"]
            supersedes = raw_fact.get("supersedes")
            if supersedes:
                for existing in facts:
                    if existing.status == "current" and existing.value == supersedes:
                        existing.status = "superseded"
            already_current = any(
                existing.status == "current" and existing.value == value for existing in facts
            )
            if not already_current:
                facts.append(_mk(raw_fact))

        if analysis.get("unresolved_items"):
            unresolved = analysis["unresolved_items"]  # only the latest turn's still-open items matter
        if analysis.get("contradicted_facts"):
            contradicted.extend(analysis["contradicted_facts"])

    return ReconciledKnowledge(
        current_solution=current_solution,
        current_solution_history=current_solution_history,
        facts=facts,
        unresolved_items=unresolved,
        contradicted_notes=contradicted,
    )


def maybe_summarize_older_messages(db: Session, contact: Contact) -> None:
    """
    Called after logging a new message. If the thread has grown past the
    verbatim window, roll everything older than the window into
    `contact.memory_summary` via a cheap LLM call. Best-effort: if the
    LLM is unavailable, the transcript just stays longer (still correct,
    just not compacted) rather than failing the send.

    Same provenance filter as build_conversation_context (see
    _conversational_evidence) -- applied here too, and for the same
    reason: without it, a DRAFT that later gets excluded from the recent
    verbatim window (once enough newer messages push it out) would still
    have been eligible to enter the rolling summary itself, poisoning
    grounding indirectly through the summary rather than the recent
    window. Both entry points need the same filter; this file has
    exactly one place that defines it now.
    """
    messages: list[Message] = _conversational_evidence(list(contact.messages))
    if len(messages) <= SUMMARIZE_THRESHOLD:
        return

    to_summarize = messages[: -RECENT_MESSAGES_VERBATIM]
    if not to_summarize:
        return

    transcript = "\n\n".join(
        f"[{m.direction}] {m.subject or ''}\n{m.body}" for m in to_summarize
    )
    prior_summary = f"Prior summary: {contact.memory_summary}\n\n" if contact.memory_summary else ""

    system_prompt = (
        "You maintain a running summary of a sales email conversation for "
        "an SDR's own reference. Summarize only what was actually said -- "
        "key facts stated, objections raised, questions asked, commitments "
        "made by either side. Do not invent anything. 4-6 sentences max."
    )
    human_prompt = f"{prior_summary}New messages to fold in:\n\n{transcript}"

    try:
        result = call_llm_text(system_prompt, human_prompt, temperature=0.1, max_tokens=220, operation="memory_summary")
        contact.memory_summary = result.text
        db.add(contact)
        db.flush()
    except (LLMUnavailableError, LLMOutputError) as e:
        logger.info("Skipping memory summarization for contact %s: %s", contact.id, e)
