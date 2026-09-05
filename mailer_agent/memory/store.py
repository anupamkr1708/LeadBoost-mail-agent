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
    Accumulated "working memory" of provenance-tracked facts across the
    WHOLE conversation, not just the latest message -- pulled from each
    inbound Message's stored semantic_analysis (see
    semantic_models.serialize_semantic_intent / mail/reply_handler_v2.py,
    which persists it as a native JSON dict on every classified inbound
    message).

    This is the layer between "recent messages verbatim"
    (build_conversation_context, above) and "durable structured facts" --
    it's what lets the planner (policy/next_action.py) know that a fact
    established three messages ago is still current, rather than only
    seeing whatever the very latest message happened to restate.

    Later facts about the same thing supersede earlier ones (a fact
    marked contradicted, or a current_solution mentioned again with a
    different value) -- this performs simple last-write-wins by fact
    text, not fuzzy deduplication, since anything cleverer would drift
    toward guessing semantic equivalence outside the LLM, which is
    exactly what this architecture avoids doing in deterministic code.
    """
    inbound = [m for m in contact.messages if m.direction == "inbound" and m.semantic_analysis]
    if not inbound:
        return ""

    current_solution = None
    facts_seen: dict[str, str] = {}  # value -> certainty, insertion order = recency
    unresolved: list[str] = []
    contradicted: list[str] = []

    for m in inbound:
        analysis = m.semantic_analysis
        if not isinstance(analysis, dict):
            continue
        cs = analysis.get("current_solution")
        if cs and cs.get("value"):
            current_solution = cs  # later messages win
        for fact in analysis.get("new_facts") or []:
            if fact and fact.get("value"):
                facts_seen[fact["value"]] = fact.get("certainty", "unknown")
        if analysis.get("unresolved_items"):
            unresolved = analysis["unresolved_items"]  # only the latest turn's still-open items matter
        if analysis.get("contradicted_facts"):
            contradicted.extend(analysis["contradicted_facts"])

    lines = []
    if current_solution:
        lines.append(f"current_solution: {current_solution['value']} (certainty={current_solution.get('certainty', 'unknown')})")
    for value, certainty in facts_seen.items():
        lines.append(f"fact: {value} (certainty={certainty})")
    if contradicted:
        lines.append(f"contradictions noted during conversation: {contradicted}")
    if unresolved:
        lines.append(f"still unresolved as of latest message: {unresolved}")

    return "\n".join(lines)


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
        summary = call_llm_text(system_prompt, human_prompt, temperature=0.1, max_tokens=220)
        contact.memory_summary = summary
        db.add(contact)
        db.flush()
    except (LLMUnavailableError, LLMOutputError) as e:
        logger.info("Skipping memory summarization for contact %s: %s", contact.id, e)
