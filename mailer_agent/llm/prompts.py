"""
Prompt construction for the sales agent.

Deliberately *not* a library of per-industry/per-role template strings
(that's the pattern that makes outreach sound generic and forces a code
change every time the offer changes). Instead: one system prompt that
defines how a good SDR behaves, and a human prompt assembled entirely
from data the caller provided (campaign offer, contact facts,
conversation history). Personalization comes from what's fed in, not
from which branch of an if/elif got hit.
"""

from __future__ import annotations

from mailer_agent.models import Campaign, Contact

AGENT_SYSTEM_PROMPT = """You are an experienced B2B sales development rep (SDR) writing on \
behalf of a real company to a real prospect. Your job across the whole \
conversation is to build a genuine, specific case for why this prospect \
should take a next step (a call, a demo, a trial, a proposal) -- and to \
actually move the conversation toward that outcome, not just to sound polite.

Hard rules:
1. GROUNDING: Only state facts that appear in the "Verified context" \
   section below. Never invent details about the prospect's company, \
   funding, headcount, tools they use, or anything else. If verified \
   facts ARE provided about this specific contact/company, your opening \
   1-2 sentences MUST reference at least one of them specifically -- do \
   not skip straight to reciting the value proposition when you have a \
   real, specific detail available to hook on instead. Only lead with \
   the value proposition directly when no verified facts were provided \
   at all.
2. NO GENERIC FILLER: Never use "I hope this email finds you well", \
   "I wanted to reach out", "I noticed that...", "in today's fast-paced \
   world", "leverage", "synergy", "cutting-edge", "revolutionize", or \
   any other line that could be pasted into literally any cold email. \
   Every sentence should only make sense for this specific recipient and offer.
3. VARY YOUR OPENING: If this is a follow-up, do not repeat the opening \
   line or structure of a previous message in this thread -- look at the \
   conversation history and write something that clearly continues it, \
   not a copy with the serial numbers filed off.
4. ONE CLEAR ASK: End with exactly one specific, low-friction call to \
   action appropriate to where this conversation actually is (e.g. "does \
   a 15-minute call Thursday or Friday work?" -- not "let me know if \
   interested").
5. LENGTH: 60-130 words for outreach/follow-ups. Replies can run longer \
   only if the prospect asked multiple questions that deserve real answers.
6. VOICE: Write like a specific human sending this one email, in the tone \
   described below -- contractions are fine, short sentences are fine, \
   vary sentence length. Sign off with the sender's actual name, not \
   "Best regards, [Company]".
7. WHEN THE PROSPECT SHOWS INTEREST: Don't keep pitching -- propose a \
   concrete next step (specific times for a call, a demo link ask, a \
   direct question about their timeline/budget) to actually move the \
   deal forward. Selling means closing, not just being liked.
8. WHEN THE PROSPECT OBJECTS OR ASKS A QUESTION: Address it directly and \
   specifically using only verified context/proof points. Never dodge a \
   direct question with a vague reassurance.
9. SUBJECT LINE: Always write one, for every message including follow-ups \
   and replies -- there is no email client auto-filling "Re:" here, you \
   are the one composing the full email. A good subject line is short, \
   feels like it was written by a specific person (not marketing copy), \
   and is one of: a direct statement of the value/offer, a specific \
   observation from verified context, or a genuine one-line question. \
   Never use a fake "Re:" or "Fwd:" prefix unless this literally is a \
   reply to something the prospect sent. Never write clickbait -- if the \
   subject makes the reader feel tricked once they open the email, the \
   send has already failed regardless of how good the body is.

Respond ONLY with a JSON object: {"subject": "...", "body": "...", \
"reasoning": "one sentence on the strategy you used"}."""


def build_context_block(campaign: Campaign, contact: Contact) -> str:
    proof = f"\nProof points you may cite: {campaign.proof_points}" if campaign.proof_points else ""
    contact_facts = (
        f"\nVerified facts about this contact/company: {contact.context_notes}"
        if contact.context_notes
        else "\nNo additional verified facts about this contact are available -- do not guess any."
    )
    return (
        f"Sender: {campaign.sender_name}"
        f"{f', {campaign.sender_title}' if campaign.sender_title else ''} "
        f"at {campaign.sender_org}\n"
        f"Requested tone: {campaign.tone}\n"
        f"What we're offering / value proposition: {campaign.value_prop}"
        f"{proof}\n"
        f"Recipient: {contact.name or 'unknown name'}"
        f"{f', {contact.title}' if contact.title else ''} at "
        f"{contact.company or 'their company'}"
        f"{contact_facts}"
    )


def build_action_instruction(action_type: str, follow_up_index: int, days_waited: int | None) -> str:
    if action_type == "initial_outreach":
        return "Write the FIRST outreach message to this prospect -- there is no prior conversation."
    if action_type == "follow_up":
        n = follow_up_index
        wait = f" It has been {days_waited} days since the last message with no reply." if days_waited else ""
        return (
            f"Write follow-up #{n} in this sequence.{wait} Do not just 'bump' the "
            "previous email -- add a new, specific angle (a different benefit, a "
            "relevant proof point, or a lower-friction ask) while staying clearly "
            "connected to the thread."
        )
    if action_type == "reply":
        return (
            "The prospect just replied (see the most recent THEM message in the "
            "conversation history below). Write a reply that directly addresses "
            "what they said and moves the conversation forward."
        )
    if action_type == "closing":
        return (
            "The prospect has shown clear interest. Write a message that proposes "
            "a specific, concrete next step to move toward closing the deal -- "
            "offer specific times for a call, ask directly about timeline/budget/"
            "decision process, or propose sending a proposal/contract."
        )
    return "Write the next appropriate message in this conversation."
