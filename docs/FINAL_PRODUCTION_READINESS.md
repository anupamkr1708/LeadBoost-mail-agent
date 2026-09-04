# Mailer Agent — Production Readiness Report

Status legend: **VERIFIED** (executed and observed) / **IMPLEMENTED** (code
exists, traced by hand, not executed) / **NOT VERIFIED** (genuinely
unknown) / **PARTIAL** / **DEFERRED**.

This is the one production-readiness document for this repository,
covering three work sessions. It replaces its own prior version in
place. §1-2 summarize sessions 1-2 (hardening pass). §3 onward covers
this session (Phase 2: semantic architecture). §4 is the honest
cross-session status table. §5-9 are the closing sections.

## The rule this report follows, unchanged across all three sessions

> Code inspection ≠ Runtime verification. Unit test ≠ Concurrency
> verification. Live LLM smoke test ≠ Deterministic regression test.
> Draft grounding ≠ Final send safety. Scheduler exists ≠ Distributed
> scheduling correctness.

**No pytest run occurred in any of the three sessions.** This sandbox
has never had network access to install `fastapi`, `sqlalchemy`,
`pydantic`, `groq`, `apscemuler`, `tenacity`, `psycopg2`. One genuine
exception this session, called out specifically in §3.9: the
architecture-boundary tests were actually executed (not just traced),
because they only need Python's stdlib `ast` module. Everything else
below follows the same standard as the prior two sessions: traced by
hand against the real code, never executed, and reported as such.

---

## 1-2. Sessions 1-2 summary (hardening pass, condensed)

Full detail was in prior versions of this file. Condensed here so this
stays the only document.

**Session 1**: implemented the approval-time grounding recheck (the
single most serious gap from the originating audit — approval only
checked suppression, never re-validated grounding); implemented
`MessageStatus.UNKNOWN` end-to-end for ambiguous SMTP outcomes; fixed a
cross-tenant data leak in `/system/*`; removed dead/broken code
(`integration.py`'s unreachable `get_system_health()`, `llm/agent.py`'s
zero-caller `classify_reply()`); found and fixed a test-isolation bug
where `llm/provider.py` bound functions at import time, defeating test
patches; rewrote the keyword-guessing `FakeLLMProvider` into a
deterministic scenario-queue fixture provider.

**Session 2**: fixed the inbound-correlation ambiguous-match guard (used
to silently guess across tenants; now refuses and logs instead);
normalized Message-ID formatting across webhook/IMAP so the same email
delivered via both channels dedupes correctly; found and fixed an
authentication-retry bug (401/403 was retried 3x due to a type-hierarchy
mismatch with tenacity's retry matching); found and fixed a suppression
gap in the auto-reply path; added a deterministic Python E2E suite
(previously didn't exist at all) and a context-contamination test across
interleaved campaigns.

---

## 3. This session: Phase 2 semantic architecture

### 3.1 Forensic audit finding: the planner was orphaned architecture

Before writing any new code, this session searched for every reference
to `NextBestActionPolicy` (the pre-existing 428-line `policy/next_action.py`).
**It had exactly one reference anywhere in the codebase**: a dead
import-check inside `integration.py`'s health-check function
(`from ... import NextBestActionPolicy  # noqa: F401`), never
instantiated, never called. The actual production reply-handling logic
lived entirely inline in `mail/reply_handler_v2.py` as a single
hardcoded line (`action_type = "closing" if POSITIVE_INTEREST and
confidence >= 0.7 else "reply"`) plus a fixed `ALWAYS_REQUIRE_APPROVAL_INTENTS`
set-membership check — exactly the "priority chain" anti-pattern the
spec asked to remove, just smaller than the orphaned module's version of
it.

This is the same failure mode named explicitly in this phase's
instructions (a subsystem that exists, is internally coherent, and is
completely disconnected from the real system). It reframed the whole
session's priority: rewriting `policy/next_action.py` in isolation,
without also replacing the inline logic that was *actually* running,
would have just created a second orphan. Every new module below has a
confirmed production caller — see §3.9 for how that was checked.

### 3.2 Semantic contracts — rewritten (`semantic_models.py`)

- `Certainty` (explicit / strongly_inferred / weakly_inferred / unknown)
  now travels with individual facts (`SemanticFact`), not just the
  overall classification confidence — so "the prospect said X" and "the
  model concluded Y from context" are represented differently rather
  than collapsed into equally-certain plain strings.
- `requested_timing: str` replaced with a structured `TimingSignal`
  (kind, normalized_target, certainty, requires_clarification). A
  backward-compat `.requested_timing` property remains for anything that
  only needs the raw phrase.
- Added `speech_act`, `user_goal`, `pain_points`, `constraints`,
  `procurement_signal`, `commitments_made`, `current_solution`,
  `competitors_mentioned`, `new_facts`, `contradicted_facts`,
  `unresolved_items`, `uncertain_aspects` — grouped by the same
  categories the classifier prompt reasons about (communicative meaning
  / prospect state / commercial signals / conversation content /
  knowledge / temporal / uncertainty), not a flat bag of fields.
- `GroundingValidation`'s docstring now says explicitly what it is —
  "claim-risk validation, not fact verification" — and names its actual
  limitation (paraphrased claims with different wording may not be
  caught).
- Added `serialize_semantic_intent()` / `SEMANTIC_SCHEMA_VERSION`,
  fixing a real bug: `semantic_analysis` (a native `Column(JSON)`) was
  being written via `json.dumps(...)`, double-encoding it into a JSON
  string containing a JSON string. Confirmed by grep that nothing else
  ever read the column, so this had zero observed impact but would have
  broken on first read.

### 3.3 Semantic interpreter — rewritten (`semantic/classifier.py`)

- Removed the out-of-office keyword phrase list entirely (`"out of
  office"`, `"on vacation"`, `"auto-reply"`, etc.). Replaced with a
  genuinely deterministic check: the RFC 3834 `Auto-Submitted` email
  header, captured in `mail/imap_reader.py` from real message headers
  and optionally accepted in the webhook payload. When that signal isn't
  available, OOO now goes through real LLM semantic classification
  (`IntentType.OUT_OF_OFFICE` is part of the normal vocabulary) instead
  of a hardcoded phrase list.
- Unsubscribe keyword matching is unchanged and stays — the one
  explicitly-permitted safety exception (§3 of this phase's
  instructions), documented as such in the module docstring, not
  confused with semantic interpretation.
- Prompt and parsing rewritten around the new schema; every sub-parser
  is defensive (malformed nested objects degrade to `None`/`UNKNOWN`
  rather than crashing the whole classification).

### 3.4 Two more concrete defects found and fixed while tracing production behavior

- `state_machine.py`'s `infer_event_from_semantic_intent` had the exact
  compound inference named in this phase's instructions as an example
  of what to remove: `positive_interest + buying_stage in (evaluating,
  deciding) -> PRICING_DISCUSSED`. Removed. Every branch is now a direct
  single-signal mapping (the prospect explicitly asked about pricing ->
  that event; positive interest alone -> that event, full stop,
  regardless of buying stage).
- `followup/conversation_aware.py`'s `_extract_requested_timing` used to
  regex-match the old free-text timing string against a hardcoded phrase
  list and **default to 30 days when nothing matched** — inventing a
  business-meaningful date from nothing. Now consumes the LLM's
  structured `normalized_target` directly; unresolvable/vague timing
  returns `None` (falls through to the campaign's own configured
  cadence) instead of a fabricated date. Verified this doesn't regress
  the legitimate, campaign-configured cadence math elsewhere in the same
  file (`campaign.follow_up_days[...]`) — see §3.11's heuristic audit.

### 3.5 The critical fallback fix (`llm/agent.py`)

The deterministic fallback used for `action_type="reply"/"closing"` when
the LLM was unavailable generated a **generic "let's hop on a call"
reply regardless of what the prospect actually said** — for objections,
pricing questions, referrals, anything. The code's own comment even
already said "can't safely auto-generate contextual reply content"
immediately above the branch that did exactly that anyway.

Fixed: `draft_message()` now raises `ContextualFallbackUnavailable` for
these action types instead of returning a draft. No content is
invented. `mail/reply_handler_v2.py` catches this and routes the contact
to `NEEDS_REVIEW` with an honest `approval_reason`, rather than
persisting a generic, not-actually-responsive draft a reviewer might
approve without noticing it doesn't address what was asked. Fallback for
`initial_outreach`/`follow_up` (agent-initiated, no specific prospect
statement to get wrong) is unchanged and still safe to auto-generate.

### 3.6 The Planner (`policy/next_action.py`, rewritten)

Replaced the orphaned priority-chain policy engine with an LLM-backed
planner: `plan_next_action(intent, context_transcript, known_facts) ->
NextActionProposal`. Deliberately a second, structurally separate
reasoning step from classification (interpreter answers "what did they
mean?"; planner answers "what should we do about it?"), not a bigger
Python if/elif tree, per this phase's explicit instruction that planning
must be context-dependent, not a fixed chain. `ActionType` is a bounded
vocabulary (acknowledge / answer / clarify / ask_targeted_question /
address_objection / provide_requested_information / propose_next_step /
request_missing_information / defer / nurture / escalate). Falls back to
a conservative `ESCALATE` proposal (never a guessed action) when the LLM
is unavailable or returns something malformed — the one place in this
codebase where "no safe fallback" has a real fallback value, because
"have a human look at it" is itself always a valid plan, unlike invented
reply content (contrast with §3.5).

### 3.7 Guardrails (`policy/guardrails.py`, new file)

Pure deterministic authorization of the planner's proposal — no LLM
call, verified by the architecture tests (§3.9) to have zero import from
`mailer_agent.llm`. Direct, action-driven replacement for the old
`ALWAYS_REQUIRE_APPROVAL_INTENTS` set: `ESCALATE` and
`ADDRESS_OBJECTION` are never auto-sendable; pricing questions are never
auto-answered; both classifier and planner confidence must clear 0.6;
either one flagging `requires_human_review` blocks auto-send. Explicitly
does NOT duplicate suppression or grounding checks — those remain
exactly where they already were (independent, unconditional gates
checked immediately before the actual send, regardless of what
guardrails decided), so there's one place each of those lives rather
than two that could drift apart.

### 3.8 Responder changes (`llm/agent.py`, `llm/prompts.py`)

`build_action_instruction` now accepts the planner's `objective`/`reason`
and, when present, grounds the Responder's prompt in the Planner's
specific decision rather than a generic "reply to what they said"
instruction — the PLAN/WORDING separation this phase's instructions
describe as the correct division of responsibility.

### 3.9 Production wiring — the part that makes §3.1's finding not repeat itself

`mail/reply_handler_v2.py::_draft_and_maybe_send_reply` was rewritten
around the real pipeline: classify (already existed) -> **plan** (new)
-> **authorize** (new) -> draft, grounded in the plan's objective ->
independent suppression/grounding gates (unchanged, still checked
immediately before send) -> send or hold. `ALWAYS_REQUIRE_APPROVAL_INTENTS`,
`_can_auto_send_reply`, and `_get_approval_reason` are deleted, fully
superseded.

**How "has a production caller" was actually checked, not just
asserted**: grepped for every call site of `plan_next_action` and
`authorize_action` and confirmed both are called from
`mail/reply_handler_v2.py`, the same file that api/webhooks.py and the
IMAP poll job both funnel into. This is the same verification method
that caught §3.1's finding in the first place, applied to the new code
before calling it done.

Full pipeline for one inbound reply, now: classify -> plan -> authorize
-> draft (grounded in plan) -> grounding validation -> suppression check
-> send or hold. **Three LLM calls per non-auto-sent reply now, not
two** (classify, plan, draft) — this changed the fixture-queuing
requirements for every existing test that exercises a reply; all were
updated (see §3.10) and traced by hand to confirm the new call count.

### 3.10 Memory, observability, replay

- `memory/store.py`: added `build_known_facts_context(contact)` —
  accumulates provenance-tracked facts across the WHOLE conversation
  (from every prior inbound message's stored `semantic_analysis`), not
  just the latest turn. This is the "working memory" layer between
  recent-messages-verbatim and a pure prose summary. Simple
  last-write-wins by fact text, deliberately not fuzzy deduplication
  (anything cleverer would mean guessing semantic equivalence in
  deterministic code, which is exactly what this architecture avoids).
  Timing-checked by hand: the accumulated context correctly includes the
  current turn's just-classified facts, because `db.flush()` on the
  inbound message happens before `contact.messages` is first read in
  this request, so the ORM relationship's first (lazy) load already
  includes it.
- `observability.py` (new): `TurnTrace` dataclass + `build_and_log_turn_trace()`
  — one structured log entry per reply turn (contact/campaign/message
  IDs, classification source, full semantic intent, planner
  proposal+confidence+source, guardrail decision, draft source, grounding
  result, final action, prompt versions). Not event sourcing, not a new
  table — a logged snapshot, shaped so it could be replayed later (same
  inputs, fake-provider fixtures reconstructed from the recorded LLM
  outputs) without needing a live model. Wired into
  `_draft_and_maybe_send_reply`; never raises (observability must not be
  able to break the pipeline it's observing).

### 3.11 Heuristic audit (spec §42) — actually run, results below

Ran `grep -rn "if .* in .*\.lower()\|any(.* in "` and
`grep -rn "timedelta(days="` across `mailer_agent/`. Every result,
classified:

| Location | Classification | Why |
|---|---|---|
| `semantic/classifier.py:336` (unsubscribe pattern match) | **SAFE DETERMINISTIC POLICY** | The one explicitly-permitted safety exception; never used for anything beyond "should we ever miss an opt-out." |
| `conversation_aware.py:73,147,173` (`timedelta(days=days_to_wait/nurture_days/delay_days)`) | **SAFE DETERMINISTIC POLICY** | All three read from `campaign.follow_up_days` (campaign-configurable); the 30-day/7-day values are fallbacks for a campaign with *no* cadence configured at all, not overrides of a parsed semantic timing signal (that fallback was the one removed in §3.4). |
| `utils/datetime_utils.py:134,138` (`timedelta(days=1)`) | **SAFE DETERMINISTIC POLICY** | Pure business-hours/weekday calendar rollforward, no sales semantics involved. |

No semantic heuristics found remaining outside the documented exception.

### 3.12 PostgreSQL concurrency — real gap found and fixed, tests written but NOT VERIFIED

While designing the concurrency test suite (not while running it — this
was found by tracing, same as everything else), discovered **`Message`
had no database-level unique constraint on `message_id_header`** — the
inbound-dedup check in `process_inbound_email_v2` was check-then-insert
with no DB-level backing, a real TOCTOU race under true concurrency
(e.g. the same email arriving via webhook and IMAP nearly
simultaneously). Fixed:
- Added `UniqueConstraint("message_id_header")` to `Message` (NULLs
  unconstrained; only non-NULL collisions rejected).
- New migration, `003_message_id_unique_constraint.py` (checks for
  existing duplicates first and refuses to apply rather than failing
  mid-`ALTER` if any are found — a data decision, not a schema one).
- `process_inbound_email_v2` now catches the resulting `IntegrityError`
  on a genuine race and treats it as a graceful duplicate (rollback,
  re-query, return the same `"skipped_duplicate"` result the normal
  check-then-insert path returns), instead of surfacing an unhandled
  500.

A second gap found the same way: `api/messages.py`'s
`approve_and_send_draft` had no row lock, so two truly simultaneous
approval requests for the same draft could both pass the DRAFT-status
check before either committed. Fixed with `.with_for_update()` on the
message fetch (compiles to a no-op on SQLite, which doesn't support the
clause — this is standard, well-documented SQLAlchemy cross-dialect
behavior, though not independently re-verified against the exact pinned
SQLAlchemy version in this sandbox).

`tests/test_postgresql_concurrency.py` (new): dual-worker claim race,
lease-expiry recovery, concurrent duplicate-Message-ID insert (proving
the new constraint), concurrent approval (proving the new row lock) —
all written against the real `followup/work_claiming.py` and
`api/messages.py` functions directly, using real threads with separate
sessions and a `threading.Barrier` to maximize genuine transaction
overlap. **Skips cleanly (not a false pass) when `POSTGRES_TEST_URL`
isn't set or unreachable — never executed in this sandbox, no
PostgreSQL instance available.** This is the single largest remaining
unverified area across all three sessions.

### 3.13 Evaluation corpus and live-LLM suite

- `tests/evaluation/test_semantic_evaluation.py` (new): dimension-by-
  dimension pipeline tests (prospect-goal reaching the planner,
  objection handling, explicit-vs-inferred fact provenance, vague-timing
  non-fabrication, state-machine direct-mapping regression, low-
  confidence auto-send blocking, grounding, and a "semantic
  generalization" test using three *different* structured
  interpretations to prove the pipeline handles equivalent meaning
  consistently). Stated honestly in its own docstring: this proves the
  pipeline's handling of a *given* interpretation, not that a real LLM
  produces good interpretations — it uses the fake provider throughout.
- `tests/evaluation/test_live_llm_semantic_quality.py` (new,
  `@pytest.mark.integration`): the genuine semantic-quality claims —
  differently-worded-but-equivalent emails converging on the same
  `current_solution`, the same surface word ("expensive") meaning
  different things in different contexts, planner objective relevance.
  Requires `GROQ_API_KEY` and network. **Never executed — no network
  access in this sandbox.** Its own docstring says so explicitly and
  warns against treating the file's existence as evidence of quality.

### 3.14 Architecture tests — the one thing genuinely executed this session

`tests/test_architecture.py` (new): parses every production file's
import statements with Python's `ast` module (not runtime imports) to
check layer boundaries — SMTP/IMAP confined to `mail/`, the vendor Groq
SDK confined to `llm/provider_v2.py`, `semantic/` never imports mail-
sending, `policy/` never imports mail-sending or the DB session,
`guardrails.py` has zero LLM dependency, and `guardrails.authorize_action`'s
signature is typed against the `NextActionProposal` dataclass (not
`dict`/`Any`) so LLM output can't flow into policy logic unchecked.

**Unlike everything else in this report, the import-boundary checks
(everything except the last, typed-signature one, which needs the real
modules importable) were actually run in this sandbox** — they only need
`ast` and `pathlib`, both stdlib, so no missing dependency blocked
execution. Ran directly (not through the pytest file, since `pytest`
itself isn't installed here either) against all 35 production files:
**zero violations, confirmed by execution, not tracing.** This is called
out specifically because it's the one VERIFIED claim in an otherwise
all-IMPLEMENTED-not-VERIFIED report, and that distinction matters enough
to not blur it.

---

## 4. Cross-session verification status

| Area | Status | Session |
|---|---|---|
| Approval-time grounding recheck | IMPLEMENTED, not run | 1 |
| `MessageStatus.UNKNOWN` wired through | IMPLEMENTED, not run | 1 |
| `/system/*` tenant scoping | IMPLEMENTED, not run | 1 |
| Tenant-safe inbound correlation | IMPLEMENTED, not run | 1+2 |
| Webhook/IMAP Message-ID dedup format | IMPLEMENTED, not run | 2 |
| Provider retry/auth-not-retried | IMPLEMENTED, not run | 2 |
| Deterministic Python E2E suite | IMPLEMENTED, not run | 2 |
| Semantic contracts (rich model, provenance) | IMPLEMENTED, not run | 3 |
| Semantic interpreter (metadata-first OOO) | IMPLEMENTED, not run | 3 |
| State machine semantic-assumption removal | IMPLEMENTED, not run | 3 |
| Structured timing (no fabricated dates) | IMPLEMENTED, not run | 3 |
| Contextual-reply fallback safety | IMPLEMENTED, not run | 3 |
| Planner + Guardrails, wired to production | IMPLEMENTED, not run | 3 |
| Memory layering (accumulated facts) | IMPLEMENTED, not run | 3 |
| Observability/replay trace | IMPLEMENTED, not run | 3 |
| Import-boundary architecture tests | **VERIFIED** (executed) | 3 |
| Heuristic audit | **VERIFIED** (executed, grep) | 3 |
| `message_id_header` unique constraint + race handling | IMPLEMENTED, not run | 3 |
| Concurrent-approval row lock | IMPLEMENTED, not run | 3 |
| PostgreSQL concurrency suite | **NOT VERIFIED / NO PG INSTANCE** | 3 (written) |
| Live-LLM semantic quality suite | **NOT VERIFIED / NO NETWORK** | 3 (written) |
| Semantic evaluation corpus (fake-LLM) | IMPLEMENTED, not run | 3 |

---

## 5. Remaining limitations, stated plainly

- **Nothing in this repository has been executed by pytest, ever,
  across three sessions.** This is the standing, primary limitation.
  Everything marked IMPLEMENTED is a hand-traced correctness claim, not
  a proof.
- PostgreSQL concurrency is entirely unverified. The tests are written
  against real production functions with real threading and a real
  unique constraint this session added — but "written correctly by
  tracing" and "verified" are different claims, and concurrency bugs are
  specifically the kind that tracing is worst at catching.
- Live-LLM semantic quality (does the model actually understand
  paraphrases, does the planner produce genuinely good objectives) is
  entirely unverified. The harness exists; the measurement doesn't yet.
- `.with_for_update()`'s no-op behavior on SQLite is based on documented
  SQLAlchemy cross-dialect design, not independently re-verified against
  the exact pinned version in a running environment.
- The three-LLM-calls-per-reply change (classify, plan, draft) increases
  latency and API cost per reply versus the two-call design it replaced.
  Not benchmarked (would need a live suite run, see above).
- No campaign-level "business objective" field exists yet; the planner
  uses one fixed, honest default objective for all campaigns. If
  campaigns ever need genuinely different objectives (book a meeting vs.
  collect a referral vs. something else), `policy/next_action.py`'s
  `DEFAULT_BUSINESS_OBJECTIVE` is the one place that would need to
  become campaign-aware — flagged rather than built speculatively, since
  nothing currently requires it.

## 6. Exact commands to run locally

```powershell
.\venv\Scripts\Activate.ps1
python -m compileall -q mailer_agent

# Full deterministic suite
python -m pytest -q

# New this session specifically
python -m pytest tests/test_architecture.py -v
python -m pytest tests/test_planner_and_guardrails.py -v
python -m pytest tests/evaluation/test_semantic_evaluation.py -v
python -m pytest tests/e2e/test_full_lifecycle.py -v

# PostgreSQL concurrency -- requires a real instance
$env:POSTGRES_TEST_URL = "postgresql://user:pass@localhost:5432/mailer_agent_test"
python -m pytest tests/test_postgresql_concurrency.py -v -m postgres

# Live LLM semantic quality -- requires network + a real key
$env:GROQ_API_KEY = "<your key>"
python -m pytest tests/evaluation/test_live_llm_semantic_quality.py -v -m integration

# Live SMTP/IMAP -- per spec, never faked; run separately and locally
$env:LIVE_SENDING_ENABLED = "true"
$env:TEST_RECIPIENT_EMAIL = "<a mailbox you control>"
```

If anything fails: investigate and report the exact assertion and
expected-vs-actual. Per this repository's standing rule (and this
phase's explicit §36), do not delete, skip, weaken, or rename a failing
test to make it pass — a failure in newly-written code is real
information about this session's work, not noise to route around.

## 7. Final readiness

**NOT READY** for `LIVE_SENDING_ENABLED=true` against real prospects
without running the commands in §6 and reporting real results —
unchanged from prior sessions' assessment, and if anything the bar is
higher now given how much new logic (planner, guardrails, memory
accumulation) sits on the critical path with zero executions behind it.

**STAGING READY** for `LIVE_SENDING_ENABLED=false` (dry-run) use once
§6's deterministic suite is confirmed passing.

Not PRODUCTION READY: PostgreSQL concurrency unverified, live-provider
semantic quality unverified (correctly, per instruction not to fake it),
and — the honest addition this session makes to that list — the
production reply path itself changed substantially (new 3-call pipeline,
new fallback behavior, new memory accumulation) and none of it has run
yet.

## 8. Exact next step

Run §6's commands. In priority order for what to fix first if failures
appear: (1) anything in `tests/test_planner_and_guardrails.py` or
`tests/e2e/test_full_lifecycle.py`, since those exercise the actual
production wiring this session changed; (2) `tests/test_architecture.py`
and the rest of the deterministic suite; (3) stand up a real PostgreSQL
test database and run the concurrency suite — this is the biggest
verification gap across all three sessions and the one most likely to
reveal something tracing missed; (4) only after 1-3 are clean, spend
real API budget on the live-LLM suite.
