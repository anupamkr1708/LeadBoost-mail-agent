# Mailer Agent — Production Readiness Report

Status legend used throughout: **VERIFIED** (executed and observed) /
**IMPLEMENTED** (code exists and was traced by hand, but not executed) /
**NOT VERIFIED** (genuinely unknown) / **PARTIAL** / **DEFERRED**.

This document replaces 13 overlapping prior reports that accumulated in
`docs/` (AUDIT_FINDINGS, CURRENT_STATE_AUDIT, RECONCILIATION_STATUS,
FINAL_RUNTIME_RECONCILIATION, and others). They are deleted, not archived
elsewhere — their claims are superseded by this one, and several of them
directly contradicted each other and this file's findings. Some of their
useful analysis is folded in below with attribution to "prior audit" where
it couldn't be independently re-verified.

## The one rule this report follows

> Code inspection ≠ Runtime verification. Unit test ≠ Concurrency
> verification. Live LLM smoke test ≠ Deterministic regression test.
> Draft grounding ≠ Final send safety. Scheduler exists ≠ Distributed
> scheduling correctness.

Concretely, in this session: **no pytest run occurred.** The sandbox this
work was done in has no network access to install `fastapi`, `sqlalchemy`,
`pydantic`, `groq`, `apscheduler`, or `alembic` (confirmed via both `pip
install` and `uv pip install --offline`, both failing with no cached
wheels available). Every claim below is either (a) traced by hand through
the actual code and cross-checked against the actual model schemas and
regex patterns involved, or (b) explicitly marked NOT VERIFIED. Where a
new test file was added, its logic was traced line-by-line against the
real implementation it targets, but **the test has never actually been
run.** Treat every new test the same way you'd treat a PR you haven't run
CI on yet: probably right, not proven.

**Action required from you:** run the commands in
[How to actually verify this](#how-to-actually-verify-this) below and
replace the NOT VERIFIED items with real results before calling any of
this production ready.

---

## 1. What this repository actually contains right now

This matters because the source prompt this work was based on, and a
follow-up critique of a prior attempt, both reference files that **do not
exist in this repository**: `tests/test_business_context.py`,
`tests/e2e/*.py`, `tests/test_api.py`, `tests/test_concurrency.py`. Their
"46 passed / 8 failed" and "40/54" figures must describe a different,
more complete snapshot than what was actually available to work from here.
No test count from either source document should be trusted for *this*
codebase.

Actual test files present: `tests/test_flow.py`, `tests/test_grounding.py`,
`tests/test_semantic_regression.py`, `tests/test_tenant_correlation.py`
(new), `tests/test_approval_safety_gate.py` (new). No E2E suite, no
PostgreSQL concurrency suite, no live-provider smoke suite exist in this
zip.

---

## 2. Fixes made this session (traced by hand, not executed)

### 2.1 Final outbound safety gate — grounding recheck at approval — **IMPLEMENTED**

This was the single most serious gap identified in the source critique
("Approval path skips revalidation" / "the biggest problem"), and it was
real: `POST /messages/{id}/approve` in `api/messages.py` checked
suppression only, never grounding. A draft could be generated when the
campaign's `proof_points` supported its claims, the campaign could be
edited afterward, and approval would send the now-unsupported draft
without complaint.

Fixed: approval now re-runs `validate_grounding()` against the *current*
campaign/contact/conversation state immediately before sending, and blocks
with `409` (not sent, message stays `DRAFT`) if the draft is no longer
grounded. See `api/messages.py::approve_and_send_draft`.

New test: `tests/test_approval_safety_gate.py`, 6 cases including the
literal scenario the original spec asked for (draft generated while
supported → campaign proof_points edited → draft now unsupported →
approval blocked; and the inverse, an unchanged grounded draft, approval
succeeds). Traced by hand against `llm/grounding.py`'s actual regex
patterns (see that file for the exact matching rules) — not executed.

### 2.2 `MessageStatus.UNKNOWN` — defined but never set — **IMPLEMENTED**

Confirmed by grep: the model defined `UNKNOWN = "unknown"  # provider may
have sent, but we don't know` and nothing in the codebase ever set it.
Every SMTP failure, ambiguous or not, was recorded as `FAILED`, which
downstream code treats as safe-to-retry — risking a duplicate send if the
ambiguous failure actually succeeded server-side.

Fixed in `mail/sender.py`: the `sendmail()` call is now wrapped so that
failures during/after transmission (connection dropped mid-DATA, timeout
after send) raise a distinct `AmbiguousSendError`, mapped to
`SendOutcome.UNKNOWN`. Explicit server refusals (bad recipient, auth
failure) remain `FAILED` and retryable — the distinction is "did the
server tell us no" vs "we lost the connection and don't know." Wired
through `followup/engine_v2.py`, `mail/reply_handler_v2.py`, and
`api/messages.py`: an `UNKNOWN` outcome now routes the contact to
`NEEDS_REVIEW` and clears `next_action_at`, so the scheduler cannot
auto-retry an ambiguous send.

No test added for this (would need a way to fault-inject mid-transmission
smtplib failures, e.g. via a fake SMTP server — not done this session).
**NOT VERIFIED** at the test level; the code path was traced by hand only.

### 2.3 Cross-tenant data leak in `/system/*` — **IMPLEMENTED**

Not mentioned in either source document — found by cross-checking
`api/system.py` against the tenant-scoping pattern used consistently in
`api/campaigns.py` and `api/contacts.py`. `system.py` was the one router
that used `require_api_key` (any valid key) instead of
`get_current_org_id` (this caller's org), so any organization's API key
could read aggregate contact/message/classification counts across *all*
organizations.

Also confirmed the exact anti-patterns the original spec named: `/metrics/
states` issued one `.count()` query per `ContactStatus` enum member
instead of one `GROUP BY`; `/metrics/classification` loaded every inbound
`Message` row into Python to count success/failure in a loop instead of
using SQL aggregation.

Rewritten: every route that returns tenant data now takes
`org_id: str = Depends(get_current_org_id)` and filters through
`Campaign.organization_id == org_id`; state/message/classification counts
are each a single grouped or aggregated SQL query. `/system/config`,
`/system/metrics/llm`, `/system/integration/validate` remain unscoped
deliberately (documented in the file) — they report process-wide
configuration and provider metrics, not tenant data.

**NOT VERIFIED** — no test written this session for the SQL aggregation
correctness; the query logic was traced by hand against SQLAlchemy's
`case()`/`func.sum()` semantics.

### 2.4 Dead and broken code removed — **VERIFIED by static trace**

- `integration.py`'s `MailerAgentCore.get_system_health()` referenced
  `Message` without importing it at module scope — would have raised
  `NameError` on first call. Confirmed via grep that nothing in the
  codebase ever called it. This is direct evidence the "integration
  layer" was never actually exercised end-to-end despite prior reports
  claiming integration was verified. Removed, along with several other
  delegation methods (`send_initial_outreach`, `send_followup`,
  `process_reply`, `classify_reply`, `get_next_best_action`,
  `should_send_followup`, `reschedule_after_reply`) and module-level
  "backward compatible" functions — confirmed via grep that *nothing*
  called through this facade for send/reply logic; the real send path is
  `followup/engine.py → engine_v2.py` and `mail/reply_handler.py →
  reply_handler_v2.py`, called directly by the API routes and scheduler.
  `integration.py` is now ~110 lines: an LLM-metrics/integration-health
  facade only, which is the only part that was actually reachable
  (`api/system.py` is its sole caller).
- `llm/agent.py`'s `classify_reply()` / `ReplyClassification` /
  `VALID_INTENTS` — confirmed zero callers anywhere in the codebase
  (`semantic/classifier.py`'s `classify_prospect_reply` is what's
  actually used for inbound classification; it produces a richer
  multi-dimensional `SemanticIntent`, not a single label). Removed, along
  with the now-unused `REPLY_CLASSIFIER_SYSTEM_PROMPT` /
  `build_classifier_prompt` in `llm/prompts.py`.

### 2.5 A genuine test-isolation bug found while fixing the test infrastructure — **IMPLEMENTED**

`llm/provider.py` (the "thin backward-compatibility wrapper" over
`provider_v2.py`) bound `provider_v2.call_llm_json` / `is_llm_available`
**at import time** (`from provider_v2 import call_llm_json as
_call_llm_json_v2`). Monkeypatching `provider_v2`'s attributes in tests —
which is exactly what the fake-LLM fixture does — never reached
`agent.py`'s `draft_message()`, because `agent.py` imports from
`provider.py`, whose function bodies were closed over the *original*
function objects captured before any test patch ran.

Practical consequence: any test exercising `draft_message()` would
silently attempt a real network call to Groq whenever a real
`GROQ_API_KEY` happened to be present in the environment — the autouse
fake-LLM fixture's protection didn't actually apply to that code path.
Whether this ever caused an actual leaked network call in past CI/dev runs
is unknown; the bug is confirmed by code inspection, not by observing a
leaked call.

Fixed: `provider.py` now delegates via module-attribute access
(`provider_v2.call_llm_json(...)`) at call time, so it genuinely honors
whatever `provider_v2` is patched to. This changed behavior for
`test_grounding.py`'s `TestDraftMessageGrounding` class, whose non-LLM
fallback-path tests had been "passing" by accident (insulated from the
fake-LLM fixture's patches by the very bug just described) rather than by
correctly forcing the fallback path. Fixed by adding an explicit
class-scoped `monkeypatch`-based `is_llm_available=False` fixture to that
test class instead of relying on the module-level `GROQ_API_KEY=""`
environment variable, which was never actually reliable (subject to
import order and settings caching across test files — the kind of
brittleness the source critique's item on test-artifact isolation was
pointing at generally, even though it didn't identify this specific bug).

**NOT VERIFIED by execution** — logic traced by hand through the import
graph; genuinely can't be 100% certain without running it.

### 2.6 Test infrastructure: `FakeLLMProvider` rewritten — **IMPLEMENTED**

This was demanded explicitly and repeatedly by both source documents. The
old fake guessed which canned response to return by substring-matching
keywords ("pricing", "meet", "competitor", ...) against the prompt text —
a second, informal semantic engine living inside the test double,
duplicating (and inevitably drifting from) the real classifier's job.

Rewritten (`tests/fake_llm_provider.py`) as a pure FIFO fixture queue:
`fake_llm.queue_response({...})` before calling code that reaches the
LLM; the fake returns exactly that, in order, and raises a clear
`AssertionError` if called with nothing queued rather than guessing.
Also supports `queue_error(...)` for testing provider-failure handling
(rate limits, malformed output) without a real outage.

`tests/test_semantic_regression.py` was rewritten around this: every test
now explicitly queues the structured JSON a real LLM response would
contain for that scenario, calls `classify_prospect_reply()` for real, and
asserts on the *parsing*, not on invented semantic truth about the English
text. This reframes what these tests actually prove — they are regression
tests for `semantic/classifier.py`'s plumbing (JSON → enums → lists →
`ClassificationResult`), not a substitute for judging whether a real LLM
understands B2B email well (that would need a real, separate,
`@pytest.mark.integration`-marked live-provider suite, which does not
exist in this repository — see §4).

Also fixed along the way, confirmed by checking `semantic_models.py`'s
actual enum values: the old fake's canned fixtures used `"urgency":
"medium"` and `"urgency": "none"` — **neither is a valid `UrgencyLevel`
value** (`immediate` / `near_term` / `long_term` / `no_timeline` are the
only ones). These invalid values were silently swallowed by the
classifier's defensive `try/except ValueError` fallback to
`NO_TIMELINE`, meaning some of the previously-reported "8 failing"
semantic regression tests may well have been failing because of the
fake's own bug, not the production code's. Added
`test_unknown_enum_values_fall_back_to_safe_defaults`, which uses that
exact invalid value on purpose, to make this defensive-parsing guarantee
an explicit regression test rather than an accident nobody noticed.

Also added, because the old file had it as a **stub that asserted
nothing** (a "conceptual test" in comment form): a real
`test_provider_failure_is_not_neutral` that queues an actual
`RateLimitError` via the fake and asserts `classify_prospect_reply`
returns `success=False` with a `failure_reason` — not a silently
downgraded "neutral" result that would hide a provider outage from
monitoring.

**NOT VERIFIED by execution.**

### 2.7 Tenant-safe inbound correlation — new test — **IMPLEMENTED, NOT VERIFIED by execution**

`tests/test_tenant_correlation.py` seeds two organizations whose contacts
share an identical email address (the actual collision scenario), and
calls `mail.reply_handler_v2.process_inbound_email_v2` directly against
an isolated in-memory SQLite session. Confirms: (a) when the inbound
source tells us which inbox received the reply (`to_email`), correlation
resolves to the correct organization even with the colliding contact
email; (b) thread-header correlation (`In-Reply-To`/`References` matching
a prior outbound `Message-ID`) resolves correctly even without
`to_email`, since a `Message-ID` is inherently owned by exactly one
contact/campaign.

This exercises the *existing* implementation in
`mail/reply_handler_v2.py::_find_contact_by_email` /
`_find_contact_by_message_id`, which on inspection already had the
correct join-on-`Campaign.sender_email` logic — this was IMPLEMENTED
correctly already, just untested. The test makes that verifiable rather
than asserted.

One caveat worth flagging, not fixed this session: if a webhook caller
omits `to_email` *and* there's no reusable thread-header match (e.g. the
very first inbound message from a prospect whose email collides across
two orgs, arriving via a webhook that doesn't supply `to_email`),
`_find_contact_by_email`'s fallback searches globally and could
misattribute. This is a real, documented-in-code limitation
(`_find_contact_by_email`'s own docstring calls it "UNSAFE in
multi-tenant production environments"), not a new finding, but it's
worth surfacing here explicitly: **PARTIAL** — safe when `to_email` is
available (which it is for both the webhook path and any well-configured
provider integration), genuinely unsafe in the degenerate case where it
isn't.

### 2.8 Migrations — duplicate confirmed, documented, not deleted — **IMPLEMENTED**

Confirmed via `diff` that `002_add_work_claiming.py` and
`002_work_claiming.py` add identical columns. Per the source spec's own
guidance ("do not blindly delete migration files containing real
historical schema changes"), the duplicate was not deleted — a real
deployment may have already run it. It's marked deprecated in its own
docstring instead, and `migrations/README.md` documents the canonical
order: `001_production_hardening.py` →
`002_constraints_and_multitenancy.py` → `002_work_claiming.py`
(canonical) → `../migrate_db.py`. Alembic (present in requirements.txt
but never wired up — no `alembic/versions/` directory exists) was
deliberately **not** introduced this session: retrofitting Alembic
history onto a database that may have already been migrated by an
unknown subset of these hand-rolled idempotent scripts is a real risk
that shouldn't be taken without the ability to test it, which this
session didn't have.

### 2.9 Housekeeping

Removed from the working tree (already covered by `.gitignore`, but had
been committed before that took effect): `_test.db`, `mailer_agent.db`,
`migration_out.txt`, all `__pycache__/` directories.

---

## 3. What was checked and found to already be correct

These were read in full and traced by hand; no changes were made because
none were needed. Listed because "I checked and it's fine" is different
from "I didn't look," and the difference matters for trust.

- **`followup/work_claiming.py`** — `SELECT ... FOR UPDATE SKIP LOCKED`
  correctly implemented for PostgreSQL, with a documented, deliberate
  SQLite fallback (no true row-locking, acceptable for single-process
  dev use, explicitly not claimed safe for concurrent workers).
  Lease-based claiming (`claimed_by` / `claimed_at`) with expiry.
  **IMPLEMENTED, NOT VERIFIED** — no PostgreSQL available in this
  sandbox to actually run the concurrency scenarios the source spec
  demands (two workers claiming the same contact, concurrent sends,
  duplicate webhooks). This remains the single largest unverified area.
- **`state_machine.py`** — transition table has no per-industry or
  otherwise hardcoded business logic; genuinely generic.
- **`llm/provider_v2.py`** — real error classification (rate limit /
  timeout / malformed output / auth), coordinated single-layer retry
  (not stacked with an outer retry from `provider.py`), in-memory
  metrics.
  **NOT VERIFIED** that its `429`/`5xx`/timeout/malformed-JSON handling
  is exercised anywhere — no dedicated provider-failure test file exists.
- **`llm/grounding.py`** — genuinely business-agnostic regex-based fact
  checking (percentages, dollar figures, headcount claims, guarantee
  language, pricing/availability terms), checked against
  `proof_points` + `context_notes` + `conversation_transcript` +
  `value_prop`, not against any hardcoded industry vocabulary. This
  directly answers the "no hardcoded heuristic logic" ask from the
  request that started this session: the grounding layer already worked
  this way before this session; it was the *approval path* that failed
  to invoke it a second time, which is what §2.1 fixed.
- **`followup/conversation_aware.py`** — cadence reads from
  `campaign.follow_up_days` (JSON list, campaign-configurable), not a
  hardcoded constant. A few natural-language timing fallbacks exist
  ("next month" → 30 days, "next week" → 7 days when the LLM's
  `requested_timing` string can't be parsed more precisely) — these are
  generic calendar semantics, not sales-specific magic numbers, and are
  judged acceptable. `grep -rn "timedelta(days=" mailer_agent/` was run;
  no hardcoded sales-cadence constants (e.g. a bare `timedelta(days=3)`
  used as *the* follow-up interval instead of a fallback) were found
  outside this file's documented fallback defaults.
- **Multi-tenancy elsewhere** — `api/campaigns.py`, `api/contacts.py`,
  `api/deps.py` consistently use `get_current_org_id()` and filter by
  `organization_id`; constant-time API key comparison
  (`secrets.compare_digest`) against every configured key.
- **Scheduler ownership** — `RUN_SCHEDULER_IN_PROCESS` setting,
  documented in `config.py`, correctly gates whether `main.py`'s
  lifespan starts an in-process APScheduler vs. expecting a separate
  `worker.py` process. **NOT VERIFIED** that two processes both
  misconfigured to run the in-process scheduler wouldn't double-execute
  — no test for this.

---

## 4. What remains genuinely unverified or missing entirely

Stated plainly, per the rule at the top of this document.

| Area | Status | Why |
|---|---|---|
| Any pytest run at all | **NOT VERIFIED** | No network in this sandbox to install dependencies. |
| PostgreSQL concurrency (§19 of the original spec: dual-worker claim races, concurrent sends, duplicate webhooks) | **NOT VERIFIED / NO TEST EXISTS** | No PostgreSQL instance available; `work_claiming.py`'s logic was read and looks correct but "looks correct" is exactly what this rule exists to not accept. |
| Python E2E suite (`tests/e2e/*.py`) | **DOES NOT EXIST** | Only `test_e2e.sh` (bash + curl + jq against a running server) exists. Not rewritten to Python this session — out of scope for the remaining time; a real gap against the original spec's explicit requirement. |
| Live-provider smoke suite (`@pytest.mark.integration`, real Groq calls) | **DOES NOT EXIST** | `test_business_context.py`, referenced by both source documents, is not in this repository. |
| Provider failure-path tests (429 / timeout / malformed JSON / auth failure against `provider_v2.py`) | **NOT VERIFIED / NO DEDICATED TEST FILE** | `provider_v2.py`'s error classification was read and traced; no test exercises it directly. |
| `MessageStatus.UNKNOWN` end-to-end | **IMPLEMENTED, NOT VERIFIED** | No fault-injection test for a mid-transmission SMTP failure. |
| `/system/*` SQL aggregation correctness | **IMPLEMENTED, NOT VERIFIED** | Query logic traced by hand; not executed against a real database with representative data. |
| Work-claim lease expiry under a genuinely long-running worker | **NOT VERIFIED** | No test simulates a worker holding a claim near the lease boundary. |
| Idempotency under actual duplicate concurrent requests (not just sequential re-approval) | **PARTIAL** | `test_already_sent_message_cannot_be_approved_again` (new) proves sequential re-approval is rejected; it does not prove two *simultaneous* approval requests can't both pass the DRAFT-status check before either commits (a real race window in SQLite without row locking; PostgreSQL's row-level locking on the `UPDATE` would need its own test). |
| Alembic migration consolidation | **DEFERRED** | Deliberately not attempted without the ability to test it against a real, possibly-already-migrated database. |
| Migration `002` duplicate | **DOCUMENTED, NOT DELETED** | See §2.8. |
| `is_suppressed()` org-scoping in `api/messages.py`'s approval path | **OBSERVED, NOT CHANGED** | It's called without an `org_id` argument, meaning it checks suppression globally across all organizations rather than just the caller's org — unlike most other tenant-data access in this codebase. This is arguably a defensible "fail safe" choice (if an address unsubscribed from *any* org's campaign, don't send to it from another org either) rather than a leak, since it can only make sending *more* conservative, not less. Left unchanged this session because changing it is a real product-semantics decision (see spec item 18: "Manual suppression and unsubscribe must use the same model — document whether org-scoped or global"), not a bug fix, and shouldn't be made unilaterally without that decision being explicit. Flagging it here so the decision doesn't get lost. |
| Quality of generated email content across multiple business contexts (spec §39-40) | **NOT ATTEMPTED** | Requires a real LLM call; no live-provider suite exists in this repo to run it through, and this session had no network access to call Groq directly either. |

---

## 5. How to actually verify this

Run these yourself; this is the actual verification this report couldn't perform.

```bash
# 1. Environment
source venv/bin/activate   # or the PowerShell equivalent
python --version
pytest --version

# 2. Compile check (should already pass — verified this session)
python -m compileall -q mailer_agent

# 3. Full suite
python -m pytest -q

# 4. Individually, to isolate any failure
python -m pytest tests/test_flow.py -v
python -m pytest tests/test_grounding.py -v
python -m pytest tests/test_semantic_regression.py -v
python -m pytest tests/test_tenant_correlation.py -v      # new this session
python -m pytest tests/test_approval_safety_gate.py -v    # new this session
```

If any of the two new test files fail, that's this session's logic error
to fix, not a pre-existing issue — they were newly written, traced by
hand, and never executed. Please report failures precisely (which
assertion, expected vs actual) rather than working around them, since the
whole point of this rewrite was to stop accepting "close enough."

For the genuinely deferred work (PostgreSQL concurrency, a Python E2E
suite, a live-provider smoke suite, provider-failure tests): these need
to be written new, not just run — they don't exist in this repository yet.
