# Mailer Agent — Production Readiness Report

Status legend: **VERIFIED** (executed and observed) / **IMPLEMENTED** (code
exists, traced by hand, not executed) / **NOT VERIFIED** (genuinely
unknown) / **PARTIAL** / **DEFERRED**.

This is the one production-readiness document for this repository,
covering four work sessions. §1-3 summarize sessions 1-3 (hardening pass
+ Phase 2 semantic architecture). §4 is this session: the first one with
**real test execution results**, reported by the user from an actual
environment with dependencies and PostgreSQL installed — not traced by
hand. §5 is the honest cross-session status table. §6-9 are the closing
sections.

## The rule this report follows, unchanged across all four sessions

> Code inspection ≠ Runtime verification. Unit test ≠ Concurrency
> verification. Live LLM smoke test ≠ Deterministic regression test.
> Draft grounding ≠ Final send safety. Scheduler exists ≠ Distributed
> scheduling correctness.

This session is different from the prior three in one important way:
**for the first time, real pass/fail/skip counts exist**, reported by
the user from an environment this assistant does not have access to
(dependencies installed, PostgreSQL 16.14 running). Every fix below was
made in direct response to a specific, real, reported failure and its
diagnosed root cause — not a hypothetical one found by inspection. That
said, the fixes THEMSELVES have not been re-executed by either party as
of this writing. §9 says exactly what to run to close that loop. Do not
read "the failure was diagnosed correctly and fixed" as "the fix is
confirmed working" — those are still different claims, per the rule
above.

---

## 1-3. Sessions 1-3 summary (condensed)

**Session 1**: implemented the approval-time grounding recheck; implemented
`MessageStatus.UNKNOWN` end-to-end; fixed a cross-tenant leak in
`/system/*`; removed dead/broken code; fixed a test-isolation bug where
`llm/provider.py` bound functions at import time; rewrote the
keyword-guessing `FakeLLMProvider`.

**Session 2**: fixed the inbound-correlation ambiguous-match guard; fixed
Message-ID normalization across webhook/IMAP; fixed an authentication-
retry bug; fixed a suppression gap in auto-reply; added a deterministic
Python E2E suite and a context-contamination test.

**Session 3 (Phase 2 semantic architecture)**: found the planner
(`policy/next_action.py`) was orphaned architecture (one dead reference,
never called); rewrote it as an LLM-backed Planner and wired it into the
real reply path via new deterministic Guardrails; removed the
`positive_interest + evaluating -> pricing_discussed` semantic
assumption from the state machine; removed the hardcoded 30-day timing
fallback; fixed the critical contextual-reply fallback (was generating a
generic CTA regardless of what the prospect said); added memory
accumulation, observability/replay, an evaluation corpus, architecture
tests (genuinely executed via `ast`, not traced), and found + fixed two
real concurrency gaps (`message_id_header` had no DB-level uniqueness;
`approve_and_send_draft` had no row lock) while *designing* the
PostgreSQL test suite.

---

## 4. This session: targeted correctness repair from real execution results

Explicitly scoped, per the instructions that opened this session: no
semantic architecture changes, no new AI components, no new frameworks —
only fixes for confirmed defects, each backed by a specific reported
failure and diagnosis.

### 4.1 (TASK A) Conversation-evidence provenance — root cause of both approval-safety-gate failures

**Reported failures**: `test_never_grounded_claim_is_blocked_at_approval`
and `test_context_changed_since_draft_generation_blocks_send`, both in
`tests/test_approval_safety_gate.py`.

**Diagnosed root cause** (via direct diagnostic, not inspection):
`memory/store.py`'s `build_conversation_context()` and
`maybe_summarize_older_messages()` both did `list(contact.messages)` —
every Message row for the contact, regardless of status — and
unconditionally labeled every outbound row `"US (sent)"` in the
transcript. A DRAFT message (never sent, possibly containing an
unsupported claim) was therefore included in its own "conversation
evidence." When `validate_grounding()` checked whether a claim in the
draft was supported by `proof_points + context_notes +
conversation_transcript + value_prop`, the draft's own unsupported claim
appeared inside `conversation_transcript` (via its own inclusion in the
message list) — making the check circular. The draft appeared to
support itself.

**Fix**: one shared filter, `_conversational_evidence()` in
`memory/store.py`, applied at both call sites:
- Inbound messages are always valid evidence.
- Outbound messages count only when `status == MessageStatus.SENT.value`
  exactly. `DRAFT`, `FAILED`, and `UNKNOWN` (ambiguous SMTP outcome —
  deliberately not treated as "probably sent," consistent with how
  `UNKNOWN` is already handled everywhere else in this codebase) are all
  excluded. Uses the real `MessageStatus`/`MessageDirection` enums
  already in `models.py` — no new status values invented, no keyword/
  semantic logic involved.

This is a plausible, traced explanation for both reported failures (the
mechanism — a draft's claim appearing in its own evidence — applies
directly to both test scenarios), but has not been re-executed to
confirm the tests now pass. See §9.

**New regression tests**: `tests/test_memory_provenance.py` (7 tests) —
DRAFT absent from context; DRAFT cannot self-ground (direct
reproduction of the circular-evidence mechanism); a draft grounded at
creation becomes ungrounded after `proof_points` changes; a stale DRAFT
never enters the rolling summary (even after aging out of the verbatim
window); SENT outbound and INBOUND messages remain valid evidence;
FAILED/UNKNOWN messages are excluded the same as DRAFT.

### 4.2 (TASK B) Pre-existing memory summaries — documented, no migration

`contact.memory_summary` is a plain `Text` column with no per-fact
provenance, so a summary persisted before the fix in §4.1 may have been
generated from a window that included a draft's content — and there is
no way to *selectively* correct that in the stored text. No schema
migration was written: the column itself doesn't change, and
automatically re-summarizing every existing contact would mean an
unreviewed LLM call per contact with its own new-error risk, which isn't
obviously safer than a stale field. Documented directly on the
`memory_summary` column in `models.py`, including the safe manual
remedy for a specific contact if one is suspected (clear that one
`memory_summary` value; it regenerates correctly, now provenance-
filtered, next time the thread crosses `SUMMARIZE_THRESHOLD`).

### 4.3 (TASK C) Provider error classification precedence

**Reported failure**: a persistent-5xx provider-failure test failed
because `"504 Gateway Timeout"` was classified as `TimeoutError`, not
`ProviderUnavailableError`. Root cause given directly: the generic
`"timeout" in error_str` text check ran *before* the explicit 5xx status-
code check, and `"504 Gateway Timeout"` contains the word "timeout" in
its standard HTTP reason phrase.

**Fix**: reordered `_classify_error()` in `llm/provider_v2.py` so every
explicit-status-code branch (401/403 → `AuthenticationError`, 429 →
`RateLimitError`, 500/502/503/504 → `ProviderUnavailableError`, 400 →
`ValidationError`) is checked before the generic text-only branches
(`timeout`, malformed JSON) that exist for errors carrying no HTTP
status at all. The retry architecture itself was not touched — still
max 3 attempts, no SDK-level retry, the same three retryable exception
types, no second retry framework.

**Verified by direct execution in this session** (not traced): a
minimal standalone reproduction of the exact bug (a lowered string
containing both a 5xx code and the word "timeout") was run in this
sandbox using pure Python, confirming the *old* precedence would
misclassify and the *new* precedence classifies correctly. This is
narrower than actually running the real test suite, but it is a genuine
execution, not an inspection.

**New/updated tests**: `tests/test_provider_failures.py` gained
`test_504_gateway_timeout_classified_as_provider_unavailable_not_timeout`
(the focused regression) and
`test_generic_timeout_without_http_status_still_classified_as_timeout`
(proves the reordering didn't disable the generic case it's meant to
still cover).

### 4.4 (TASK D) SQLite E2E fixture cross-connection bug

**Reported failure**: `tests/e2e/test_full_lifecycle.py`, 4 failures,
`sqlite3.OperationalError: no such table: campaigns`.

**Diagnosed root cause**: `create_engine("sqlite:///:memory:", ...)`
without `poolclass=StaticPool` lets SQLAlchemy's connection pool hand out
a *different* connection — and therefore a completely separate, empty
in-memory database, since SQLite's `:memory:` databases are connection-
local — to different consumers. `Base.metadata.create_all()` ran against
one connection; `TestClient`'s request handling (which can dispatch
through a different thread depending on the ASGI transport) could be
handed a different one with no tables at all.

**Fix**: added `poolclass=StaticPool` (forces the whole engine to share
exactly one underlying connection) alongside the existing
`check_same_thread=False`. Production database wiring
(`mailer_agent/db.py`) was not touched — this is purely a test-fixture
change. `tests/test_approval_safety_gate.py` and
`tests/test_tenant_correlation.py` do not need this fix (already
reported passing) because they never go through `TestClient`/cross-
thread access — they call functions directly with the same session
object in the same thread.

### 4.5 (TASK E) `tests/test_flow.py` isolation and auth

**Reported failure**: 401s cascading into `KeyError("id")` (a 401
response body has no `"id"` key).

**Diagnosed root cause**: the file relied on module-level
`os.environ["API_KEY"] = ""` / `os.environ["DATABASE_URL"] = ...`
mutations, combined with `api/deps.py`'s key→org map and
`mailer_agent.db`'s engine, both built once at first import. Whichever
test file's imports happen to run first in the session decides what
those singletons end up bound to; a later file's environment mutations
have no effect on an already-built map/engine. This is the exact
fragility class flagged and fixed for other files in sessions 2-3 (see
`tests/e2e/test_full_lifecycle.py`'s own docstring), just not yet
applied here.

**Fix**: rewrote the `client` fixture around the same robust pattern
already used in `tests/e2e/test_full_lifecycle.py` — FastAPI
`dependency_overrides` for `get_db`/`require_api_key`/`get_current_org_id`,
a fresh `StaticPool`-backed in-memory engine per test, no environment-
variable reliance at all. `test_inbound_webhook_matches_thread` also
needed an explicit fix beyond the fixture: this session's Planner
changes mean a reply now makes up to three LLM calls (classify → plan →
draft), and this test doesn't care about classification content — so it
now explicitly forces `is_llm_available()` to `False` for a clean,
fixture-free deterministic-fallback path, rather than needing to queue
three fixtures for a test whose assertions don't depend on them. No
`201`/`200` assertions were weakened; the fixture change is orthogonal to
what each test actually checks.

### 4.6 (TASK F) Architecture test's postponed-annotation bug

**Reported failure**: `test_guardrails_authorize_action_takes_typed_proposal`
failed because `guardrails.py` has `from __future__ import annotations`
(PEP 563), so `inspect.signature(...).annotation` returns the *string*
`"NextActionProposal"`, not the class object the test compared against
with `is`.

**Fix**: switched to `typing.get_type_hints(authorize_action)`, which
resolves postponed-annotation strings back into real objects against the
function's own module globals. The production type contract itself
(`guardrails.authorize_action(proposal: NextActionProposal, ...)`) was
not touched — this was purely a test bug, not a production one.

**Verified by direct execution in this session** (not traced): built a
minimal standalone reproduction matching the exact real shape
(`from __future__ import annotations`, a dataclass parameter, keyword-
only arguments, a dataclass return type) and ran it in this sandbox.
Confirmed `inspect.signature(...).annotation` returns the string (the
bug, reproduced) and `typing.get_type_hints(...)` returns the real class
object matching `is NextActionProposal` (the fix, confirmed). This is a
genuine execution of the exact logic pattern, though not the real
`mailer_agent.policy.guardrails` module itself (which needs `tenacity`/
`pydantic-settings`, unavailable in this sandbox).

### 4.7 (TASK G) Integration tests leaking into the default suite

**Reported issue**: default `python -m pytest -q` executed live Groq
integration tests, making the default suite nondeterministic and rate-
limit dependent.

**Diagnosed root cause**: session 3 registered the `integration` marker
in `pytest.ini` and made the autouse `fake_llm` fixture skip patching for
integration-marked tests (so they'd genuinely hit the real API when
explicitly requested) — but never told pytest to *exclude* that marker
by default. Registering a marker and excluding it by default are two
different steps, and only the first was done.

**Fix**: added `addopts = -m "not integration"` to `pytest.ini`. Default
`pytest -q` now excludes anything marked `integration` (which includes
`tests/evaluation/test_live_llm_semantic_quality.py` and — since it's
marked with both `integration` and `postgres` — `tests/test_postgresql_concurrency.py`
too). Running `python -m pytest -m integration -v` explicitly overrides
the ini default (pytest's marker-expression option takes the
last-specified value when both `addopts` and the command line specify
`-m`, so the explicit CLI flag wins) — this is standard, documented
pytest behavior for this exact `addopts`-default-plus-CLI-override
pattern, not independently re-executed in this sandbox (no `pytest`
installed here at all).

### 4.8 What was explicitly NOT touched (TASK H, honored)

No new AI components, classifier layers, retrieval/vector DB, or
generic agent framework were introduced. `SemanticIntent` was not
redesigned. The planner architecture (LLM proposes via
`policy/next_action.py`, guardrails authorize via
`policy/guardrails.py`) is unchanged from session 3. The retry
architecture in `provider_v2.py` (max 3 attempts, no SDK-level retry,
exactly the same three retryable exception types) is unchanged — only
the *classification order* that decides which exception type an error
maps to was reordered.

---

## 5. Cross-session verification status

| Area | Status | Session |
|---|---|---|
| Approval-time grounding recheck | IMPLEMENTED, not run | 1 |
| `MessageStatus.UNKNOWN` wired through | IMPLEMENTED, not run | 1 |
| `/system/*` tenant scoping | IMPLEMENTED, not run | 1 |
| Tenant-safe inbound correlation | **VERIFIED PASSING** (5/5, user-reported) | 1+2 |
| Provider retry/auth-not-retried | PARTIAL — auth-retry fix verified conceptually; 5xx/timeout precedence bug found+fixed this session | 2+4 |
| Deterministic Python E2E suite | Was failing (4/4, cross-connection bug); fix applied, not re-run | 2, fixed 4 |
| Semantic contracts, interpreter, planner, guardrails | **VERIFIED PASSING** (grounding 32/32, planner/guardrails 16/16, semantic regression 17/17, evaluation 12/12, user-reported) | 3 |
| Approval safety gate (grounding recheck at approval) | Was failing (2/2 relevant tests); root cause diagnosed and fixed this session, not re-run | 3, fixed 4 |
| Conversation memory provenance | **NEWLY FIXED** this session, not run | 4 |
| Architecture tests | Was failing (1 test, postponed-annotation bug); fix applied and verified via standalone reproduction, not re-run against the real module | 3, fixed 4 |
| `tests/test_flow.py` | Was failing (401 cascade); fix applied, not re-run | 4 |
| Integration/deterministic test separation | **NEWLY FIXED** this session (`addopts`), not run | 4 |
| PostgreSQL concurrency suite | Instance now confirmed running (PostgreSQL 16.14) by the user; suite itself not yet reported run | 3 (written), 4 (env confirmed) |
| Live-LLM semantic quality suite | NOT VERIFIED / no report yet | 3 (written) |

---

## 6. Files changed this session

`mailer_agent/memory/store.py` (§4.1), `mailer_agent/models.py` (§4.2,
documentation only), `mailer_agent/llm/provider_v2.py` (§4.3),
`tests/e2e/test_full_lifecycle.py` (§4.4), `tests/test_flow.py` (§4.5),
`tests/test_architecture.py` (§4.6), `pytest.ini` (§4.7). New:
`tests/test_memory_provenance.py` (7 tests, §4.1).
`tests/test_provider_failures.py` gained 2 tests (§4.3).

## 7. Why each change was necessary (one line each)

- `memory/store.py`: without it, a draft's own unsupported claim could
  appear as its own supporting evidence, defeating the entire grounding
  safety gate.
- `models.py`: no code change needed; documents a real data-lineage gap
  so it isn't silently forgotten.
- `provider_v2.py`: without it, a real 502/503/504 provider outage is
  misreported as a client-side timeout in monitoring/observability, even
  though the retry behavior itself was already correct.
- `test_full_lifecycle.py`: without `StaticPool`, the E2E suite cannot
  reliably see its own test data through `TestClient`.
- `test_flow.py`: without dependency overrides, this file's auth/DB
  setup silently depends on which other test file happens to import
  first.
- `test_architecture.py`: without `get_type_hints`, this test fails
  under postponed annotations regardless of whether the production
  typing contract is actually correct — a false negative, not evidence
  of a real problem.
- `pytest.ini`: without `addopts`, every default CI run burns real Groq
  API quota and is nondeterministic.

## 8. Confirmation: no semantic architecture redesign

No new classifier layers, no DeBERTa/RoBERTa/RAG/vector DB, no
LangChain/Kafka/Celery, no generic agent framework. `SemanticIntent` is
unchanged from session 3. The planner/guardrails split is unchanged. The
only planner/guardrails-adjacent change this session was to a *test*
(`test_architecture.py`'s type-resolution method), not to
`policy/next_action.py` or `policy/guardrails.py` themselves — both are
byte-for-byte unchanged since session 3. The retry *architecture* in
`provider_v2.py` (attempt count, which exceptions retry, no SDK-level
retry) is unchanged; only classification *precedence* — which bucket a
given error string falls into before the existing retry logic sees it —
was reordered.

## 9. Exact commands to run — this is the loop that needs closing

```powershell
.\venv\Scripts\Activate.ps1

# 1
python -m compileall -q mailer_agent

# 2-3: the two directly-diagnosed failures this session's TASK A targets
python -m pytest tests/test_grounding.py -v
python -m pytest tests/test_approval_safety_gate.py -v

# 4: TASK C
python -m pytest tests/test_provider_failures.py -v

# 5: TASK D
python -m pytest tests/e2e/test_full_lifecycle.py -v

# 6: TASK E
python -m pytest tests/test_flow.py -v

# 7: TASK F
python -m pytest tests/test_architecture.py -v

# 8-11: should be unaffected by this session's changes -- confirm they still are
python -m pytest tests/test_semantic_regression.py -v
python -m pytest tests/test_planner_and_guardrails.py -v
python -m pytest tests/evaluation/test_semantic_evaluation.py -v
python -m pytest tests/test_tenant_correlation.py -v

# New this session -- not in the original list but exercises TASK A directly
python -m pytest tests/test_memory_provenance.py -v

# 12: the actual point of TASK G -- confirm this does NOT touch the network
python -m pytest -q

# Explicit integration run (only when you want to spend real API budget/
# have PostgreSQL reachable)
python -m pytest -m integration -v
```

Report back: exact pass/fail/skip counts, and for anything still failing,
the exact assertion and actual-vs-expected — the same standard this
report has held since session 1. Do not delete, skip, weaken, or rename
a failing test to make it pass.
