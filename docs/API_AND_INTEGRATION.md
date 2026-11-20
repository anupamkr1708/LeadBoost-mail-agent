# API and LeadBoost integration

This is the practical guide to calling Mailer from LeadBoost. It explains the workflow and the rules that are easy to get wrong. Mailbox request/response schemas live in the formal [L1 mailbox contract](LEADBOOST_L1_MAILBOX_CONTRACT.md) and are not repeated here. For the machine-readable route list, set `API_DOCS_ENABLED=true` on a non-production API and read `/openapi.json`; it is off by default.

Evidence labels used below (`IMPLEMENTED, NOT HOST-VERIFIED` and others) are defined in the [readiness report](FINAL_PRODUCTION_READINESS.md).

## 1. Who calls whom

Mailer is called **server-to-server** from the LeadBoost backend over HTTPS. The browser never talks to Mailer. Mailer keys, the Mailer database URL, mailbox passwords and the key mapping must never be placed in a frontend bundle or in any `NEXT_PUBLIC_*` variable.

LeadBoost decides what to send and whether it is authorized. Mailer validates, records and delivers, and reports state. Mailer does not discover leads, scrape, or score.

## 2. Authentication and organization mapping

Every protected route requires `X-API-Key: <tenant key>`. The key resolves to exactly one Mailer organization; nothing in the request body can change that.

The two systems hold **inverse** mappings of the same secret:

| Side | Variable | Shape |
|---|---|---|
| Mailer API | `ORG_KEY_MAP` | `{"<mailer_tenant_key>": "<mailer_org_id>"}` |
| LeadBoost backend | `MAILING_AGENT_ORG_API_KEYS` | `{"<leadboost_org_id>": "<mailer_tenant_key>"}` |
| LeadBoost backend | `MAILING_AGENT_BASE_URL` | `https://<mailer-api-host>` (no key in the URL) |

Use one distinct key per LeadBoost organization. These keys are not the operator key (`X-System-Operator-Key`, used only for `/system/config`, `/system/metrics/llm`, `/system/integration/validate` and `/system/metrics/{dispatch,worker,inbound}`) and not the mailbox encryption key. See [SECURITY_AND_SECRETS.md](SECURITY_AND_SECRETS.md).

| Status | Meaning |
|---|---|
| 401 | Missing or unknown key (same body either way) |
| 503 | No usable key source configured, or operator key / encryption key unset for a route that needs it |

`ALLOW_INSECURE_NO_AUTH` is never honored on integration or mailbox routes.

## 3. Provisioning a mailbox (once per organization)

A mailbox is Mailer-owned: Mailer stores its SMTP (and optionally IMAP) settings and encrypted passwords and sends from it. The LeadBoost-side provisioning flow is in the [L1 contract](LEADBOOST_L1_MAILBOX_CONTRACT.md); the routes are:

| Method and path | Purpose | Notes |
|---|---|---|
| `POST /mailboxes` | Create (always ACTIVE) | `201`. Passwords are write-only. `409` if `(organization, email)` already exists: adopt it with `GET /mailboxes`. IMAP is all-or-none (`422` on a partial set). `503` if `MAILBOX_ENCRYPTION_KEY` is unset. |
| `GET /mailboxes` | List the organization's mailboxes | |
| `GET /mailboxes/{public_reference}` | Retrieve one | Unknown and foreign references are both `404`. |
| `PATCH /mailboxes/{public_reference}` | Change status, replace passwords, update SMTP/IMAP transport fields | Identity fields are rejected with `422`. |

Dispatch requires **exactly one ACTIVE mailbox** for the organization. No ACTIVE mailbox, or more than one, answers `409` and nothing is queued.

## 4. Requesting an outreach

There are two intake routes. Both return `202 Accepted` with `{accepted, mailing_agent_reference}`, and both identify the operation by `(organization, idempotency_key)`. Confirm in the LeadBoost client which one it calls.

**Generated request (Mailer drafts the text).**
`POST /integrations/leadboost/outreach-requests`

Strict body (unknown fields are rejected with `422`):

- `external_action_id` (required), `idempotency_key` (required, 1 to 255 chars), `correlation_id` (optional, log-only)
- `recipient`: `email` (required), optional `name`, `title`, `company`
- `context`: `value_proposition` (required, up to 2000 chars), `recipient_facts` (list of short strings)
- `in_reply_to` (optional): `{message_reference}`, the opaque token Mailer returned for an inbound message. Makes this a **managed reply**, which is held for content approval (see §5A).

There is no sender, SMTP, mailbox, subject, body, tone or tenant field. The worker later drafts the message and grounds it against the supplied context.

**Exact-message action (caller supplies the text).**
`POST /integrations/leadboost/outreach-actions`

Same envelope with a caller-written `message`. The worker sends it verbatim; no LLM draft is involved.

### Idempotency

- A replay of an accepted key returns the existing operation and touches nothing.
- The same key with a different fingerprint (`external_action_id` plus recipient email) is `409`.
- Concurrent identical requests converge on one dispatch.
- Retry after a transport error with the **same** key. Use a **new** key only for a genuinely new action, such as retrying a `failed` dispatch.

### Other refusals

| Status | Cause |
|---|---|
| 409 | No ACTIVE mailbox; more than one ACTIVE mailbox; idempotency fingerprint conflict; recipient already has an active follow-up scheduled on the integration campaign |
| 503 | First dispatch for an organization while `LEADBOOST_INTEGRATION_SENDER_EMAIL` is unset |
| 422 | Body validation |

## 5. Reading state

| Method and path | Returns |
|---|---|
| `GET /integrations/leadboost/outreach-actions/{idempotency_key}` | `{accepted, state, mailing_agent_reference, updated_at}` |
| `GET /integrations/leadboost/outreach-actions/{idempotency_key}/conversation?limit=N` | The action (state, timestamps, mailbox reference), recent messages (direction, type, subject, body, per-message delivery state) and `has_more`. `limit` defaults to 20, maximum 50. |

An unknown key, or another organization's key, is `404`. Both endpoints are read-only.

### Interpreting `state`

| `state` | Meaning | What LeadBoost should do |
|---|---|---|
| `queued` | Accepted. May be waiting for generation, being generated, or waiting to send (the internal `generating` and, for managed replies, `awaiting_approval` states are reported as `queued`). | Wait; poll. |
| `sending` | Claimed; the SMTP attempt is in progress or about to start. | Wait; poll. |
| `sent` | The SMTP server accepted the message. **Not** proof the recipient received it. | Treat as delivered to the provider. |
| `failed` | Definitely not sent (generation refused, gate failed, SMTP refused, or a managed-reply draft was cancelled; the internal `cancelled` state is reported as `failed` here). | Fix the cause; submit a new key if still wanted. |
| `unknown` | The send's outcome could not be established. Mailer will not resend. | Do not auto-retry. Have an operator check the Sent folder or provider log. |

Accepted, queued, claimed, SMTP attempted, SMTP accepted and recipient receipt are distinct events; only the states above are visible through the API.

## 5A. Managed replies: exact-content approval

A request carrying `in_reply_to` is a **managed reply**. Its text is generated by Mailer, so the person approving it
must approve the text that will actually be sent, not an instruction to generate some. Mailer therefore generates the
draft first, holds it, and sends only after the *exact* stored subject and body have been released by digest.

```
POST outreach-requests (in_reply_to)      202, content_approval_required = true
      |
   QUEUED (no message) -> GENERATING -> AWAITING_APPROVAL   <- draft persisted, NOT sendable
      |                         |            |        \
      |                         |            |         `-- cancel --> CANCELLED  (terminal, never sends)
      |                         |            `-- release(digest) --> QUEUED (approved digest recorded)
      |                         |                                       -> SENDING -> SENT | FAILED | UNKNOWN
      `-- LLM failure / grounding hard-block --> FAILED (no Message exists; nothing sendable)
```

`AWAITING_APPROVAL` is invisible to the SMTP worker: its claim takes only `queued` rows that have a message. Nothing
moves a held row toward SMTP except the release below. Generation of **ordinary** (non-reply) outreach is unchanged.

### Endpoints (all authenticated by the organization's integration key; the organization is never taken from the request)

| Method and path | Purpose |
|---|---|
| `GET  /integrations/leadboost/reply-drafts/{idempotency_key}` | Read-only. Returns `{state, subject, body, content_sha256, draft_version, updated_at}`. Subject, body, digest and version are present exactly when a draft exists. |
| `POST /integrations/leadboost/reply-drafts/{idempotency_key}/release` | Body `{"content_sha256": "<64 lowercase hex>"}`. Authorizes the exact held draft for sending. Does **not** send. |
| `POST /integrations/leadboost/reply-drafts/{idempotency_key}/cancel` | Revokes an unreleased draft. Idempotent. |

`idempotency_key` may contain `/` (a direct API caller chooses it): send it percent-encoded (`%2F`), as LeadBoost does.
The Mailer reads everything after `reply-drafts/` and before an optional final `/release` or `/cancel` as the key, so
`GET .../reply-drafts/a%2Fb` reads the draft whose key is `a/b`. The last segment alone picks the operation: a `POST`
always needs its own `/release` or `/cancel` suffix, and a `GET` always reads (`GET .../k/release` reads the draft whose
key is `k/release`, never "release k"). The C9.1 status read keeps its single-segment key.

`state` is one of `generating`, `awaiting_approval`, `queued` (released, not yet claimed), `sending`, `sent`, `failed`,
`unknown`, `cancelled`. An unknown key, another organization's key, and an ordinary (non-reply) dispatch are all the same
`404`. No response carries a database id, a dispatch id, an RFC header, grounding context, a mailbox, or raw error text.

**Release** requires, atomically: the row is `awaiting_approval`; the exact message the draft read described; and that
the digest recomputed from the *currently stored* subject and body equals the caller's. It then records the digest and
approval time and moves the **same** dispatch to `queued`. It never calls the LLM, never creates a dispatch or a
message, and never changes the recipient or parent. Replaying a release with the digest already recorded is `200` and
changes nothing, in any later state; a different digest after release is `409`.

| Status | `detail` | Cause |
|---|---|---|
| 404 | `Reply draft not found` | Unknown key, foreign tenant, or not a managed reply (an ordinary dispatch). Never an existing legacy reply. |
| 409 | `legacy_reply_requires_reconciliation` | A reply accepted before approval-by-digest existed (flag false, `reply_parent` present): it exists, is outside the approval flow, and may be queued, sending, sent or ambiguous. Read, release and cancel all refuse; cancel is a `200` only when the row is `failed`/`cancelled`. |
| 409 | `draft_changed` | Stored content no longer matches the digest (stale or modified); nothing written, re-read and re-approve |
| 409 | `draft_not_ready` / `draft_generation_failed` / `draft_cancelled` | Nothing releasable exists |
| 409 | `draft_already_released_with_different_content` | A release was already recorded with another digest |
| 409 | `draft_already_released` (cancel) | Too late to cancel: the message now belongs to the send worker |
| 422 | `draft_too_large` / `draft_unverifiable` | Fails closed: subject > 998 or body > 20 000 characters, or text that cannot be fingerprinted. A draft that cannot be previewed in full can never be approved. |
| 422 | (validation) | Digest is not exactly 64 lowercase hex characters |

### The digest (version 1, implemented once in `mailer_agent/mail/content_digest.py`)

`sha256( json.dumps({"body": body, "subject": subject}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") ).hexdigest()`

JSON keeps field boundaries unambiguous. A `None` subject hashes as `""`, matching what the sender is handed. There is
**no normalization**: `a\n` and `a\r\n` differ, as do NFC and NFD forms. Callers must treat the digest as opaque and echo
it back unchanged. Deterministic vectors are in `tests/test_content_digest.py`.

**What it covers, honestly:** the subject and body *text*. It does not cover headers Mailer generates at send time
(`From`, `Message-ID`, `Date`) or MIME transfer encoding. `In-Reply-To` and `References` are derived from the stored
parent and are verified by tests rather than by the digest.

### Send-time integrity check

Immediately before it commits an SMTP attempt, the SMTP worker hashes the exact `subject`/`body_text` it is about to
pass to the sender and compares them with the recorded approval. Mismatch, or a managed reply with no recorded approval,
fails the dispatch (`approval_content_mismatch` / `approval_missing`): SMTP is never contacted, nothing is regenerated,
and the stored reason carries no message content. Suppression, grounding, mailbox, `LIVE_SENDING_ENABLED`, lease and
`UNKNOWN` semantics are unchanged; an ambiguous SMTP outcome is still never resent.

### What did not change

`POST /messages/{id}/approve` still refuses every integration-managed message in every state, including a held managed
reply (`tests/test_message_approval_external_dispatch_guard.py`, plus a new held-reply case). The draft stays an
integration-owned `DRAFT`; it never becomes a native draft. Ordinary outreach, native campaigns and native replies,
the conversation endpoint (still read-only, never calls an LLM) and the reconciliation endpoint's five-state vocabulary
are unchanged.

### Rollout and compatibility

The first release is deliberately conservative: it **blocks** while any unresolved legacy managed reply exists, instead of
migrating old approvals. Full operator procedure (with the exact order, the stuck-row table and the guarded SQL):
[`migrations/README.md`, "Legacy managed replies"](../migrations/README.md). In short:

1. **Pause LeadBoost managed-reply API traffic** (edge `503` on `POST /api/v2/outreach-actions*`), **drain old API
   requests** to zero in flight, and keep the **old Mailer workers running to drain** legacy rows. Do not start any
   new-code worker yet: a new generation worker would hold a legacy queued reply in `AWAITING_APPROVAL`, where the new
   LeadBoost UI (which treats the old action as legacy) cannot review or release it.
2. **Run the gate** -- read-only without `--apply`, and run automatically before any migration by `scripts/migrate_release.sh`:

   ```bash
   DATABASE_URL=... python scripts/migrate_preflight.py gate
   ```

   It **refuses (exit 4, nothing changed)** while ANY *legacy* managed-reply dispatch -- accepted before approval-by-digest
   existed, i.e. `content_approval_required` not set -- is **queued (with or without a generated message)**,
   **generating**, or **sending**, and lists the blocking rows (ids, state, age only). Resolve each deliberately -- let the
   old workers finish it, or abandon an unclaimed one with the guarded statement in the README -- never by editing it into
   the approval states. Legacy `unknown` rows are reported, not blocking (nothing re-sends them). To inspect by hand (the
   column is JSON, so this is a JSON-key test; a `LIKE` on it is not valid on PostgreSQL):

   ```sql
   SELECT state, (message_id IS NOT NULL) AS has_message, count(*)
     FROM external_dispatches
    WHERE json_typeof(grounding_context -> 'reply_parent') = 'object'   -- jsonb_typeof if the column is jsonb
      AND state IN ('queued', 'generating', 'sending')
      AND content_approval_required = false          -- omit this line before migration 013 has run
    GROUP BY 1, 2 ORDER BY 1, 2;                      -- blocking = ANY row returned
   ```
3. When the gate passes, **stop/drain the old workers** (graceful stop, confirm no old process remains) and re-run the gate.
   Then run migration 013, deploy and start the new workers. The gate stays quiet once the new code is the only code
   running, because it sets the flag on every reply at acceptance.
4. Deploy Mailer before LeadBoost, then lift the traffic pause. A caller that does not implement the draft/release
   endpoints (an older LeadBoost) gets a held draft that **never sends**; it reads as `queued`. That is deliberate: it fails
   closed rather than sending unreviewed generated text.

**Legacy replies and the draft endpoints.** A reply accepted before migration 013 *exists* at the Mailer but is not in the
approval flow, so it is never a `404`: read and release answer `409 legacy_reply_requires_reconciliation`; cancel answers the
same `409` unless the Mailer can prove nothing was or can be sent (`failed`, or an operator-`cancelled` row), in which case it
is a `200` with no content. No endpoint ever writes to a legacy row. LeadBoost reads `404` as "the Mailer has no such
operation" (and may then cancel its own action), so a `404` for an existing legacy reply would be unsafe.

### Known limitations

- A held draft has no lease and no expiry; it waits until released or cancelled. A stale-draft sweep is not implemented.
- Content is immutable in v1: no editing and no regeneration, so `draft_version` is always `1`. Either would need
  versioned re-approval.
- Release/cancel/fenced-completion atomicity rests on single guarded `UPDATE` statements. It is tested with sequential
  orderings and a two-thread race on SQLite, under genuine concurrency on PostgreSQL 16
  (`tests/test_managed_reply_approval_postgres.py`: 25 release-vs-cancel races, 12 concurrent identical releases, a
  wrong-tenant/wrong-message race, cancel vs the generation worker's fenced completion, and concurrent SMTP claimers
  that never see a held draft), and end to end by LeadBoost's cross-service E2E. Other PostgreSQL versions and
  replicated/multi-node setups were not tested.

## 6. Inbound email

- **IMAP (preferred).** A mailbox with IMAP configured is polled by the worker; the mailbox fixes the organization. LeadBoost sees replies as inbound messages in the conversation endpoint.
- **Webhook (optional).** `POST /webhooks/inbound-email` with a tenant key lets an inbound-parse provider post replies. The organization comes from the key, not the payload.

Mail polled through a mailbox that cannot be matched to exactly one contact in that organization is never attached to a contact and never appears in a conversation; it is retained separately for operator review (`unmatched_inbound_messages`). The webhook and legacy paths do not retain it (see [SYSTEM_DESIGN.md](architecture/SYSTEM_DESIGN.md#9-reply-ingestion-mailbox-ownership-and-deduplication)).

## 7. Health and diagnostics

| Path | Auth | Meaning |
|---|---|---|
| `GET /health` | none | The API process is up. No database or worker probe. |
| `GET /system/health` | tenant key | Organization-scoped counts from the database; implies the database is reachable. |
| `GET /system/metrics/classification`, `/system/metrics/states` | tenant key | Organization-scoped metrics. |
| `GET /system/config`, `/system/metrics/llm`, `/system/integration/validate`, `/system/metrics/dispatch`, `/system/metrics/worker`, `/system/metrics/inbound` | `X-System-Operator-Key` | Process-wide, operator only. `503` when the operator key is unset. The last three are read-only aggregates described in [OPERATOR_OBSERVABILITY.md](OPERATOR_OBSERVABILITY.md); they are not per-tenant analytics and a tenant key never opens them. |

There is no `/metrics` route (the deployed API answered 404; `HOSTED OBSERVED`). The worker has no HTTP endpoint; observe it through its logs.

## 8. Operational prerequisites before the first request

1. `DATABASE_URL`, `ORG_KEY_MAP` and `MAILBOX_ENCRYPTION_KEY` set on the API; `DATABASE_URL`, `MAILBOX_ENCRYPTION_KEY` and `GROQ_API_KEY` on the worker, with the **same** encryption key on both.
2. The migration wrapper has run against the database (see [migrations/README.md](../migrations/README.md)).
3. A mailbox created for the organization.
4. `LEADBOOST_INTEGRATION_SENDER_EMAIL` set on the API.
5. A worker running. Without one, requests are accepted and stay `queued`.
6. `LIVE_SENDING_ENABLED=true` on the worker only when real sends are intended; otherwise dispatches end as `failed` with `live_sending_disabled`.
