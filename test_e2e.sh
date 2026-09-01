#!/usr/bin/env bash
#
# Mailer Agent — full end-to-end API test script.
#
# Exercises every endpoint the service exposes, in a realistic order:
# create a campaign -> add contacts (both the strict format and the
# flexible LeadBoost-shaped ingestion) -> send initial outreach ->
# force a follow-up -> simulate an inbound reply -> approve the drafted
# reply -> pause/resume -> mark won -> suppression.
#
# HOW TO USE
#   1. Install jq (used to build/parse JSON safely): 
#        macOS:  brew install jq
#        Ubuntu: sudo apt-get install jq
#        Windows (WSL): sudo apt-get install jq
#   2. Fill in the CONFIG block below with your own values.
#   3. chmod +x test_e2e.sh
#   4. ./test_e2e.sh
#
# Safe to run against a server in dry-run mode (LIVE_SENDING_ENABLED=false,
# the default) -- nothing will actually be emailed, every send is just
# logged server-side, but the full flow (drafting, memory, cadence,
# reply classification, suppression) still runs for real. See the
# "How to run the system" guide for the sequence of flipping toggles
# to test with a real inbox.

set -uo pipefail

# ============================================================================
# CONFIG -- fill these in
# ============================================================================

# Where the service is running. Can be overridden via environment variable.
BASE_URL="${BASE_URL:-http://localhost:8000}"

# API key for authentication. MUST be set via environment variable if the
# server requires authentication. Do NOT hardcode secrets here.
API_KEY="${API_KEY:-}"

# --- Your sending identity (used to create the test campaign) --------------
# These are non-sensitive test fixtures - safe to have defaults but can be overridden
SENDER_NAME="${SENDER_NAME:-Test Sender}"
SENDER_ORG="${SENDER_ORG:-Test Organization}"
SENDER_EMAIL="${SENDER_EMAIL:-test-sender@example.com}"
REPLY_TO_EMAIL="${REPLY_TO_EMAIL:-test-sender@example.com}"
VALUE_PROP="${VALUE_PROP:-We help B2B SaaS teams cut support response time 40% with an AI triage layer that installs in under a day.}"
PROOF_POINTS="${PROOF_POINTS:-Used by 3 YC-backed startups; average setup time 6 hours.}"

# Cadence to test with. [3,7,14] is realistic for production but useless
# for manual testing -- [0,0] means "immediately eligible", which pairs
# well with the /force-followup calls below either way.
FOLLOW_UP_DAYS="${FOLLOW_UP_DAYS:-[3,7]}"

# --- Test contacts -----------------------------------------------------------
# Non-sensitive test fixtures - use example.com addresses by default
CONTACT_1_NAME="${CONTACT_1_NAME:-Test Contact}"
CONTACT_1_EMAIL="${CONTACT_1_EMAIL:-test-contact@example.com}"
CONTACT_1_TITLE="${CONTACT_1_TITLE:-Head of Support}"
CONTACT_1_COMPANY="${CONTACT_1_COMPANY:-ExampleCorp}"

# This one is deliberately shaped like a raw LeadBoost `Lead` row, to
# exercise the flexible /leads/ingest endpoint the way LeadBoost itself
# will call it.
LEADBOOST_LEAD_EMAIL="${LEADBOOST_LEAD_EMAIL:-test-leadboost@example.com}"


# ============================================================================
# Helpers
# ============================================================================

AUTH_ARGS=()
if [ -n "$API_KEY" ]; then
  AUTH_ARGS=(-H "X-API-Key: $API_KEY")
fi

section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

pp() {
  if command -v jq >/dev/null 2>&1; then jq .; else cat; fi
}

check_jq() {
  if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq is required. Install it (brew install jq / apt-get install jq) and re-run." >&2
    exit 1
  fi
}
check_jq

# ============================================================================
# 1. Health check
# ============================================================================
section "1. GET /health"
curl -s "${BASE_URL}/health" | pp

# ============================================================================
# 2. Create a campaign
# ============================================================================
section "2. POST /campaigns"
CAMPAIGN_BODY=$(jq -n \
  --arg name "Manual E2E Test Campaign" \
  --arg sender_name "$SENDER_NAME" \
  --arg sender_org "$SENDER_ORG" \
  --arg sender_email "$SENDER_EMAIL" \
  --arg reply_to_email "$REPLY_TO_EMAIL" \
  --arg value_prop "$VALUE_PROP" \
  --arg proof_points "$PROOF_POINTS" \
  --argjson follow_up_days "$FOLLOW_UP_DAYS" \
  '{name:$name, sender_name:$sender_name, sender_org:$sender_org, sender_email:$sender_email,
    reply_to_email:$reply_to_email, value_prop:$value_prop, proof_points:$proof_points,
    follow_up_days:$follow_up_days}')

CAMPAIGN_JSON=$(curl -s -X POST "${BASE_URL}/campaigns" "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" -d "$CAMPAIGN_BODY")
echo "$CAMPAIGN_JSON" | pp
CAMPAIGN_ID=$(echo "$CAMPAIGN_JSON" | jq -r '.id // empty')

if [ -z "$CAMPAIGN_ID" ]; then
  echo "ERROR: campaign creation failed, stopping here. Check the response above." >&2
  exit 1
fi
echo ">> CAMPAIGN_ID=$CAMPAIGN_ID"

# ============================================================================
# 3. Fetch it back / list campaigns
# ============================================================================
section "3. GET /campaigns/$CAMPAIGN_ID and GET /campaigns"
curl -s "${BASE_URL}/campaigns/${CAMPAIGN_ID}" "${AUTH_ARGS[@]}" | pp
curl -s "${BASE_URL}/campaigns" "${AUTH_ARGS[@]}" | pp

# ============================================================================
# 4. Add a contact via the strict /contacts endpoint
# ============================================================================
section "4. POST /campaigns/$CAMPAIGN_ID/contacts (strict format)"
CONTACT_BODY=$(jq -n \
  --arg name "$CONTACT_1_NAME" --arg email "$CONTACT_1_EMAIL" \
  --arg title "$CONTACT_1_TITLE" --arg company "$CONTACT_1_COMPANY" \
  '{contacts: [{name:$name, email:$email, title:$title, company:$company,
    context_notes: "Posted 2 support-engineer job listings last month."}]}')

CONTACTS_JSON=$(curl -s -X POST "${BASE_URL}/campaigns/${CAMPAIGN_ID}/contacts" "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" -d "$CONTACT_BODY")
echo "$CONTACTS_JSON" | pp
CONTACT_1_ID=$(echo "$CONTACTS_JSON" | jq -r '.[0].id // empty')
echo ">> CONTACT_1_ID=$CONTACT_1_ID"

# ============================================================================
# 5. Add leads via the flexible /leads/ingest endpoint
#    (one LeadBoost-shaped lead, one generic-shaped lead, one bad lead
#    with no email to prove the batch degrades gracefully)
# ============================================================================
section "5. POST /campaigns/$CAMPAIGN_ID/leads/ingest (flexible format)"
INGEST_BODY=$(jq -n \
  --arg email "$LEADBOOST_LEAD_EMAIL" \
  '{leads: [
      {company_name:"ExampleCorp Two", contact_name:"Rahul Verma", contact_title:"VP Engineering",
       email:$email, industry:"B2B SaaS", employees:"51-200", revenue_band:"$10M-50M",
       qualification_label:"Hot Lead", score:87.5},
      {full_name:"Sam Lee", lead_email:"sam-generic-format@example.com", org:"OtherCorp", job_title:"CTO"},
      {company_name:"NoEmailCorp"}
  ]}')

INGEST_JSON=$(curl -s -X POST "${BASE_URL}/campaigns/${CAMPAIGN_ID}/leads/ingest" "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" -d "$INGEST_BODY")
echo "$INGEST_JSON" | pp
CONTACT_2_ID=$(echo "$INGEST_JSON" | jq -r '.created[0].contact_id // empty')
echo ">> CONTACT_2_ID=$CONTACT_2_ID (should show 2 created, 1 skipped for missing email)"

# ============================================================================
# 6. List contacts in the campaign
# ============================================================================
section "6. GET /campaigns/$CAMPAIGN_ID/contacts"
curl -s "${BASE_URL}/campaigns/${CAMPAIGN_ID}/contacts" "${AUTH_ARGS[@]}" | pp

# ============================================================================
# 7. Start the campaign -- sends initial outreach to every NEW contact
# ============================================================================
section "7. POST /campaigns/$CAMPAIGN_ID/start"
curl -s -X POST "${BASE_URL}/campaigns/${CAMPAIGN_ID}/start" "${AUTH_ARGS[@]}" | pp

# ============================================================================
# 8. Check contact status + full thread after the initial send
# ============================================================================
section "8. GET /contacts/$CONTACT_1_ID and its thread"
curl -s "${BASE_URL}/contacts/${CONTACT_1_ID}" "${AUTH_ARGS[@]}" | pp
curl -s "${BASE_URL}/contacts/${CONTACT_1_ID}/thread" "${AUTH_ARGS[@]}" | pp

# ============================================================================
# 9. Force a follow-up now (bypasses the follow_up_days wait)
# ============================================================================
section "9. POST /contacts/$CONTACT_1_ID/force-followup"
curl -s -X POST "${BASE_URL}/contacts/${CONTACT_1_ID}/force-followup" "${AUTH_ARGS[@]}" | pp

echo
echo "Thread after follow-up (subject should differ from the initial email --"
echo "confirms the agent is varying subject lines, not reusing 'Re: <original>'):"
curl -s "${BASE_URL}/contacts/${CONTACT_1_ID}/thread" "${AUTH_ARGS[@]}" | pp

# ============================================================================
# 10. Simulate an inbound reply via the webhook endpoint
#     (this is the same code path a real Postmark/Mailgun webhook -- or
#     the IMAP poller -- would trigger)
# ============================================================================
section "10. POST /webhooks/inbound-email (simulated prospect reply)"
WEBHOOK_BODY=$(jq -n \
  --arg from_email "$CONTACT_1_EMAIL" \
  '{from_email:$from_email, subject:"Re: following up",
    body_text:"This looks interesting -- can we do a call next week? Also, what does pricing look like for a team of 50?",
    message_id:"<manual-test-reply-1@example.com>"}')

WEBHOOK_JSON=$(curl -s -X POST "${BASE_URL}/webhooks/inbound-email" "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" -d "$WEBHOOK_BODY")
echo "$WEBHOOK_JSON" | pp

# ============================================================================
# 11. Inspect the thread -- should now show the inbound message, its
#     classified intent, and (if AUTO_REPLY_ENABLED=false, the default)
#     a drafted reply awaiting approval
# ============================================================================
section "11. GET /contacts/$CONTACT_1_ID/thread (after the reply)"
THREAD_JSON=$(curl -s "${BASE_URL}/contacts/${CONTACT_1_ID}/thread" "${AUTH_ARGS[@]}")
echo "$THREAD_JSON" | pp
DRAFT_MESSAGE_ID=$(echo "$THREAD_JSON" | jq -r '.messages[] | select(.status=="draft") | .id' | tail -n1)
echo ">> DRAFT_MESSAGE_ID=${DRAFT_MESSAGE_ID:-none found}"

# ============================================================================
# 12. Approve and send the drafted reply (only runs if one was found --
#     if AUTO_REPLY_ENABLED=true on the server, there may be nothing to
#     approve because it already auto-sent)
# ============================================================================
if [ -n "${DRAFT_MESSAGE_ID:-}" ]; then
  section "12. POST /messages/$DRAFT_MESSAGE_ID/approve"
  curl -s -X POST "${BASE_URL}/messages/${DRAFT_MESSAGE_ID}/approve" "${AUTH_ARGS[@]}" | pp
else
  section "12. Skipped -- no draft message pending approval"
fi

# ============================================================================
# 13. Pause / resume a contact
# ============================================================================
section "13. POST /contacts/$CONTACT_2_ID/pause then /resume"
curl -s -X POST "${BASE_URL}/contacts/${CONTACT_2_ID}/pause" "${AUTH_ARGS[@]}" | pp
curl -s -X POST "${BASE_URL}/contacts/${CONTACT_2_ID}/resume" "${AUTH_ARGS[@]}" | pp

# ============================================================================
# 14. Mark a contact as closed-won
# ============================================================================
section "14. POST /contacts/$CONTACT_2_ID/mark-won"
curl -s -X POST "${BASE_URL}/contacts/${CONTACT_2_ID}/mark-won" "${AUTH_ARGS[@]}" | pp

# ============================================================================
# 15. Suppression: add, check, and prove re-adding is blocked
# ============================================================================
section "15. POST /suppress + GET /suppress/{email}"
SUPPRESS_EMAIL="suppress-test@example.com"
curl -s -X POST "${BASE_URL}/suppress" "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg email "$SUPPRESS_EMAIL" '{email:$email, reason:"manual test"}')" | pp
curl -s "${BASE_URL}/suppress/${SUPPRESS_EMAIL}" "${AUTH_ARGS[@]}" | pp

echo
echo "Trying to add the now-suppressed address as a contact (should come back empty):"
curl -s -X POST "${BASE_URL}/campaigns/${CAMPAIGN_ID}/contacts" "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg email "$SUPPRESS_EMAIL" '{contacts: [{email:$email}]}')" | pp

# ============================================================================
# Done
# ============================================================================
section "DONE"
echo "Campaign ID: $CAMPAIGN_ID"
echo "Contact 1 ID: $CONTACT_1_ID  (thread: GET ${BASE_URL}/contacts/${CONTACT_1_ID}/thread)"
echo "Contact 2 ID: $CONTACT_2_ID  (thread: GET ${BASE_URL}/contacts/${CONTACT_2_ID}/thread)"
echo
echo "If LIVE_SENDING_ENABLED=false on the server (the default), nothing above"
echo "actually sent an email -- check the server's own console/log output to see"
echo "what was generated and logged as '[DRY RUN] Would send to ...'."
