#!/usr/bin/env bash
#
# Run E2E test with environment loaded from .env
#
# Usage: ./run_e2e_test.sh

set -euo pipefail

# Load environment from .env
if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Copy .env.example to .env and fill it in."
    exit 1
fi

# Export all variables from .env (skip comments and empty lines)
export $(grep -v '^#' .env | grep -v '^$' | xargs)

# Show what we're using (without exposing passwords)
echo "=========================================="
echo "Running E2E Test with Configuration:"
echo "=========================================="
echo "API_KEY: ${API_KEY:0:20}... (${#API_KEY} chars)"
echo "SENDER_EMAIL: ${SMTP_USERNAME}"
echo "BASE_URL: ${BASE_URL:-http://localhost:8000}"
echo "LIVE_SENDING_ENABLED: ${LIVE_SENDING_ENABLED}"
echo "AUTO_REPLY_ENABLED: ${AUTO_REPLY_ENABLED}"
echo ""

# Override test-specific settings if needed
export SENDER_NAME="${SENDER_NAME:-Test Sender}"
export SENDER_ORG="${SENDER_ORG:-Test Organization}"
export SENDER_EMAIL="${SMTP_USERNAME}"  # Use actual SMTP username
export REPLY_TO_EMAIL="${SMTP_USERNAME}"

# Use real email for testing (or override)
export CONTACT_1_NAME="${CONTACT_1_NAME:-Test Contact}"
export CONTACT_1_EMAIL="${CONTACT_1_EMAIL:-${SMTP_USERNAME}}"  # Send to self for testing
export CONTACT_1_TITLE="${CONTACT_1_TITLE:-Test Manager}"
export CONTACT_1_COMPANY="${CONTACT_1_COMPANY:-Test Company}"

export LEADBOOST_LEAD_EMAIL="${LEADBOOST_LEAD_EMAIL:-${SMTP_USERNAME}}"

echo "=========================================="
echo "Test will send emails to:"
echo "  - ${CONTACT_1_EMAIL}"
echo "  - ${LEADBOOST_LEAD_EMAIL}"
echo ""
echo "IMPORTANT: These emails will be REALLY SENT"
echo "since LIVE_SENDING_ENABLED=${LIVE_SENDING_ENABLED}"
echo ""
echo "Press CTRL+C to cancel, or wait 5 seconds to continue..."
echo "=========================================="
sleep 5

# Run the actual test script
./test_e2e.sh
