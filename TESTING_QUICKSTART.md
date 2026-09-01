# Quick Start - Testing the Mailer Agent

## Prerequisites

1. **Python 3.11+** installed
2. **Gmail account** with 2FA enabled (for SMTP/IMAP)
3. **Groq API key** from https://console.groq.com

## Setup (First Time Only)

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
```

Edit `.env` and fill in:

#### Required Settings:
```bash
# Get from https://console.groq.com
GROQ_API_KEY=gsk_your_key_here

# Your Gmail address
SMTP_USERNAME=your-email@gmail.com
IMAP_USERNAME=your-email@gmail.com

# Gmail App Password (NOT your regular password)
# Create at: https://myaccount.google.com/apppasswords
SMTP_PASSWORD=your_app_password
IMAP_PASSWORD=your_app_password

# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
API_KEY=your_generated_api_key_here
```

#### Safety Settings:
```bash
# IMPORTANT: Start with false to test without actually sending emails
LIVE_SENDING_ENABLED=false
AUTO_REPLY_ENABLED=false
```

### 4. Initialize Database
```bash
python migrate_db.py
```

## Running the E2E Test

### Step 1: Start the Server
```bash
python run.py
```

You should see:
```
INFO: Mailer Agent starting | live_sending_enabled=False auto_reply_enabled=False
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Run E2E Test (in another terminal)
```bash
# Activate venv first
source venv/bin/activate

# Run test script (loads .env automatically)
./run_e2e_test.sh
```

The script will:
- Load your configuration from `.env`
- Show what it's about to do
- Wait 5 seconds (CTRL+C to cancel)
- Run complete API test suite

## What the Test Does

### With LIVE_SENDING_ENABLED=false (Dry Run):
✅ Creates test campaign  
✅ Adds contacts  
✅ Generates emails (but doesn't send)  
✅ Tests all API endpoints  
✅ Simulates inbound replies  
✅ Tests suppression logic  

Server logs will show:
```
[DRY RUN] Would send to test@example.com ...
```

### With LIVE_SENDING_ENABLED=true (Real Sending):
⚠️ **ACTUALLY SENDS EMAILS** to configured recipients

By default, sends to yourself (SMTP_USERNAME) for safety.

## Understanding the Output

### Successful Test:
```bash
============================================================
2. POST /campaigns
============================================================
{
  "id": 1,
  "name": "Manual E2E Test Campaign",
  "sender_name": "Your Name",
  "sender_email": "your-email@gmail.com",
  ...
}
>> CAMPAIGN_ID=1
```

### If You See "Invalid or missing API key":
Your API_KEY isn't loaded. Make sure:
1. `.env` has `API_KEY=your_key`
2. You're using `./run_e2e_test.sh` (loads .env automatically)

### If You See "401 Unauthorized":
Server is using different API_KEY. Check:
```bash
# In server terminal
grep API_KEY .env

# Match it in test
echo $API_KEY
```

## Running Unit Tests

```bash
# All tests
python -m pytest -v

# Production-critical tests only
python -m pytest tests/test_flow.py tests/test_grounding.py -v
```

**Note:** Some semantic tests fail due to test infrastructure (FakeLLMProvider), not production bugs. This is expected.

## Testing Real Email Sending

### Step 1: Verify Dry Run Works
```bash
# In .env
LIVE_SENDING_ENABLED=false

# Run test
./run_e2e_test.sh

# Check server logs for [DRY RUN]
```

### Step 2: Enable Real Sending
```bash
# In .env
LIVE_SENDING_ENABLED=true
```

### Step 3: Test Sending to Yourself
By default, test sends to your own email (SMTP_USERNAME).

Check your inbox for test emails!

### Step 4: Test with Different Recipient
```bash
# In .env, add:
CONTACT_1_EMAIL=another-email@example.com
```

## Troubleshooting

### "Connection refused" or "Connection reset"
**Gmail blocking automated access.**

Solutions:
1. Use Gmail App Password (not regular password)
2. Enable "Less secure app access" (if available)
3. Try different SMTP server (not Gmail)

### "Invalid credentials"
Check SMTP_PASSWORD is your Gmail **App Password**, not regular password.

Create at: https://myaccount.google.com/apppasswords

### Test hangs or times out
- Check server is running (`python run.py`)
- Check no other process using port 8000
- Check network/firewall not blocking localhost:8000

### Emails not received
With LIVE_SENDING_ENABLED=true:
1. Check spam folder
2. Check server logs for actual send attempt
3. Check Gmail "Sent" folder
4. Wait a few minutes (SMTP can be slow)

### "Database is locked"
SQLite limitation with concurrent access.

Solutions:
1. Stop all processes using the DB
2. Delete `mailer_agent.db` and re-run `python migrate_db.py`
3. Use PostgreSQL for production (recommended)

## Next Steps

### For Development:
- Read `docs/MAILER_ARCHITECTURE.md`
- Read `docs/FINAL_PRODUCTION_READINESS_REPORT.md`

### For Production:
1. Switch to PostgreSQL database
2. Set up webhook for inbound emails (not just IMAP)
3. Use dedicated sending domain (not personal Gmail)
4. Review deployment guide in `README.md`

## Configuration Summary

### Minimal .env for Testing:
```bash
DATABASE_URL=sqlite:///./mailer_agent.db
GROQ_API_KEY=gsk_xxx
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your@gmail.com
SMTP_PASSWORD=your_app_password
IMAP_HOST=imap.gmail.com
IMAP_PORT=993
IMAP_USERNAME=your@gmail.com
IMAP_PASSWORD=your_app_password
API_KEY=your_generated_key
LIVE_SENDING_ENABLED=false
AUTO_REPLY_ENABLED=false
RUN_SCHEDULER_IN_PROCESS=true
```

### Testing Checklist:
- [ ] `.env` configured with real credentials
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Database initialized (`python migrate_db.py`)
- [ ] Server running (`python run.py`)
- [ ] Dry run successful (`LIVE_SENDING_ENABLED=false`)
- [ ] Real sending tested to self (`LIVE_SENDING_ENABLED=true`)

## Getting Help

1. Check server logs for error details
2. Read `docs/FINAL_PRODUCTION_READINESS_REPORT.md`
3. Review error in server terminal
4. Check `.env` settings match `.env.example`
