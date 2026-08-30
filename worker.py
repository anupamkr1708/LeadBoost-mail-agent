"""
Standalone worker process: runs the reply-poll and follow-up/dispatch
jobs on their own schedule, with no HTTP server attached.

Deploy this as a Render "Background Worker" (start command: `python
worker.py`) alongside a separate "Web Service" running the FastAPI app
(start command: see Procfile). Keeping them as two processes means the
API can scale (multiple gunicorn workers) without ever running the
scheduler more than once -- if the scheduler ran inside every gunicorn
worker process, you'd get duplicate sends and duplicate reply
processing.

For local development, you don't need this at all: `python run.py`
with the default RUN_SCHEDULER_IN_PROCESS=true runs everything in one
process.
"""

import logging
import time

from mailer_agent.db import init_db
from mailer_agent.followup.scheduler import start_scheduler, stop_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mailer_agent.worker")

if __name__ == "__main__":
    init_db()
    logger.info("Starting Mailer Agent background worker (scheduler-only, no HTTP server)")
    scheduler = start_scheduler()
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down background worker")
        stop_scheduler()
