"""
Standalone worker process: runs the reply-poll and follow-up/dispatch
jobs on their own schedule, with no HTTP server attached.

Deploy this as a Render "Background Worker" (start command: ``python
worker.py``) alongside a separate "Web Service" running the FastAPI app
(start command: see Procfile). Keeping them as two processes means the
API can scale (multiple gunicorn workers) without ever running the
scheduler more than once -- if the scheduler ran inside every gunicorn
worker process, you'd get duplicate sends and duplicate reply
processing.

For local development, you don't need this at all: ``python run.py``
with the default RUN_SCHEDULER_IN_PROCESS=true runs everything in one
process.

Phase 7 — Scheduler ownership
------------------------------
The scheduler is deliberately split from the API process:

* ``run.py`` / ``mailer_agent/api/main.py`` lifespan handler:
  Only starts the scheduler when ``RUN_SCHEDULER_IN_PROCESS=true``
  (the default for single-process local dev).  When set to false (the
  recommended production setting) the API process logs a warning and
  never starts any APScheduler jobs -- it is purely an HTTP server.

* ``worker.py`` (this file):
  Always starts the scheduler.  In production, exactly one instance of
  this process should run (Render Background Worker, or a single
  container replica).  Having exactly one scheduler process means there
  are never duplicate reply-poll or follow-up-dispatch cycles from the
  scheduler side.

* Residual duplicate-work risk:
  Even with one scheduler, if a job cycle takes longer than its
  interval the next run begins while the previous one is still active.
  APScheduler's ``max_instances=1`` per job prevents this (any
  overlapping trigger is silently skipped).  For the remaining
  concurrent-worker case (two ``worker.py`` processes running
  simultaneously, e.g. during a rolling deploy), Phase 8's database-
  level work claiming (SELECT FOR UPDATE SKIP LOCKED) is the defence.

Phase 8 — Graceful shutdown and claim release
----------------------------------------------
On shutdown (KeyboardInterrupt / SIGTERM), this process releases any
work claims it currently holds so the next worker cycle (possibly on a
newly-started worker) can pick them up immediately rather than waiting
for the lease to expire (default 300 s).
"""

import logging
import os
import time

from mailer_agent.db import init_db, session_scope
from mailer_agent.followup.scheduler import start_scheduler, stop_scheduler
from mailer_agent.followup.work_claiming import make_worker_id, release_all_claims

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mailer_agent.worker")

if __name__ == "__main__":
    init_db()

    worker_id = make_worker_id()
    # Expose worker_id in env so child scheduler jobs inherit it automatically.
    os.environ.setdefault("WORKER_ID", worker_id)

    logger.info(
        "Starting Mailer Agent background worker (scheduler-only, no HTTP server) "
        "worker_id=%s",
        worker_id,
    )

    scheduler = start_scheduler()
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down background worker (worker_id=%s)", worker_id)
        stop_scheduler()

        # Phase 8: Release any DB claims so the next worker can pick them up
        # immediately rather than waiting for lease expiry.
        try:
            with session_scope() as db:
                released = release_all_claims(db, worker_id)
                if released:
                    logger.info("Released %d claim(s) on graceful shutdown", released)
        except Exception as exc:  # noqa: BLE001 — best-effort, don't mask shutdown
            logger.warning("Could not release claims on shutdown: %s", exc)
