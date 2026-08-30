"""
Entrypoint: `python run.py`

Starts the FastAPI app (with its background scheduler, wired via the
lifespan handler in mailer_agent/api/main.py) using uvicorn.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("mailer_agent.api.main:app", host="0.0.0.0", port=8000, reload=False)
