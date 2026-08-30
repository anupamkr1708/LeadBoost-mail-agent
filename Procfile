web: gunicorn mailer_agent.api.main:app -k uvicorn.workers.UvicornWorker -w 2 --bind 0.0.0.0:$PORT --timeout 60
worker: python worker.py
