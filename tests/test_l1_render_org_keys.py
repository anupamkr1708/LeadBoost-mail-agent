"""L1: the API service must be able to receive the per-tenant key map (API only)."""
import re
from pathlib import Path

TEXT = (Path(__file__).resolve().parents[1] / "render.yaml").read_text(encoding="utf-8")


def _block(name):
    blocks = re.split(r"\n  - type: ", TEXT.split("\nservices:", 1)[1])[1:]
    return next(b for b in blocks if re.search(rf"^\s*name:\s*{name}\b", b, re.MULTILINE))


def test_org_key_map_declared_on_api_without_a_value_and_not_on_worker():
    api = _block("mailer-agent-api")
    assert re.search(r"- key: ORG_KEY_MAP\s*\n\s*sync: false", api)
    assert not re.search(r"ORG_KEY_MAP\s*\n\s*value:", api)
    assert "ORG_KEY_MAP" not in _block("mailer-agent-worker")   # workers never authenticate callers


def test_leadboost_integration_sender_identity_is_declared_on_the_api_without_a_value():
    """Without it the REAL Mailer answers 503 to the first LeadBoost dispatch of every organization
    (found in the L1 cross-service E2E). It must be provisioned, never committed."""
    api = _block("mailer-agent-api")
    assert re.search(r"- key: LEADBOOST_INTEGRATION_SENDER_EMAIL\s*\n\s*sync: false", api)
    assert not re.search(r"LEADBOOST_INTEGRATION_SENDER_EMAIL\s*\n\s*value:", api)
