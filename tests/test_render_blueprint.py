"""
Deployment guard: render.yaml must provision MAILBOX_ENCRYPTION_KEY for every
service that touches mailbox credentials.

Mailer M1/M2 deliberately fail closed when the key is missing: the API cannot
create/update a mailbox (503) and the dispatch worker cannot decrypt a
credential to send. A blueprint that omits the key for either service deploys
a system that accepts work it can never perform. Plain-text parsing on
purpose -- no YAML dependency is declared in requirements.txt.
"""

from __future__ import annotations

import re
from pathlib import Path

RENDER_YAML = Path(__file__).resolve().parents[1] / "render.yaml"


def _service_blocks() -> dict[str, str]:
    text = RENDER_YAML.read_text(encoding="utf-8")
    services_part = text.split("\nservices:", 1)[1]
    blocks = re.split(r"\n  - type: ", services_part)[1:]
    out = {}
    for block in blocks:
        name = re.search(r"^\s*name:\s*(\S+)", block, re.MULTILINE).group(1)
        out[name] = block
    return out


def test_blueprint_declares_both_services():
    assert {"mailer-agent-api", "mailer-agent-worker"} <= set(_service_blocks())


def test_mailbox_encryption_key_is_provisioned_for_api_and_worker_without_a_value():
    blocks = _service_blocks()
    for name in ("mailer-agent-api", "mailer-agent-worker"):
        block = blocks[name]
        m = re.search(r"- key: MAILBOX_ENCRYPTION_KEY\s*\n\s*sync: false", block)
        assert m, f"{name} must declare MAILBOX_ENCRYPTION_KEY with sync: false"
        # A secret must never be committed in the blueprint.
        assert not re.search(r"MAILBOX_ENCRYPTION_KEY\s*\n\s*value:", block)
