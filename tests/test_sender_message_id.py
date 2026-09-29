"""
send_email(message_id_header=...) -- the only sender.py change C7 needs.

Purpose: let a caller persist a Message-ID BEFORE the SMTP call and then
transmit that exact string, so stored ID == transmitted ID. This closes the
ID mismatch only; it does not make delivery exactly-once.

Uses the deterministic in-process FakeSMTPServer (ephemeral loopback port,
no external network).
"""

from __future__ import annotations

import re

import pytest

from mailer_agent.mail import sender
from tests.dispatch_support import FakeSMTPServer

KW = dict(
    to_email="prospect@example.com",
    from_email="outreach@sender.example.org",
    from_name="Outreach",
    subject="Hello",
    body_text="Body line 1\nBody line 2\n",
)


@pytest.fixture()
def live(monkeypatch):
    with FakeSMTPServer() as srv:
        s = sender.settings
        monkeypatch.setattr(s, "smtp_host", "127.0.0.1")
        monkeypatch.setattr(s, "smtp_port", srv.port)
        monkeypatch.setattr(s, "smtp_use_tls", False)
        monkeypatch.setattr(s, "smtp_username", "u")
        monkeypatch.setattr(s, "smtp_password", "p")
        monkeypatch.setattr(s, "live_sending_enabled", True)
        yield srv


def _header(raw: str, name: str) -> str:
    m = re.search(rf"^{name}: (.*)$", raw, re.MULTILINE | re.IGNORECASE)
    assert m, f"{name} header missing from wire message"
    return m.group(1).strip()


def test_supplied_message_id_is_the_one_on_the_wire_and_in_the_result(live):
    supplied = "<persisted-before-smtp.abc123@sender.example.org>"
    result = sender.send_email(**KW, message_id_header=supplied)

    assert result.outcome == sender.SendOutcome.SENT
    assert result.message_id == supplied
    assert len(live.accepted) == 1
    assert _header(live.accepted[0], "Message-ID") == supplied


def test_supplied_id_is_not_regenerated(live, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("make_msgid must not be called when an ID is supplied")

    monkeypatch.setattr(sender, "make_msgid", boom)
    result = sender.send_email(**KW, message_id_header="<x@y.example>")
    assert result.message_id == "<x@y.example>"


def test_legacy_call_without_argument_still_mints_its_own_id(live):
    r1 = sender.send_email(**KW)
    r2 = sender.send_email(**KW)

    assert r1.outcome == r2.outcome == sender.SendOutcome.SENT
    assert re.fullmatch(r"<[^<>@\s]+@sender\.example\.org>", r1.message_id)
    assert r1.message_id != r2.message_id
    assert _header(live.accepted[0], "Message-ID") == r1.message_id
    assert _header(live.accepted[1], "Message-ID") == r2.message_id


def test_generate_message_id_uses_the_sender_domain_rule():
    assert generate_domain("a@b.example.org") == "b.example.org"
    assert generate_domain("no-at-sign") == "localhost"


def generate_domain(from_email: str) -> str:
    return sender.generate_message_id(from_email).rstrip(">").split("@")[-1]


@pytest.mark.parametrize(
    "bad",
    ["", "no-brackets@x", "<a@b>\r\nBcc: evil@x", "<a@b>\nX: y", "<a b@c>", "<a\x00@b>", "<>", "  "],
)
def test_invalid_supplied_id_is_rejected_before_any_network(live, bad):
    with pytest.raises(ValueError):
        sender.send_email(**KW, message_id_header=bad)
    assert live.connections == 0 and live.accepted == []


def test_dry_run_honours_supplied_id_and_never_touches_the_network(monkeypatch):
    monkeypatch.setattr(sender.settings, "live_sending_enabled", False)
    with FakeSMTPServer() as srv:
        monkeypatch.setattr(sender.settings, "smtp_host", "127.0.0.1")
        monkeypatch.setattr(sender.settings, "smtp_port", srv.port)
        result = sender.send_email(**KW, message_id_header="<dry@x.example>")
        assert result.message_id == "<dry@x.example>"
        assert srv.connections == 0
