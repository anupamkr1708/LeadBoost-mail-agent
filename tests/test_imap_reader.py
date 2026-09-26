"""
Regression tests for mail/imap_reader.py's fetch_unseen_replies().

Root bug this guards against: the original implementation fetched with
"(RFC822)", which marks a message \\Seen as a side effect of the fetch
itself -- before any parsing happens. If parsing failed for one message
in a batch, that message was already \\Seen on the server and would
never be offered by a future UNSEEN search again: a silent, permanent
loss of exactly the message that most needed a retry. The fix fetches
with BODY.PEEK[] (no implicit Seen) and only issues an explicit STORE
+FLAGS \\Seen after that specific message was successfully parsed.

This uses a mocked imaplib.IMAP4_SSL rather than a real IMAP server --
unlike the SMTP path (tests/test_smtp_local_integration.py), no simple
local test server is readily available for IMAP, so this proves the
calling-convention/control-flow logic (what gets fetched, what gets
stored, what survives a per-message exception) rather than a full live
protocol round-trip. Flagged as such in the final report -- this is
IMPLEMENTED and logic-verified, not verified against a real IMAP
server the way the SMTP fix was.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mailer_agent.mail import imap_reader


def _raw_message(*, subject="Hello", from_addr="prospect@example.com", body="Some reply text.", message_id="<abc@example.com>"):
    return (
        f"From: {from_addr}\r\n"
        f"Subject: {subject}\r\n"
        f"Message-ID: {message_id}\r\n"
        f"Content-Type: text/plain\r\n"
        f"\r\n"
        f"{body}\r\n"
    ).encode("utf-8")


@pytest.fixture(autouse=True)
def imap_credentials(monkeypatch):
    monkeypatch.setattr(imap_reader.settings, "imap_username", "test@example.com")
    monkeypatch.setattr(imap_reader.settings, "imap_password", "test-password")


def _mock_conn(*, uids, fetch_side_effect):
    """Build a MagicMock standing in for imaplib.IMAP4_SSL(...)."""
    conn = MagicMock()
    conn.search.return_value = ("OK", [b" ".join(uids)])
    conn.fetch.side_effect = fetch_side_effect
    return conn


def test_uses_peek_not_rfc822_so_fetch_itself_never_marks_seen():
    conn = _mock_conn(
        uids=[b"1"],
        fetch_side_effect=[("OK", [(b"1 (BODY.PEEK[])", _raw_message())])],
    )
    with patch("imaplib.IMAP4_SSL", return_value=conn):
        results = imap_reader.fetch_unseen_replies()

    assert len(results) == 1
    fetch_item_used = conn.fetch.call_args_list[0].args[1]
    assert "PEEK" in fetch_item_used, (
        "Must fetch with BODY.PEEK[], not RFC822 -- RFC822 marks \\Seen "
        "as a side effect of the fetch, before parsing has even happened."
    )


def test_successfully_parsed_message_is_explicitly_marked_seen():
    conn = _mock_conn(
        uids=[b"1"],
        fetch_side_effect=[("OK", [(b"1 (BODY.PEEK[])", _raw_message())])],
    )
    with patch("imaplib.IMAP4_SSL", return_value=conn):
        results = imap_reader.fetch_unseen_replies()

    assert len(results) == 1
    conn.store.assert_called_once_with(b"1", "+FLAGS", "\\Seen")


def test_one_malformed_message_does_not_lose_or_seen_mark_it(caplog):
    """
    The core regression: uid 1 fails to parse (fetch returns garbage
    that raises), uid 2 is a normal, parseable message. Uid 1 must NOT
    be marked \\Seen (so it's retried next poll) and must not crash
    processing of uid 2.
    """
    def fetch_side_effect(uid, _item):
        if uid == b"1":
            # Simulate a fetch that returns something malformed enough
            # to blow up during parsing (None payload where bytes are
            # expected raises inside email.message_from_bytes).
            return ("OK", [(b"1 (BODY.PEEK[])", None)])
        return ("OK", [(b"2 (BODY.PEEK[])", _raw_message(message_id="<msg2@example.com>"))])

    conn = _mock_conn(uids=[b"1", b"2"], fetch_side_effect=fetch_side_effect)
    with patch("imaplib.IMAP4_SSL", return_value=conn):
        results = imap_reader.fetch_unseen_replies()

    assert len(results) == 1
    assert results[0].message_id == "<msg2@example.com>"
    # uid 1 must never have been marked Seen.
    seen_calls = [c for c in conn.store.call_args_list if c.args[0] == b"1"]
    assert seen_calls == [], "A message that failed to parse must not be marked \\Seen -- it needs to be retried."
    # uid 2 (the one that succeeded) must have been marked Seen.
    conn.store.assert_any_call(b"2", "+FLAGS", "\\Seen")


def test_connection_is_still_closed_after_a_per_message_failure():
    def fetch_side_effect(uid, _item):
        return ("OK", [(b"1 (BODY.PEEK[])", None)])  # will raise during parse

    conn = _mock_conn(uids=[b"1"], fetch_side_effect=fetch_side_effect)
    with patch("imaplib.IMAP4_SSL", return_value=conn):
        results = imap_reader.fetch_unseen_replies()

    assert results == []
    conn.close.assert_called_once()
    conn.logout.assert_called_once()


def test_multiple_messages_all_parsed_and_all_marked_seen():
    def fetch_side_effect(uid, _item):
        return ("OK", [(f"{uid.decode()} (BODY.PEEK[])".encode(), _raw_message(message_id=f"<msg{uid.decode()}@example.com>"))])

    conn = _mock_conn(uids=[b"1", b"2", b"3"], fetch_side_effect=fetch_side_effect)
    with patch("imaplib.IMAP4_SSL", return_value=conn):
        results = imap_reader.fetch_unseen_replies()

    assert len(results) == 3
    assert conn.store.call_count == 3
