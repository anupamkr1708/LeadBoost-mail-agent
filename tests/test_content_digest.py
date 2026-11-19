"""
Exact-content digest (mail/content_digest.py): deterministic vectors, boundary-ambiguity resistance and fail-closed
behaviour. The expected hex values were computed independently of the module -- literal canonical bytes through
hashlib -- so these tests cannot merely agree with the implementation.
"""

from __future__ import annotations

import hashlib

import pytest

from mailer_agent.mail import content_digest as cd

VECTORS = [
    ("Re: Hi", "Hello", b'{"body":"Hello","subject":"Re: Hi"}',
     "2dc399390aa6f2d862527317573c57749374d5301b2881bda0d54161cb0872f1"),
    (None, "Hello", b'{"body":"Hello","subject":""}',
     "956e4bd61170a8556a481ba2e333273a2c1619d1c271f65de61e8f7e07ad3458"),
    ("Re: café", "Grüße ☃\n", '{"body":"Grüße ☃\\n","subject":"Re: café"}'.encode("utf-8"),
     "ea4877840c0cd4337f0b5f1cf4b2d0619fac76f124037370a3766c023b3e2159"),
]


@pytest.mark.parametrize("subject, body, canonical, expected", VECTORS)
def test_deterministic_vectors(subject, body, canonical, expected):
    assert cd.canonical_bytes(subject, body) == canonical
    assert hashlib.sha256(canonical).hexdigest() == expected      # the vector itself is self-consistent
    assert cd.content_sha256(subject, body) == expected
    assert cd.is_valid_digest(expected)


def test_none_subject_and_empty_subject_are_the_same_content():
    # exact_message.build_exact_send_input hands "" to the sender for a None subject.
    assert cd.content_sha256(None, "x") == cd.content_sha256("", "x")


def test_field_boundaries_cannot_be_shifted():
    assert cd.content_sha256("a", "bc") != cd.content_sha256("ab", "c")
    assert cd.content_sha256("a\nb", "") != cd.content_sha256("a", "b")


def test_no_newline_normalisation():
    assert cd.content_sha256("s", "a\nb") != cd.content_sha256("s", "a\r\nb")
    assert cd.content_sha256("s", "a\nb") == "85fa6a330bdc6459c7116f98b464b72035224a64ef7c72f0c6b355412ebc9bd1"
    assert cd.content_sha256("s", "a\r\nb") == "9e9329a966c6f22f4b2641bce649acc10e85fd5137a56f6896c46eb152431b3b"


def test_one_character_change_changes_the_digest():
    assert cd.content_sha256("Re: Hi", "Hello") != cd.content_sha256("Re: Hi", "Hello ")
    assert cd.content_sha256("Re: Hi", "Hello") != cd.content_sha256("Re: hi", "Hello")


def test_unicode_normal_forms_are_distinct():
    assert cd.content_sha256("s", "\u00e9") != cd.content_sha256("s", "e\u0301")


@pytest.mark.parametrize("bad", ["\ud800", "ok\udfff"])
def test_unencodable_text_fails_closed(bad):
    with pytest.raises(cd.DigestError):
        cd.content_sha256("s", bad)
    with pytest.raises(cd.DigestError):
        cd.content_sha256(bad, "b")


@pytest.mark.parametrize("subject, body", [(1, "b"), ("s", None), ("s", b"bytes")])
def test_wrong_types_fail_closed(subject, body):
    with pytest.raises(cd.DigestError):
        cd.content_sha256(subject, body)


@pytest.mark.parametrize("value", [None, "", "abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65, 12, b"a" * 64])
def test_is_valid_digest_rejects_everything_but_64_lowercase_hex(value):
    assert cd.is_valid_digest(value) is False


def test_digests_equal_is_exact_and_never_true_for_missing_or_malformed():
    d = cd.content_sha256("s", "b")
    assert cd.digests_equal(d, d) is True
    assert cd.digests_equal(d, d[:-1] + ("0" if d[-1] != "0" else "1")) is False
    assert cd.digests_equal(None, d) is False and cd.digests_equal(d, None) is False
    assert cd.digests_equal(d.upper(), d) is False
    assert cd.digests_equal("", "") is False and cd.digests_equal(None, None) is False


def test_preview_limits_refuse_rather_than_truncate():
    cd.enforce_preview_limits("s" * cd.MAX_SUBJECT_CHARS, "b" * cd.MAX_BODY_CHARS)       # exactly at the limit: fine
    with pytest.raises(cd.DraftTooLarge):
        cd.enforce_preview_limits("s", "b" * (cd.MAX_BODY_CHARS + 1))
    with pytest.raises(cd.DraftTooLarge):
        cd.enforce_preview_limits("s" * (cd.MAX_SUBJECT_CHARS + 1), "b")
    assert issubclass(cd.DraftTooLarge, cd.DigestError)
