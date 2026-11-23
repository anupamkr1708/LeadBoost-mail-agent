"""
C9.5 hardening (a) -- the reply-draft routes accept an idempotency_key that contains "/".

    GET  /integrations/leadboost/reply-drafts/{idempotency_key:path}
    POST /integrations/leadboost/reply-drafts/{idempotency_key:path}/release
    POST /integrations/leadboost/reply-drafts/{idempotency_key:path}/cancel

LeadBoost's own keys are ``reply:`` + 40 hex and never contain "/", but a direct API caller may choose any key and
LeadBoost's client percent-encodes it with ``quote(key, safe="")``. The server decodes ``%2F`` BEFORE routing, so
with single-segment declarations a key containing "/" matched no route at all (the router's generic 404) and a
held draft under such a key could never be read, released or cancelled -- it would sit in AWAITING_APPROVAL forever.

A ``:path`` converter is greedy, so this file also pins the routing it must not disturb: which handler a path
reaches, that a mutating route can only be reached with its own suffix, that no other route in the app can match
these URLs (or be shadowed by them), and the wrong-method / edge-shape answers (see the "deliberate changes"
section: they are the unavoidable consequence of the greedy converter and are asserted exactly).

Isolation follows tests/test_managed_reply_content_approval.py: per-test SQLite file, dependency overrides, no
SMTP, no LLM. Nothing here sends mail (release only moves a held row into the existing sendable queue).
"""

from __future__ import annotations

import itertools
import re
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.routing import Match

from mailer_agent.api.deps import get_current_org_id, get_integration_org_id, require_api_key
from mailer_agent.api.main import app
from mailer_agent.db import get_db
from mailer_agent.mail import content_digest as cd
from mailer_agent.models import Base, ExternalDispatch, ExternalDispatchState as S
from tests.dispatch_support import seed_dispatch, seed_org_mailboxes

ORG_A, ORG_B = "org-a", "org-b"
DRAFTS = "/integrations/leadboost/reply-drafts"
NOT_FOUND = {"detail": "Reply draft not found"}
LEGACY_CODE = "legacy_reply_requires_reconciliation"
SENDER = "outreach@mailer.example.com"

HEX40 = "0123456789abcdef" * 2 + "01234567"
LB_KEY = "reply:" + HEX40            # the only shape LeadBoost itself ever produces

# Keys a direct API caller could choose. Each is sent percent-encoded exactly as LeadBoost's client does.
SLASH_KEYS = [
    pytest.param("a/b", id="one-slash"),
    pytest.param("a/b/c", id="two-slashes"),
    pytest.param("reply:" + HEX40 + "/extra", id="leadboost-shape-plus-slash"),
    pytest.param("x/release", id="ends-in-release"),
    pytest.param("x/cancel", id="ends-in-cancel"),
    pytest.param("x/release/cancel", id="release-then-cancel"),
    pytest.param("x/cancel/release", id="cancel-then-release"),
    pytest.param("trailing/", id="trailing-slash"),
    pytest.param("/leading", id="leading-slash"),
    pytest.param("a b/c?d#e%f", id="space-query-fragment-percent"),
]
# A key that IS a route word, and the unchanged LeadBoost shape: neither contains "/" but both sit next to the
# suffix routes, so they prove the greedy converter did not change what a plain key reaches.
PLAIN_KEYS = [
    pytest.param(LB_KEY, id="leadboost-shape"),
    pytest.param("reply-1", id="simple"),
    pytest.param("release", id="equals-route-word-release"),
    pytest.param("cancel", id="equals-route-word-cancel"),
]
ALL_KEYS = PLAIN_KEYS + SLASH_KEYS


class _Who:
    org = ORG_A


def enc(key: str) -> str:
    """The key exactly as LeadBoost's reply_draft_client sends it."""
    return quote(key, safe="")


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def sm(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'kp.db'}", connect_args={"check_same_thread": False, "timeout": 15})
    Base.metadata.create_all(bind=eng)
    sm_ = sessionmaker(bind=eng)
    with sm_() as s:
        seed_org_mailboxes(s)
    yield sm_
    eng.dispose()


@pytest.fixture()
def who():
    w = _Who()
    w.org = ORG_A
    return w


def _install_overrides(sm, who):
    def _get_db():
        s = sm()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[get_integration_org_id] = lambda: who.org
    app.dependency_overrides[get_current_org_id] = lambda: who.org


@pytest.fixture()
def client(sm, who):
    _install_overrides(sm, who)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def raw_client(sm, who):
    """Same app, but redirects are not followed (to see what a bare shape really answers)."""
    _install_overrides(sm, who)
    try:
        yield TestClient(app, follow_redirects=False)
    finally:
        app.dependency_overrides.clear()


# ------------------------------------------------------------------ helpers

_emails = itertools.count(1)


def hold(sm, key, *, org=ORG_A):
    """One held C9.5 reply draft (AWAITING_APPROVAL, flag true, persisted DRAFT message). The body embeds the key
    so every row has its OWN digest: a request that reached the wrong row cannot pass for the right one."""
    with sm() as s:
        seed_dispatch(
            s, org=org, idem=key, email=f"lead{next(_emails)}@example.com",
            state=S.AWAITING_APPROVAL.value, content_approval_required=True,
            body=f"Hello, this draft belongs to {key!r}.\n", sender_email=SENDER,
        )


def row(sm, key, *, org=ORG_A):
    with sm() as s:
        d = s.query(ExternalDispatch).filter_by(organization_id=org, idempotency_key=key).one()
        return {"state": d.state, "approved": d.approved_content_sha256, "approved_at": d.approved_at}


def states(sm, *keys, org=ORG_A):
    return {k: row(sm, k, org=org)["state"] for k in keys}


def digest_of(client, key):
    r = client.get(f"{DRAFTS}/{enc(key)}")
    assert r.status_code == 200, r.text
    return r.json()["content_sha256"]


def release(client, key, digest, *, quoted=True):
    return client.post(f"{DRAFTS}/{enc(key) if quoted else key}/release", json={"content_sha256": digest})


def cancel(client, key, *, quoted=True):
    return client.post(f"{DRAFTS}/{enc(key) if quoted else key}/cancel")


def snapshot_all(sm):
    with sm() as s:
        return sorted(
            (d.organization_id, d.idempotency_key, d.state, d.approved_content_sha256, str(d.updated_at))
            for d in s.query(ExternalDispatch).all()
        )


# ====================================================== 1. the three routes reach the row, whatever the key


@pytest.mark.parametrize("key", ALL_KEYS)
def test_a_held_draft_is_readable_under_any_key(sm, client, key):
    hold(sm, key)
    r = client.get(f"{DRAFTS}/{enc(key)}")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["state"] == "awaiting_approval"
    assert re.fullmatch(r"[0-9a-f]{64}", j["content_sha256"])
    assert j["body"] == f"Hello, this draft belongs to {key!r}.\n"
    assert j["content_sha256"] == cd.content_sha256(j["subject"], j["body"])


@pytest.mark.parametrize("key", ALL_KEYS)
def test_a_held_draft_is_releasable_under_any_key_and_the_replay_is_idempotent(sm, client, key):
    hold(sm, key)
    digest = digest_of(client, key)
    r = release(client, key, digest)
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "queued"
    got = row(sm, key)
    assert got["state"] == S.QUEUED.value and got["approved"] == digest and got["approved_at"] is not None
    again = release(client, key, digest)                     # same digest after release: 200, nothing changes
    assert again.status_code == 200 and again.json()["state"] == "queued"
    assert row(sm, key) == got


@pytest.mark.parametrize("key", ALL_KEYS)
def test_a_held_draft_is_cancellable_under_any_key_and_the_replay_is_idempotent(sm, client, key):
    hold(sm, key)
    r = cancel(client, key)
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "cancelled"
    assert row(sm, key)["state"] == S.CANCELLED.value
    again = cancel(client, key)
    assert again.status_code == 200 and again.json()["state"] == "cancelled"


@pytest.mark.parametrize("key", ["a/b", "a/b/c", "x/release", "x/cancel"])
def test_the_unencoded_slash_form_reaches_the_same_row_as_the_encoded_form(sm, client, key):
    """Some HTTP clients send the raw slash instead of %2F; the decoded path is identical either way."""
    hold(sm, key)
    assert client.get(f"{DRAFTS}/{key}").json() == client.get(f"{DRAFTS}/{enc(key)}").json()
    digest = digest_of(client, key)
    assert release(client, key, digest, quoted=False).status_code == 200
    assert row(sm, key)["state"] == S.QUEUED.value


def test_the_handlers_own_404_is_what_an_unknown_slash_key_gets(client):
    """Before the change this was the ROUTER's generic ``{"detail": "Not Found"}`` for any key containing "/"
    (no route matched). Now every shape reaches the handler and gets the handler's own, identical body."""
    for key in ("never/existed", "never/existed/release", "never/existed/cancel", "never-existed"):
        for r in (client.get(f"{DRAFTS}/{enc(key)}"), release(client, key, "a" * 64), cancel(client, key)):
            assert r.status_code == 404 and r.json() == NOT_FOUND, (key, r.status_code, r.text)


# ====================================== 2. greedy converter: the right handler, the right row, never the neighbour


def test_a_mutating_route_is_only_reached_through_its_own_suffix_and_only_hits_its_own_key(sm, client):
    """"k", "k/release" and "k/cancel" are three different rows. Because the converter is greedy, the LAST segment
    decides the operation and everything before it is the key. Distinct bodies => distinct digests, so a request
    that landed on the wrong row would not be able to succeed with the right digest."""
    K, KR, KC = "k", "k/release", "k/cancel"
    for k in (K, KR, KC):
        hold(sm, k)
    d = {k: digest_of(client, k) for k in (K, KR, KC)}
    assert len(set(d.values())) == 3

    # POST .../k/cancel is "cancel the draft k" -- not "something about k/cancel".
    assert client.post(f"{DRAFTS}/k/cancel").status_code == 200
    assert states(sm, K, KR, KC) == {K: "cancelled", KR: "awaiting_approval", KC: "awaiting_approval"}

    # POST .../k%2Fcancel/cancel is "cancel the draft k/cancel".
    assert cancel(client, KC).status_code == 200
    assert states(sm, K, KR, KC) == {K: "cancelled", KR: "awaiting_approval", KC: "cancelled"}

    # POST .../k/release is "release the draft k" (already cancelled): a 409 for k -- it must NOT release k/release,
    # even though k/release is held and the digest in the body is not k's.
    r = client.post(f"{DRAFTS}/k/release", json={"content_sha256": d[K]})
    assert r.status_code == 409 and r.json() == {"detail": "draft_cancelled"}
    r = client.post(f"{DRAFTS}/k/release", json={"content_sha256": d[KR]})
    assert r.status_code == 409 and r.json() == {"detail": "draft_cancelled"}
    assert states(sm, K, KR, KC) == {K: "cancelled", KR: "awaiting_approval", KC: "cancelled"}

    # POST .../k%2Frelease/release is "release the draft k/release", with ITS digest.
    assert release(client, KR, d[KR]).status_code == 200
    assert states(sm, K, KR, KC) == {K: "cancelled", KR: "queued", KC: "cancelled"}
    assert row(sm, KR)["approved"] == d[KR] and row(sm, K)["approved"] is None


def test_get_reads_the_key_that_is_everything_after_the_prefix(sm, client):
    """GET has no suffix, so GET .../k/release can only mean the key "k/release" (a wrong-method 405 for the
    POST release route is no longer possible -- see section 5)."""
    hold(sm, "k")
    hold(sm, "k/release")
    assert client.get(f"{DRAFTS}/k").json()["body"] == "Hello, this draft belongs to 'k'.\n"
    assert client.get(f"{DRAFTS}/k/release").json()["body"] == "Hello, this draft belongs to 'k/release'.\n"
    assert client.get(f"{DRAFTS}/k%2Frelease").json()["body"] == "Hello, this draft belongs to 'k/release'.\n"


# ====================================== 3. no other route can match these URLs; these routes shadow nothing


def _scope(method, path):
    return {"type": "http", "method": method, "path": path, "root_path": "", "headers": []}


def _matching(method, path, *, full_only=True):
    out = []
    for r in app.router.routes:
        if not hasattr(r, "matches"):
            continue
        m = r.matches(_scope(method, path))[0]
        if m == Match.FULL or (not full_only and m == Match.PARTIAL):
            out.append(getattr(r, "name", repr(r)))
    return out


def test_exactly_three_routes_live_under_reply_drafts_and_all_three_declare_the_path_converter():
    mine = [r for r in app.router.routes if getattr(r, "path", "").startswith(f"{DRAFTS}")]
    assert sorted((tuple(sorted(r.methods)), r.path) for r in mine) == sorted([
        (("GET",), f"{DRAFTS}/{{idempotency_key:path}}"),
        (("POST",), f"{DRAFTS}/{{idempotency_key:path}}/release"),
        (("POST",), f"{DRAFTS}/{{idempotency_key:path}}/cancel"),
    ])


@pytest.mark.parametrize(
    "method,path,expected",
    [
        ("GET", f"{DRAFTS}/reply-1", "read_reply_draft"),
        ("GET", f"{DRAFTS}/a/b", "read_reply_draft"),
        ("GET", f"{DRAFTS}/a/release", "read_reply_draft"),       # a key, not "the POST route with the wrong verb"
        ("GET", f"{DRAFTS}/a/cancel", "read_reply_draft"),
        ("POST", f"{DRAFTS}/reply-1/release", "release_reply_draft"),
        ("POST", f"{DRAFTS}/a/b/release", "release_reply_draft"),
        ("POST", f"{DRAFTS}/a/release/release", "release_reply_draft"),
        ("POST", f"{DRAFTS}/a/cancel/release", "release_reply_draft"),
        ("POST", f"{DRAFTS}/reply-1/cancel", "cancel_reply_draft"),
        ("POST", f"{DRAFTS}/a/b/cancel", "cancel_reply_draft"),
        ("POST", f"{DRAFTS}/a/release/cancel", "cancel_reply_draft"),
        ("POST", f"{DRAFTS}/a/cancel/cancel", "cancel_reply_draft"),
    ],
)
def test_every_url_has_exactly_one_matching_handler_and_it_is_the_intended_one(method, path, expected):
    assert _matching(method, path) == [expected]


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/integrations/leadboost/outreach-actions/reply-1"),                # C9.1 status
        ("GET", "/integrations/leadboost/outreach-actions/reply-1/conversation"),   # C9.3 conversation
        ("GET", "/integrations/leadboost/outreach-actions/a/b/conversation"),
        ("POST", "/integrations/leadboost/outreach-actions"),
        ("POST", "/integrations/leadboost/outreach-requests"),
    ],
)
def test_the_other_leadboost_integration_routes_are_not_claimed_by_a_reply_draft_route(method, path):
    claimed = [n for n in _matching(method, path, full_only=False)
               if n in {"read_reply_draft", "release_reply_draft", "cancel_reply_draft"}]
    assert claimed == []
    assert _matching(method, path)            # and each still resolves to its own handler


def test_conversation_and_c91_routes_still_answer_as_before(sm, client):
    """A neighbouring ``:path`` route (the conversation read) keeps working for a slash key, and the C9.1 status
    route keeps its single-segment shape (it is deliberately untouched here)."""
    hold(sm, "a/b")
    base = "/integrations/leadboost/outreach-actions"
    assert client.get(f"{base}/{enc('a/b')}/conversation").status_code == 200
    assert client.get(f"{base}/{enc('a/b')}").status_code == 404                      # C9.1: unchanged limit at base
    assert client.get(f"{base}/{enc('a/b')}").json() == {"detail": "Not Found"}


# ====================================== 4. tenant isolation, size limit, legacy classification: unchanged


@pytest.mark.parametrize("key", ["reply-1", "a/b", "x/release"])
def test_another_tenants_key_is_the_same_404_as_a_key_that_does_not_exist(sm, client, who, key):
    hold(sm, key, org=ORG_B)
    before = snapshot_all(sm)
    who.org = ORG_A
    for r in (client.get(f"{DRAFTS}/{enc(key)}"), release(client, key, "a" * 64), cancel(client, key)):
        assert r.status_code == 404 and r.json() == NOT_FOUND
    assert snapshot_all(sm) == before, "a foreign key must change nothing"
    who.org = ORG_B                                           # and the owner still reaches it
    assert client.get(f"{DRAFTS}/{enc(key)}").status_code == 200


def test_the_same_slash_key_in_two_tenants_resolves_to_each_tenants_own_row(sm, client, who):
    hold(sm, "shared/key", org=ORG_A)
    hold(sm, "shared/key", org=ORG_B)
    who.org = ORG_B
    assert cancel(client, "shared/key").status_code == 200
    assert row(sm, "shared/key", org=ORG_B)["state"] == "cancelled"
    assert row(sm, "shared/key", org=ORG_A)["state"] == "awaiting_approval"


def test_key_length_limit_is_unchanged_255_ok_256_rejected(sm, client):
    ok, too_long = "a/" + "b" * 253, "a/" + "b" * 254
    assert len(ok) == 255 and len(too_long) == 256
    hold(sm, ok)
    assert client.get(f"{DRAFTS}/{enc(ok)}").status_code == 200
    for r in (client.get(f"{DRAFTS}/{enc(too_long)}"), release(client, too_long, "a" * 64), cancel(client, too_long)):
        assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["path", "idempotency_key"]


LEGACY_CONTEXT = {"reply_parent": {"message_id": 41, "subject": "Re: Hello"}}


def test_a_legacy_c94_reply_under_a_slash_key_is_still_never_a_404_and_never_written(sm, client):
    """Legacy classification must survive the routing change: a pre-013 reply (flag false, reply_parent present)
    is a 409 reconciliation error, and an ordinary non-reply dispatch is the plain 404."""
    with sm() as s:
        seed_dispatch(s, org=ORG_A, idem="legacy/a/b", email="legacy@example.com", state=S.SENT.value,
                      sender_email=SENDER, grounding_context=LEGACY_CONTEXT)
        seed_dispatch(s, org=ORG_A, idem="ordinary/a/b", email="ordinary@example.com", state=S.QUEUED.value,
                      sender_email=SENDER)
    before = snapshot_all(sm)
    for r in (client.get(f"{DRAFTS}/{enc('legacy/a/b')}"), release(client, "legacy/a/b", "a" * 64),
              cancel(client, "legacy/a/b")):
        assert r.status_code == 409 and r.json() == {"detail": LEGACY_CODE}
    for r in (client.get(f"{DRAFTS}/{enc('ordinary/a/b')}"), release(client, "ordinary/a/b", "a" * 64),
              cancel(client, "ordinary/a/b")):
        assert r.status_code == 404 and r.json() == NOT_FOUND
    assert snapshot_all(sm) == before


def test_a_digest_mismatch_under_a_slash_key_still_changes_nothing(sm, client):
    hold(sm, "a/b")
    before = snapshot_all(sm)
    r = release(client, "a/b", "0" * 64)
    assert r.status_code == 409 and r.json() == {"detail": "draft_changed"}
    assert snapshot_all(sm) == before


# ====================================== 5. wrong methods and edge shapes (asserted exactly; see CHANGES_V5.md)


@pytest.mark.parametrize(
    "method,suffix",
    [("POST", ""), ("DELETE", ""), ("PUT", ""), ("PATCH", ""), ("PUT", "/release"), ("PATCH", "/cancel"),
     ("DELETE", "/release"), ("DELETE", "/cancel")],
)
@pytest.mark.parametrize("key", ["reply-1", "a/b"])
def test_wrong_method_is_405_for_plain_and_slash_keys(sm, client, key, method, suffix):
    hold(sm, key)
    before = snapshot_all(sm)
    r = client.request(method, f"{DRAFTS}/{enc(key)}{suffix}")
    assert r.status_code == 405 and r.json() == {"detail": "Method Not Allowed"}
    assert snapshot_all(sm) == before


def test_deliberate_change_get_on_key_slash_release_or_cancel_is_a_read_of_that_longer_key(sm, client):
    """BEFORE: GET .../k/release was a 405 (the POST release route matched the path, not the verb). With a greedy
    converter GET has to mean "read the key whose text is 'k/release'" or a draft with such a key could never
    be read. The mutating routes are unaffected: POST still needs its own suffix (section 2)."""
    hold(sm, "k")
    before = snapshot_all(sm)
    for suffix in ("/release", "/cancel"):
        r = client.get(f"{DRAFTS}/k{suffix}")
        assert r.status_code == 404 and r.json() == NOT_FOUND
    assert snapshot_all(sm) == before, "a GET must never mutate"


def test_deliberate_change_an_empty_key_is_a_validation_error_not_a_router_404(sm, client):
    """BEFORE: an empty key matched no route (404). A greedy converter matches the empty string, so FastAPI's own
    ``min_length=1`` on the key now rejects it with a 422 -- after authentication, and before any lookup."""
    hold(sm, "reply-1")
    before = snapshot_all(sm)
    for r in (
        client.get(f"{DRAFTS}/"),
        client.post(f"{DRAFTS}//release", json={"content_sha256": "a" * 64}),
        client.post(f"{DRAFTS}//cancel"),
    ):
        assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["path", "idempotency_key"]
    assert snapshot_all(sm) == before


def test_deliberate_change_the_bare_collection_url_redirects_to_the_empty_key_instead_of_404(raw_client):
    """BEFORE: GET .../reply-drafts (no key, no trailing slash) was a 404. Starlette's redirect_slashes now finds
    the empty-key form and answers 307 to it (which is then the 422 above). Nothing is read, written or leaked."""
    r = raw_client.get(DRAFTS)
    assert r.status_code in (307, 404)
    if r.status_code == 307:
        assert r.headers["location"].endswith(f"{DRAFTS}/")
        assert raw_client.get(r.headers["location"]).status_code == 422


@pytest.mark.parametrize("key", ["reply-1", "a/b"])
def test_real_integration_auth_still_runs_first_for_plain_and_slash_keys(sm, monkeypatch, key):
    """Only get_db is overridden; the REAL integration dependency runs (as in the conversation auth test). The
    organization still comes from the API key alone, a foreign key is the plain 404, a missing/unknown key is
    refused before any lookup, and with no keys configured it fails closed (503) -- for every key shape alike."""
    from mailer_agent.api import deps

    hold(sm, key)
    monkeypatch.setattr(deps, "_KEY_MAP", {"key-a": ORG_A, "key-b": ORG_B})

    def _get_db():
        s = sm()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    try:
        c = TestClient(app)
        url = f"{DRAFTS}/{enc(key)}"
        ok = c.get(url, headers={"X-API-Key": "key-a"})
        assert ok.status_code == 200 and ok.json()["state"] == "awaiting_approval"
        cross = c.get(url, headers={"X-API-Key": "key-b"})
        assert cross.status_code == 404 and cross.json() == NOT_FOUND
        for r in (c.get(url), c.get(url, headers={"X-API-Key": "nope"}),
                  c.post(f"{url}/cancel"), c.post(f"{url}/cancel", headers={"X-API-Key": "nope"})):
            assert r.status_code in (401, 403), r.text
        assert row(sm, key)["state"] == S.AWAITING_APPROVAL.value
        monkeypatch.setattr(deps, "_KEY_MAP", {})
        assert c.get(url).status_code == 503
    finally:
        app.dependency_overrides.clear()
