"""
M3 real-IMAP end-to-end: a disposable local Dovecot (implicit TLS, self-signed
cert trusted through SSL_CERT_FILE, throw-away accounts and maildirs). No real
mailbox is ever contacted. Everything on the Mailer side is the production code
path: mail/mailbox_inbound.poll_all_mailboxes -> imap_reader (UID FETCH
BODY.PEEK / UID STORE) -> reply_handler_v2.process_mailbox_inbound -> database.
Mail is injected over the IMAP protocol itself (APPEND), and Seen state is read
back from the server, so "marked Seen only after commit" is observed for real.

Skipped unless the `dovecot` binary exists and the process is root (Dovecot
needs it to drop to its service users). CI installs dovecot-imapd and runs this
file under sudo; see .github/workflows/ci.yml. Uses PostgreSQL when
POSTGRES_TEST_URL is reachable, otherwise a SQLite file.
"""

from __future__ import annotations

import imaplib
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.imap_e2e]

if not shutil.which("dovecot") or (hasattr(os, "geteuid") and os.geteuid() != 0):
    pytest.skip("needs the dovecot binary and root (see module docstring)", allow_module_level=True)

from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from mailer_agent.config import get_settings  # noqa: E402
from mailer_agent.mail import mailbox_inbound  # noqa: E402
from mailer_agent.mailbox_secrets import encrypt_secret  # noqa: E402
from mailer_agent.models import Base, Mailbox, MailboxStatus  # noqa: E402
from tests.m3_support import give_imap, inbound_rows, raw_email, seed_org_with_imap_mailbox  # noqa: E402

OUT_A, OUT_B = "<out-a@mailer.a>", "<out-b@mailer.b>"


class Dovecot:
    def __init__(self, root: Path, port: int):
        self.root, self.port = root, port
        self.users = root / "users"
        self.accounts: dict[str, str] = {}
        self._mtime = time.time()

    # --- accounts --------------------------------------------------------------
    def _write_users(self):
        self.users.write_text("".join(f"{u}:{{PLAIN}}{p}\n" for u, p in self.accounts.items()))
        # Dovecot reloads passwd-file only when its mtime changes; rewrites inside the
        # same timestamp tick would otherwise be invisible, so force a new mtime.
        self._mtime += 5
        os.utime(self.users, (self._mtime, self._mtime))
        subprocess.run(["doveadm", "-c", str(self.root / "dovecot.conf"), "reload"],
                       check=False, capture_output=True, timeout=20)

    def _wait_until_accepted(self, user):
        """Block until the server really accepts this account's current password
        (auth-process reload is asynchronous)."""
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                self._conn(user).logout()
                return
            except imaplib.IMAP4.error:
                time.sleep(0.1)
        raise AssertionError(f"dovecot never accepted {user}")

    def new_account(self) -> tuple[str, str]:
        user, pw = f"u{uuid.uuid4().hex[:10]}@e2e.example", f"Pw-{uuid.uuid4().hex}"
        self.accounts[user] = pw
        self._write_users()
        self._wait_until_accepted(user)
        return user, pw

    def set_password(self, user, pw):
        self.accounts[user] = pw
        self._write_users()
        self._wait_until_accepted(user)

    # --- protocol-level helpers ------------------------------------------------
    def _conn(self, user):
        c = imaplib.IMAP4_SSL("localhost", self.port, timeout=15)
        c.login(user, self.accounts[user])
        return c

    def deliver(self, user, raw: bytes):
        c = self._conn(user)
        try:
            assert c.append("INBOX", None, None, raw)[0] == "OK"
        finally:
            c.logout()

    def unseen(self, user) -> list[bytes]:
        c = self._conn(user)
        try:
            c.select("INBOX")
            return c.uid("SEARCH", None, "UNSEEN")[1][0].split()
        finally:
            c.logout()

    def forget_seen(self, user):
        c = self._conn(user)
        try:
            c.select("INBOX")
            c.uid("STORE", "1:*", "-FLAGS", "\\Seen")
        finally:
            c.logout()


@pytest.fixture(scope="session")
def dovecot():
    root = Path(tempfile.mkdtemp(prefix="m3-dovecot-"))
    for d in ("run", "state", "mail"):
        (root / d).mkdir()
    subprocess.run(["id", "vmail"], capture_output=True).returncode == 0 or subprocess.run(
        ["useradd", "-r", "-M", "-s", "/usr/sbin/nologin", "vmail"], check=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    import datetime
    import ipaddress
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=2))
            .add_extension(x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    (root / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (root / "key.pem").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    (root / "dovecot.conf").write_text(f"""
base_dir = {root}/run
state_dir = {root}/state
log_path = {root}/dovecot.log
protocols = imap
listen = 127.0.0.1
default_internal_user = dovecot
default_login_user = dovenull
disable_plaintext_auth = no
auth_mechanisms = plain login
ssl = required
ssl_cert = <{root}/cert.pem
ssl_key = <{root}/key.pem
mail_location = maildir:{root}/mail/%u/Maildir
passdb {{
  driver = passwd-file
  args = scheme=PLAIN username_format=%u {root}/users
}}
userdb {{
  driver = static
  args = uid=vmail gid=vmail home={root}/mail/%u
}}
service imap-login {{
  inet_listener imap {{
    port = 0
  }}
  inet_listener imaps {{
    port = {port}
    ssl = yes
  }}
}}
""")
    (root / "users").write_text("")
    shutil.chown(root / "mail", "vmail", "vmail")
    subprocess.run(["chmod", "-R", "a+rX", str(root)], check=True)
    proc = subprocess.Popen(["dovecot", "-F", "-c", str(root / "dovecot.conf")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("dovecot did not start")
    old = os.environ.get("SSL_CERT_FILE")
    os.environ["SSL_CERT_FILE"] = str(root / "cert.pem")   # trust the throw-away cert; no code change
    try:
        yield Dovecot(root, port)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        if old is None:
            os.environ.pop("SSL_CERT_FILE", None)
        else:
            os.environ["SSL_CERT_FILE"] = old
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    url = os.environ.get("POSTGRES_TEST_URL")
    eng = None
    if url:
        try:
            eng = create_engine(url)
            with eng.connect() as c:
                c.execute(text("SELECT 1"))
        except Exception:
            eng = None
    if eng is None:
        eng = create_engine(f"sqlite:///{tmp_path / 'e2e.db'}", connect_args={"check_same_thread": False, "timeout": 15})
    Base.metadata.create_all(bind=eng)
    if eng.dialect.name == "postgresql":
        with eng.connect() as c:
            c.execute(text("TRUNCATE messages, external_dispatches, contacts, campaigns, mailboxes, suppression_list CASCADE"))
            c.commit()
    import mailer_agent.llm.provider_v2 as provider_module
    monkeypatch.setattr(provider_module, "is_llm_available", lambda: False)
    yield sessionmaker(bind=eng, autoflush=False)
    eng.dispose()


@pytest.fixture
def orgs(dovecot, db_env):
    ua, pa = dovecot.new_account()
    ub, pb = dovecot.new_account()
    with db_env() as db:
        mb_a, ct_a, _ = seed_org_with_imap_mailbox(
            db, org="org-a", host="localhost", imap_user=ua, imap_password=pa,
            contact_email="lead@example.com", outbound_message_id=OUT_A)
        mb_b, ct_b, _ = seed_org_with_imap_mailbox(
            db, org="org-b", host="localhost", imap_user=ub, imap_password=pb,
            contact_email="lead@example.com", outbound_message_id=OUT_B)
        for mb in (mb_a, mb_b):
            mb.imap_port = dovecot.port
        db.commit()
        return SimpleNamespace(sm=db_env, ua=ua, pa=pa, ub=ub, pb=pb, mb_a=mb_a.id, mb_b=mb_b.id,
                               ct_a=ct_a.id, ct_b=ct_b.id, mailbox_email_b=mb_b.email_address)


def cycle(o):
    return {x.mailbox_id: x for x in mailbox_inbound.poll_all_mailboxes(o.sm)}


def stored(o, contact_id=None):
    with o.sm() as db:
        rows = inbound_rows(db, contact_id)
        for r in rows:
            db.expunge(r)
        return rows


# ---------------------------------------------------------------------------------

def test_two_mailboxes_two_orgs_each_message_lands_in_its_own_org_thread(dovecot, orgs):
    dovecot.deliver(orgs.ua, raw_email(message_id="<a1@prospect>", in_reply_to=OUT_A, body="Reply for org A"))
    dovecot.deliver(orgs.ub, raw_email(message_id="<b1@prospect>", in_reply_to=OUT_B, body="Reply for org B"))
    out = cycle(orgs)
    assert out[orgs.mb_a].status == out[orgs.mb_b].status == mailbox_inbound.STATUS_OK
    a, b = stored(orgs, orgs.ct_a), stored(orgs, orgs.ct_b)
    assert [(r.message_id_header, r.mailbox_id, r.body) for r in a] == [("<a1@prospect>", orgs.mb_a, "Reply for org A")]
    assert [(r.message_id_header, r.mailbox_id, r.body) for r in b] == [("<b1@prospect>", orgs.mb_b, "Reply for org B")]
    assert a[0].in_reply_to_header == OUT_A and b[0].in_reply_to_header == OUT_B
    assert dovecot.unseen(orgs.ua) == [] and dovecot.unseen(orgs.ub) == []     # real server: now Seen


def test_cross_tenant_injection_is_never_attached(dovecot, orgs):
    """Org B's contact, Org B's thread id and Org B's mailbox address, delivered
    into mailbox A: must not touch org B, and must not be attached to org A either
    (org A has a contact with that address, but the thread id is B's)."""
    with orgs.sm() as db:
        from mailer_agent.models import Contact
        db.get(Contact, orgs.ct_a).email = "someone-else@a.example"        # A no longer shares the address
        db.commit()
    dovecot.deliver(orgs.ua, raw_email(
        from_addr="lead@example.com", to_addr=orgs.mailbox_email_b, message_id="<x@p>",
        in_reply_to=OUT_B, references=[OUT_B]))
    cycle(orgs)
    assert stored(orgs) == []
    assert dovecot.unseen(orgs.ua) == []          # unmatched -> marked Seen (approved behaviour)


def test_duplicate_ingestion_converges_on_one_message(dovecot, orgs):
    dovecot.deliver(orgs.ua, raw_email(message_id="<dup@p>", in_reply_to=OUT_A))
    cycle(orgs)
    dovecot.forget_seen(orgs.ua)                  # the server offers the same message again
    assert len(dovecot.unseen(orgs.ua)) == 1
    cycle(orgs)
    assert len(stored(orgs, orgs.ct_a)) == 1 and dovecot.unseen(orgs.ua) == []


def test_message_without_message_id_is_deduplicated_through_a_real_server(dovecot, orgs):
    dovecot.deliver(orgs.ua, raw_email(message_id=None, in_reply_to=OUT_A))
    cycle(orgs)
    dovecot.forget_seen(orgs.ua)
    cycle(orgs)
    rows = stored(orgs, orgs.ct_a)
    assert len(rows) == 1 and rows[0].message_id_header.startswith("synthetic:")


def test_disabled_mailbox_is_not_polled_and_its_mail_stays_unseen(dovecot, orgs):
    with orgs.sm() as db:
        db.get(Mailbox, orgs.mb_a).status = MailboxStatus.DISABLED.value
        db.commit()
    dovecot.deliver(orgs.ua, raw_email(message_id="<d@p>", in_reply_to=OUT_A))
    dovecot.deliver(orgs.ub, raw_email(message_id="<e@p>", in_reply_to=OUT_B))
    cycle(orgs)
    assert len(dovecot.unseen(orgs.ua)) == 1 and stored(orgs, orgs.ct_a) == []      # untouched
    assert len(stored(orgs, orgs.ct_b)) == 1
    with orgs.sm() as db:
        db.get(Mailbox, orgs.mb_a).status = MailboxStatus.ACTIVE.value
        db.commit()
    cycle(orgs)
    assert len(stored(orgs, orgs.ct_a)) == 1 and dovecot.unseen(orgs.ua) == []


def test_one_mailbox_with_a_bad_credential_does_not_stop_the_other(dovecot, orgs):
    with orgs.sm() as db:
        db.get(Mailbox, orgs.mb_a).imap_password_enc = encrypt_secret("not-the-real-password")
        db.commit()
    dovecot.deliver(orgs.ua, raw_email(message_id="<f@p>", in_reply_to=OUT_A))
    dovecot.deliver(orgs.ub, raw_email(message_id="<g@p>", in_reply_to=OUT_B))
    out = cycle(orgs)
    assert out[orgs.mb_a].status == mailbox_inbound.STATUS_IMAP_ERROR
    assert out[orgs.mb_b].status == mailbox_inbound.STATUS_OK
    assert stored(orgs, orgs.ct_a) == [] and len(stored(orgs, orgs.ct_b)) == 1
    assert len(dovecot.unseen(orgs.ua)) == 1


def test_commit_failure_leaves_the_message_unseen_on_the_real_server_then_retry_succeeds(dovecot, orgs):
    dovecot.deliver(orgs.ua, raw_email(message_id="<c@p>", in_reply_to=OUT_A))
    state = {"fail": True}

    def factory():
        s = orgs.sm()
        real = s.commit

        def commit():
            if state["fail"]:
                raise OperationalError("COMMIT", {}, Exception("db down"))
            return real()

        s.commit = commit
        return s

    mailbox_inbound.poll_all_mailboxes(factory)
    assert stored(orgs, orgs.ct_a) == []
    assert len(dovecot.unseen(orgs.ua)) == 1       # FETCH used BODY.PEEK and nothing marked it Seen
    state["fail"] = False
    mailbox_inbound.poll_all_mailboxes(factory)
    assert len(stored(orgs, orgs.ct_a)) == 1 and dovecot.unseen(orgs.ua) == []


def test_credential_rotation_uses_the_newly_stored_credential(dovecot, orgs):
    new_pw = "Pw-ROTATED-" + uuid.uuid4().hex
    dovecot.set_password(orgs.ua, new_pw)                        # upstream rotated first
    dovecot.deliver(orgs.ua, raw_email(message_id="<r@p>", in_reply_to=OUT_A))
    assert cycle(orgs)[orgs.mb_a].status == mailbox_inbound.STATUS_IMAP_ERROR   # stale stored credential
    with orgs.sm() as db:
        db.get(Mailbox, orgs.mb_a).imap_password_enc = encrypt_secret(new_pw)
        db.commit()
    assert cycle(orgs)[orgs.mb_a].status == mailbox_inbound.STATUS_OK
    assert len(stored(orgs, orgs.ct_a)) == 1


def test_threading_in_reply_to_and_references_over_real_imap(dovecot, orgs):
    dovecot.deliver(orgs.ua, raw_email(
        from_addr="assistant@prospect.example", message_id="<t1@p>", in_reply_to=OUT_A))
    dovecot.deliver(orgs.ua, raw_email(
        from_addr="assistant@prospect.example", message_id="<t2@p>",
        in_reply_to="<nope@x>", references=["<root@x>", OUT_A]))
    cycle(orgs)
    assert sorted(r.message_id_header for r in stored(orgs, orgs.ct_a)) == ["<t1@p>", "<t2@p>"]
    assert stored(orgs, orgs.ct_b) == []


def test_no_secret_appears_in_logs_responses_or_plaintext_columns(dovecot, orgs, caplog):
    with orgs.sm() as db:
        db.get(Mailbox, orgs.mb_b).imap_password_enc = encrypt_secret("wrong-for-b")
        db.commit()
    dovecot.deliver(orgs.ua, raw_email(message_id="<l@p>", in_reply_to=OUT_A))
    with caplog.at_level(logging.DEBUG):
        cycle(orgs)
    key = get_settings().mailbox_encryption_key.get_secret_value()
    for secret in (orgs.pa, orgs.pb, "wrong-for-b", key):
        assert secret not in caplog.text
    with orgs.sm() as db:
        dump = str(db.execute(text("SELECT imap_username, imap_password_enc, smtp_password_enc FROM mailboxes")).all())
        assert orgs.pa not in dump and orgs.pb not in dump
