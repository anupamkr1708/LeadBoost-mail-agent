"""
Shared builders for the M3 (per-Mailbox IMAP) tests. Deterministic, offline.

FakeImapWorld is an in-memory stand-in for imaplib.IMAP4_SSL that implements only
what mail/imap_reader.drain_unseen and mail/mailbox_inbound.poll_mailbox use
(LOGIN, SELECT, UID SEARCH/FETCH/STORE, LOGOUT) and records everything a real
server could observe: which (host, username, password) logged in, and the order
of FETCH/STORE events. The real-protocol check is tests/test_m3_imap_e2e.py.
"""

from __future__ import annotations

import imaplib
from dataclasses import dataclass, field
from email.message import EmailMessage

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mailer_agent.mailbox_secrets import encrypt_secret
from mailer_agent.models import Base, Contact, Message, Mailbox
from tests.dispatch_support import seed_dispatch


def make_session_factory(tmp_path, name="m3.db"):
    eng = create_engine(
        f"sqlite:///{tmp_path / name}", connect_args={"check_same_thread": False, "timeout": 15}
    )
    Base.metadata.create_all(bind=eng)
    return sessionmaker(bind=eng, autoflush=False), eng


def raw_email(
    *, from_addr="lead@example.com", to_addr="outreach@sender.example.org",
    subject="Re: Quick question", body="Sounds interesting, tell me more.",
    message_id="<reply-1@prospect.example>", in_reply_to=None, references=None,
) -> bytes:
    m = EmailMessage()
    m["From"] = from_addr
    m["To"] = to_addr
    m["Subject"] = subject
    if message_id:
        m["Message-ID"] = message_id
    if in_reply_to:
        m["In-Reply-To"] = in_reply_to
    if references:
        m["References"] = " ".join(references)
    m.set_content(body)
    return m.as_bytes()


@dataclass
class FakeAccount:
    password: str
    messages: list[dict] = field(default_factory=list)  # {uid, raw, seen}
    next_uid: int = 1


class FakeImapWorld:
    def __init__(self):
        self.accounts: dict[tuple[str, str], FakeAccount] = {}
        self.logins: list[tuple[str, str, str]] = []      # (host, username, password) attempted
        self.connects: list[str] = []                     # hosts a connection was opened to
        self.events: list[tuple] = []                     # ("fetch"|"store", host, user, uid)
        self.unreachable_hosts: set[str] = set()
        self.fail_store = False
        self.corrupt_uids: set[tuple[str, int]] = set()            # FETCH returns an unusable body
        self.on_login = None                               # optional hook(host, username)

    def add_account(self, host, username, password) -> FakeAccount:
        acct = FakeAccount(password=password)
        self.accounts[(host, username)] = acct
        return acct

    def inject(self, host, username, raw: bytes) -> int:
        acct = self.accounts[(host, username)]
        uid = acct.next_uid
        acct.next_uid += 1
        acct.messages.append({"uid": uid, "raw": raw, "seen": False})
        return uid

    def is_seen(self, host, username, uid) -> bool:
        return next(m for m in self.accounts[(host, username)].messages if m["uid"] == uid)["seen"]

    def unseen_uids(self, host, username) -> list[int]:
        return [m["uid"] for m in self.accounts[(host, username)].messages if not m["seen"]]

    # patch target: imaplib.IMAP4_SSL
    def connection_factory(self):
        world = self

        class _Conn:
            def __init__(self, host, port=993, timeout=None, **_kw):
                if host in world.unreachable_hosts:
                    raise TimeoutError("simulated connect timeout")
                world.connects.append(host)
                self.host, self.user, self.acct = host, None, None

            def login(self, user, password):
                world.logins.append((self.host, user, password))
                if world.on_login:
                    world.on_login(self.host, user)
                acct = world.accounts.get((self.host, user))
                if acct is None or acct.password != password:
                    raise imaplib.IMAP4.error("[AUTHENTICATIONFAILED] Invalid credentials")
                self.user, self.acct = user, acct
                return "OK", [b"logged in"]

            def select(self, _box):
                assert self.acct is not None, "SELECT before LOGIN"
                return "OK", [str(len(self.acct.messages)).encode()]

            def uid(self, cmd, *args):
                assert self.acct is not None, "UID before LOGIN"
                cmd = cmd.upper()
                if cmd == "SEARCH":
                    unseen = [str(m["uid"]).encode() for m in self.acct.messages if not m["seen"]]
                    return "OK", [b" ".join(unseen)]
                uid = int(args[0])
                msg = next(m for m in self.acct.messages if m["uid"] == uid)
                if cmd == "FETCH":
                    assert "PEEK" in args[1], "must fetch with BODY.PEEK so FETCH never sets \\Seen"
                    world.events.append(("fetch", self.host, self.user, uid))
                    if (self.host, uid) in world.corrupt_uids:
                        return "OK", [(b"x (BODY[])", None)]
                    return "OK", [(b"x (BODY[] {n})", msg["raw"]), b")"]
                if cmd == "STORE":
                    world.events.append(("store", self.host, self.user, uid))
                    if world.fail_store:
                        return "NO", [b"store failed"]
                    msg["seen"] = True
                    return "OK", [b"stored"]
                raise AssertionError(f"unexpected UID command {cmd}")

            def logout(self):
                return "BYE", [b"bye"]

        return _Conn


def give_imap(db, mailbox: Mailbox, *, host, username, password, port=993) -> Mailbox:
    mailbox.imap_host, mailbox.imap_port = host, port
    mailbox.imap_username = username
    mailbox.imap_password_enc = encrypt_secret(password)
    db.commit()
    return mailbox


def seed_org_with_imap_mailbox(
    db, *, org, host, imap_user, imap_password, contact_email="lead@example.com",
    mailbox_email=None, outbound_message_id=None,
):
    """One integration campaign/contact/outbound-message/dispatch for `org`, its
    sole mailbox given an IMAP identity. Returns (mailbox, contact, outbound_message)."""
    mailbox_email = mailbox_email or f"outreach@{org}.example.org"
    dispatch = seed_dispatch(db, org=org, email=contact_email, sender_email=mailbox_email)
    mailbox = db.get(Mailbox, dispatch.mailbox_id)
    give_imap(db, mailbox, host=host, username=imap_user, password=imap_password)
    msg = db.get(Message, dispatch.message_id)
    if outbound_message_id:
        msg.message_id_header = outbound_message_id
        db.commit()
    return mailbox, db.get(Contact, dispatch.contact_id), msg


def inbound_rows(db, contact_id=None):
    q = db.query(Message).filter(Message.direction == "inbound")
    if contact_id is not None:
        q = q.filter(Message.contact_id == contact_id)
    return q.order_by(Message.id).all()
