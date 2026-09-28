"""
Pre-C6 design-correction tests.

See mailer_agent/models.py::resolve_expired_sending_lease for the full
reasoning. These tests do NOT exercise any worker/claim/lease-recovery
implementation -- that code does not exist yet (C6-C8 are not
implemented). They pin down the corrected DESIGN RULE itself as an
executable, version-controlled contract, ahead of that implementation:

An earlier design draft (the Phase B.2 reconciliation report's
crash/failure matrix) treated "worker crashed after Transaction B
committed but before send_email() was ever called" as safe to requeue
and retry, distinct from "worker crashed during/after the SMTP call"
(UNKNOWN, never auto-resent). That distinction is not observable from
anything this schema persists -- ExternalDispatch.claimed_at records
only when a row was claimed, not whether send_email() had been reached
-- so a lease-expiry recovery sweep cannot tell the two apart and must
not act as if it could. The corrected, single rule: an expired SENDING
lease always resolves to UNKNOWN, never automatically back to QUEUED.

Whichever worker implementation C6 adds must produce this same
resolution for an expired SENDING lease. These tests are the
enforcement point for that requirement.
"""

from mailer_agent.models import ExternalDispatchState, resolve_expired_sending_lease


def test_expired_sending_lease_resolves_to_unknown():
    assert resolve_expired_sending_lease() == ExternalDispatchState.UNKNOWN


def test_expired_sending_lease_never_resolves_to_queued():
    """
    The specific regression this corrects: an earlier design draft
    treated "crash before SMTP started" as safe-to-requeue. That
    distinction is not observable from persisted state (see
    resolve_expired_sending_lease's docstring), so it must never be
    implemented. This is the literal negative-space check for that.
    """
    assert resolve_expired_sending_lease() != ExternalDispatchState.QUEUED


def test_expired_sending_lease_never_resolves_to_sent_or_failed():
    """
    Equally important and easy to get wrong in a different direction:
    an expired lease must not be optimistically marked SENT (we don't
    know it was) or pessimistically marked FAILED (we don't know it
    wasn't) -- only UNKNOWN honestly represents "we cannot tell".
    """
    assert resolve_expired_sending_lease() not in (
        ExternalDispatchState.SENT,
        ExternalDispatchState.FAILED,
    )


def test_resolve_expired_sending_lease_is_unconditional():
    """
    No positively-persisted "SMTP call started" signal exists in this
    schema today (see docstring) -- so the function must take no
    arguments and have no conditional branch that could resolve
    differently under any input. If a later phase adds such a signal,
    this test (and the function's signature) is the place to change,
    deliberately, not a side effect of adding a parameter elsewhere.
    """
    import inspect

    sig = inspect.signature(resolve_expired_sending_lease)
    assert len(sig.parameters) == 0


def test_no_automatic_requeue_helper_exists_on_the_state_enum():
    """
    Guards against a future implementation quietly reintroducing an
    auto-requeue path -- e.g. a QUEUED-returning helper named something
    like `recover`/`retry`/`requeue` added directly onto
    ExternalDispatchState -- instead of going through the single,
    documented resolve_expired_sending_lease() contract point. Not
    exhaustive (a determined future change can still violate the rule
    elsewhere), but it pins the enum itself to having no such surface
    today.
    """
    disallowed_names = {"recover", "retry", "requeue", "auto_requeue"}
    members_and_methods = set(dir(ExternalDispatchState))
    assert disallowed_names.isdisjoint(members_and_methods)
