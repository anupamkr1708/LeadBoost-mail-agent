"""
API routes for system monitoring and integration validation.

Multi-tenancy: every route here that returns tenant data is scoped to the
calling organization via get_current_org_id(), exactly like
campaigns/contacts/messages. Aggregate counts (contact-state distribution,
message counts, classification rates) are computed with SQL-side
GROUP BY / COUNT so they scale with an index scan instead of loading every
row into Python -- and, critically, so one organization's dashboard can
never reveal another organization's volume.

/system/integration/validate, /system/metrics/llm, and /system/config are
the exceptions: they report process-wide booleans/counters (is this
deployment healthy, provider retry stats, safety-toggle configuration),
not any tenant's data, so they are intentionally left unscoped.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from mailer_agent.api.deps import get_current_org_id, require_api_key
from mailer_agent.config import get_settings
from mailer_agent.db import get_db
from mailer_agent.integration import mailer_agent
from mailer_agent.models import Campaign, Contact, ContactStatus, Message, MessageDirection, MessageStatus

router = APIRouter(prefix="/system", tags=["system"], dependencies=[Depends(require_api_key)])


@router.get("/health")
def get_system_health(
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    """
    Comprehensive health metrics for the CALLING ORGANIZATION ONLY.

    Returns contact-state distribution, message send stats, classification
    success rate, LLM provider metrics (process-wide -- see note below),
    and the safety-toggle configuration. All per-tenant counts are
    computed with a single grouped SQL query each, scoped to org_id.
    """
    # --- Contact state distribution: one GROUP BY, not one query per enum ---
    state_rows = (
        db.query(Contact.status, func.count(Contact.id))
        .join(Campaign, Contact.campaign_id == Campaign.id)
        .filter(Campaign.organization_id == org_id)
        .group_by(Contact.status)
        .all()
    )
    contact_states = {status.value: 0 for status in ContactStatus}
    for status_value, count in state_rows:
        contact_states[status_value] = count

    # --- Message stats: single aggregate query, not .all() + Python loop ---
    msg_row = (
        db.query(
            func.count(Message.id),
            func.sum(case((Message.status == MessageStatus.SENT.value, 1), else_=0)),
            func.sum(case((Message.status == MessageStatus.FAILED.value, 1), else_=0)),
            func.sum(case((Message.status == MessageStatus.UNKNOWN.value, 1), else_=0)),
        )
        .join(Contact, Message.contact_id == Contact.id)
        .join(Campaign, Contact.campaign_id == Campaign.id)
        .filter(Campaign.organization_id == org_id)
        .one()
    )
    total_messages, sent_messages, failed_messages, unknown_messages = (
        (v or 0) for v in msg_row
    )

    # --- Classification success rate: aggregate over inbound only ---
    class_row = (
        db.query(
            func.count(Message.id),
            func.sum(case((Message.classification_success.is_(True), 1), else_=0)),
        )
        .join(Contact, Message.contact_id == Contact.id)
        .join(Campaign, Contact.campaign_id == Campaign.id)
        .filter(
            Campaign.organization_id == org_id,
            Message.direction == MessageDirection.INBOUND.value,
        )
        .one()
    )
    classification_total, classification_success = ((v or 0) for v in class_row)

    settings = get_settings()

    return {
        "organization_id": org_id,
        "contact_states": contact_states,
        "messages": {
            "total": total_messages,
            "sent": sent_messages,
            "failed": failed_messages,
            "unknown": unknown_messages,
            "success_rate": (sent_messages / total_messages) if total_messages else 0,
        },
        "classification": {
            "success": classification_success,
            "total": classification_total,
            "success_rate": (
                classification_success / classification_total if classification_total else 0
            ),
        },
        # LLM metrics are process-wide (the provider client has no concept
        # of organization), not per-tenant -- labeled explicitly so this
        # isn't mistaken for org-scoped data.
        "llm_metrics_process_wide": mailer_agent.get_llm_metrics(),
        "settings": {
            "live_sending_enabled": settings.live_sending_enabled,
            "auto_reply_enabled": settings.auto_reply_enabled,
        },
    }


@router.get("/integration/validate")
def validate_integration():
    """
    Validate that all components are properly integrated.

    Checks: LLM availability, configuration, module imports, database
    connectivity. Deployment-wide booleans only -- no tenant data -- so
    this intentionally is not org-scoped.
    """
    return mailer_agent.validate_integration()


@router.get("/metrics/llm")
def get_llm_metrics():
    """
    Process-wide LLM provider metrics (total calls, success rate, rate
    limits, timeouts, malformed outputs). Not org-scoped: the underlying
    provider client and its retry/metrics counters are shared by the
    whole process, not partitioned per tenant.
    """
    return mailer_agent.get_llm_metrics()


@router.get("/metrics/classification")
def get_classification_metrics(
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    """Semantic classification metrics for the calling organization."""
    total_row = (
        db.query(
            func.count(Message.id),
            func.sum(case((Message.classification_success.is_(True), 1), else_=0)),
        )
        .join(Contact, Message.contact_id == Contact.id)
        .join(Campaign, Contact.campaign_id == Campaign.id)
        .filter(
            Campaign.organization_id == org_id,
            Message.direction == MessageDirection.INBOUND.value,
        )
        .one()
    )
    total, success = ((v or 0) for v in total_row)

    if total == 0:
        return {"total": 0, "success": 0, "success_rate": 0, "failure_reasons": {}}

    failure_rows = (
        db.query(Message.classification_failure_reason, func.count(Message.id))
        .join(Contact, Message.contact_id == Contact.id)
        .join(Campaign, Contact.campaign_id == Campaign.id)
        .filter(
            Campaign.organization_id == org_id,
            Message.direction == MessageDirection.INBOUND.value,
            Message.classification_success.is_(False),
            Message.classification_failure_reason.isnot(None),
        )
        .group_by(Message.classification_failure_reason)
        .all()
    )
    failure_reasons = {reason: count for reason, count in failure_rows}

    return {
        "total": total,
        "success": success,
        "success_rate": success / total,
        "failed": total - success,
        "failure_reasons": failure_reasons,
    }


@router.get("/metrics/states")
def get_state_distribution(
    org_id: str = Depends(get_current_org_id),
    db: Session = Depends(get_db),
):
    """Contact-state distribution for the calling organization (single GROUP BY query)."""
    rows = (
        db.query(Contact.status, func.count(Contact.id))
        .join(Campaign, Contact.campaign_id == Campaign.id)
        .filter(Campaign.organization_id == org_id)
        .group_by(Contact.status)
        .all()
    )
    distribution = {status.value: 0 for status in ContactStatus}
    for status_value, count in rows:
        distribution[status_value] = count

    total = sum(distribution.values())

    return {
        "distribution": distribution,
        "total": total,
        "percentages": {
            state: (count / total * 100 if total > 0 else 0)
            for state, count in distribution.items()
        },
    }


@router.get("/config")
def get_system_config():
    """
    Safe (non-secret) subset of process configuration. Not org-scoped --
    these are deployment-wide settings, not per-tenant data.
    """
    settings = get_settings()

    return {
        "live_sending_enabled": settings.live_sending_enabled,
        "auto_reply_enabled": settings.auto_reply_enabled,
        "llm_model": settings.llm_model,
        "llm_temperature": settings.llm_temperature,
        "llm_max_tokens": settings.llm_max_tokens,
        "max_sends_per_cycle": settings.max_sends_per_cycle,
        "send_delay_seconds": settings.send_delay_seconds,
        "imap_poll_seconds": settings.imap_poll_seconds,
        "followup_poll_seconds": settings.followup_poll_seconds,
        "run_scheduler_in_process": settings.run_scheduler_in_process,
        "database_type": "postgresql" if "postgresql" in settings.database_url else "sqlite",
    }
