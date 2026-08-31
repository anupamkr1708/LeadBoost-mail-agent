"""
API routes for system monitoring and integration validation.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from mailer_agent.api.deps import require_api_key
from mailer_agent.db import get_db
from mailer_agent.integration import mailer_agent

router = APIRouter(prefix="/system", tags=["system"], dependencies=[Depends(require_api_key)])


@router.get("/health")
def get_system_health(db: Session = Depends(get_db)):
    """
    Get comprehensive system health metrics.
    
    Returns:
    - Contact state distribution
    - Message statistics
    - Classification success rate
    - LLM metrics
    - Configuration status
    """
    return mailer_agent.get_system_health(db)


@router.get("/integration/validate")
def validate_integration():
    """
    Validate that all components are properly integrated.
    
    Checks:
    - LLM availability
    - Configuration
    - Module imports
    - Database connectivity
    """
    return mailer_agent.validate_integration()


@router.get("/metrics/llm")
def get_llm_metrics():
    """
    Get LLM provider metrics.
    
    Returns:
    - Total calls
    - Success rate
    - Rate limits hit
    - Timeouts
    - Malformed outputs
    """
    return mailer_agent.get_llm_metrics()


@router.get("/metrics/classification")
def get_classification_metrics(db: Session = Depends(get_db)):
    """
    Get semantic classification metrics.
    """
    from mailer_agent.models import Message, MessageDirection
    
    inbound_messages = db.query(Message).filter(
        Message.direction == MessageDirection.INBOUND.value
    ).all()
    
    if not inbound_messages:
        return {
            "total": 0,
            "success": 0,
            "success_rate": 0,
            "failure_reasons": {}
        }
    
    success_count = sum(1 for m in inbound_messages if m.classification_success)
    
    # Group by failure reason
    failure_reasons = {}
    for m in inbound_messages:
        if not m.classification_success and m.classification_failure_reason:
            reason = m.classification_failure_reason
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    return {
        "total": len(inbound_messages),
        "success": success_count,
        "success_rate": success_count / len(inbound_messages),
        "failed": len(inbound_messages) - success_count,
        "failure_reasons": failure_reasons
    }


@router.get("/metrics/states")
def get_state_distribution(db: Session = Depends(get_db)):
    """
    Get distribution of contact states.
    """
    from mailer_agent.models import Contact, ContactStatus
    
    distribution = {}
    for status in ContactStatus:
        count = db.query(Contact).filter(Contact.status == status.value).count()
        distribution[status.value] = count
    
    total = sum(distribution.values())
    
    return {
        "distribution": distribution,
        "total": total,
        "percentages": {
            state: (count / total * 100 if total > 0 else 0)
            for state, count in distribution.items()
        }
    }


@router.get("/config")
def get_system_config():
    """
    Get system configuration (safe subset).
    """
    from mailer_agent.config import get_settings
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
        "database_type": "postgresql" if "postgresql" in settings.database_url else "sqlite"
    }
