"""
Main integration module for production-grade Mailer Agent.

Wires together all upgraded components:
- Semantic intelligence (semantic/classifier.py)
- State machine (state_machine.py)
- Policy engine (policy/next_action.py)
- Conversation-aware scheduling (followup/conversation_aware.py)
- Enhanced reply handling (mail/reply_handler_v2.py)
- Improved LLM provider (llm/provider_v2.py)
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from mailer_agent.config import get_settings
from mailer_agent.followup.conversation_aware import FollowUpScheduler
from mailer_agent.followup.engine_v2 import IntegratedFollowUpEngine
from mailer_agent.llm import provider_v2 as llm_provider
from mailer_agent.mail.imap_reader import InboundEmail
from mailer_agent.mail.reply_handler_v2 import process_inbound_email_v2
from mailer_agent.models import Campaign, Contact
from mailer_agent.policy.next_action import NextBestActionPolicy
from mailer_agent.semantic.classifier import classify_prospect_reply
from mailer_agent.semantic_models import ClassificationResult, IntentType

logger = logging.getLogger("mailer_agent.integration")
settings = get_settings()


class MailerAgentCore:
    """
    Core integration class for the production-grade Mailer Agent.
    
    Provides a unified interface to all upgraded functionality.
    """
    
    def __init__(self):
        self.followup_engine = IntegratedFollowUpEngine()
        self.followup_scheduler = FollowUpScheduler()
        self.policy_engine = NextBestActionPolicy(
            auto_reply_enabled=settings.auto_reply_enabled
        )
    
    # ===== Outbound Operations =====
    
    def send_initial_outreach(self, db: Session, contact: Contact) -> dict:
        """
        Send initial outreach using integrated engine.
        """
        return self.followup_engine.send_initial_outreach(db, contact)
    
    def send_followup(self, db: Session, contact: Contact) -> Optional[dict]:
        """
        Send follow-up using conversation-aware logic.
        """
        return self.followup_engine.send_followup_if_due(db, contact)
    
    def run_followup_cycle(self, db: Session, limit: Optional[int] = None) -> list[dict]:
        """
        Run follow-up cycle for all due contacts.
        """
        return self.followup_engine.run_followup_cycle(db, limit)
    
    # ===== Inbound Operations =====
    
    def process_reply(self, db: Session, email_in: InboundEmail) -> dict:
        """
        Process inbound reply with full semantic intelligence.
        """
        return process_inbound_email_v2(db, email_in)
    
    def classify_reply(
        self,
        campaign: Campaign,
        contact: Contact,
        inbound_body: str,
        conversation_context: str
    ) -> ClassificationResult:
        """
        Classify reply with multi-dimensional semantic analysis.
        """
        return classify_prospect_reply(
            campaign=campaign,
            contact=contact,
            inbound_body=inbound_body,
            conversation_context=conversation_context
        )
    
    # ===== Intelligence Operations =====
    
    def get_next_best_action(
        self,
        contact: Contact,
        classification_result: ClassificationResult,
        has_approved_pricing: bool = False,
        has_case_studies: bool = False
    ):
        """
        Determine next best action based on semantic analysis.
        """
        if not classification_result.success or not classification_result.semantic_intent:
            # Classification failed - escalate
            from mailer_agent.policy.next_action import ActionType, NextBestAction
            return NextBestAction(
                action=ActionType.ESCALATE_HUMAN,
                requires_human_approval=True,
                approval_reason=f"Classification failed: {classification_result.failure_reason}",
                confidence=0.0,
                reasoning="Cannot determine action without successful classification"
            )
        
        return self.policy_engine.determine_next_action(
            contact=contact,
            semantic_intent=classification_result.semantic_intent,
            has_approved_pricing=has_approved_pricing,
            has_case_studies=has_case_studies
        )
    
    def should_send_followup(self, contact: Contact) -> tuple[bool, str]:
        """
        Check if follow-up should send now.
        """
        return self.followup_scheduler.should_send_followup_now(contact)
    
    def reschedule_after_reply(
        self,
        contact: Contact,
        campaign: Campaign,
        classification_result: ClassificationResult
    ) -> Optional[datetime]:
        """
        Reschedule follow-up after prospect reply.
        """
        if not classification_result.success or not classification_result.semantic_intent:
            # Can't reschedule without understanding
            return None
        
        return self.followup_scheduler.reschedule_after_reply(
            contact,
            campaign,
            classification_result.semantic_intent
        )
    
    # ===== Monitoring & Metrics =====
    
    def get_llm_metrics(self) -> dict:
        """
        Get LLM provider metrics.
        """
        return llm_provider.llm_metrics.get_stats()
    
    def get_system_health(self, db: Session) -> dict:
        """
        Get overall system health metrics.
        """
        from mailer_agent.models import ContactStatus, MessageStatus
        
        # Contact state distribution
        contact_states = {}
        for status in ContactStatus:
            count = db.query(Contact).filter(Contact.status == status.value).count()
            contact_states[status.value] = count
        
        # Message statistics
        total_messages = db.query(Message).count()
        sent_messages = db.query(Message).filter(
            Message.status == MessageStatus.SENT.value
        ).count()
        failed_messages = db.query(Message).filter(
            Message.status == MessageStatus.FAILED.value
        ).count()
        
        # Classification success rate
        from mailer_agent.models import Message, MessageDirection
        inbound_messages = db.query(Message).filter(
            Message.direction == MessageDirection.INBOUND.value
        ).all()
        
        classification_success = sum(
            1 for m in inbound_messages if m.classification_success
        )
        classification_total = len(inbound_messages)
        classification_rate = (
            classification_success / classification_total
            if classification_total > 0 else 0
        )
        
        return {
            "contact_states": contact_states,
            "messages": {
                "total": total_messages,
                "sent": sent_messages,
                "failed": failed_messages,
                "success_rate": sent_messages / total_messages if total_messages > 0 else 0
            },
            "classification": {
                "success": classification_success,
                "total": classification_total,
                "success_rate": classification_rate
            },
            "llm_metrics": self.get_llm_metrics(),
            "settings": {
                "live_sending_enabled": settings.live_sending_enabled,
                "auto_reply_enabled": settings.auto_reply_enabled,
                "database_url": settings.database_url.split("@")[-1] if "@" in settings.database_url else "sqlite"
            }
        }
    
    def validate_integration(self) -> dict:
        """
        Validate that all components are properly integrated.
        """
        checks = {}
        
        # Check LLM availability
        checks["llm_available"] = llm_provider.is_llm_available()
        
        # Check settings
        checks["groq_api_key_set"] = bool(settings.groq_api_key)
        checks["smtp_configured"] = bool(settings.smtp_username and settings.smtp_password)
        checks["imap_configured"] = bool(settings.imap_username and settings.imap_password)
        
        # Check module imports
        try:
            from mailer_agent.semantic_models import ClassificationResult, SemanticIntent
            checks["semantic_models_imported"] = True
        except Exception as e:
            checks["semantic_models_imported"] = False
            checks["semantic_models_error"] = str(e)
        
        try:
            from mailer_agent.state_machine import transition_contact_state
            checks["state_machine_imported"] = True
        except Exception as e:
            checks["state_machine_imported"] = False
            checks["state_machine_error"] = str(e)
        
        try:
            from mailer_agent.policy.next_action import NextBestActionPolicy
            checks["policy_engine_imported"] = True
        except Exception as e:
            checks["policy_engine_imported"] = False
            checks["policy_engine_error"] = str(e)
        
        # Check database
        try:
            from mailer_agent.db import engine
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            checks["database_connected"] = True
        except Exception as e:
            checks["database_connected"] = False
            checks["database_error"] = str(e)
        
        # Overall status
        checks["integration_valid"] = all([
            checks.get("llm_available", False),
            checks.get("semantic_models_imported", False),
            checks.get("state_machine_imported", False),
            checks.get("policy_engine_imported", False),
            checks.get("database_connected", False),
        ])
        
        return checks


# Global instance
mailer_agent = MailerAgentCore()


# Convenience functions for backward compatibility
def send_initial_outreach(db: Session, contact: Contact) -> dict:
    """Backward compatible wrapper."""
    return mailer_agent.send_initial_outreach(db, contact)


def send_followup_if_due(db: Session, contact: Contact) -> Optional[dict]:
    """Backward compatible wrapper."""
    return mailer_agent.send_followup(db, contact)


def process_inbound_email(db: Session, email_in: InboundEmail) -> dict:
    """Backward compatible wrapper."""
    return mailer_agent.process_reply(db, email_in)


def run_followup_cycle(db: Session, limit: Optional[int] = None) -> list[dict]:
    """Backward compatible wrapper."""
    return mailer_agent.run_followup_cycle(db, limit)
