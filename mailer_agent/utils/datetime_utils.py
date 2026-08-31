"""
Timezone-aware datetime utilities.

All datetimes in the Mailer Agent should be timezone-aware (UTC).
Using naive datetimes (datetime.utcnow()) is a footgun: Python treats
naive datetimes as local time in some contexts, leading to subtle bugs
when the server timezone differs from UTC.

Rule: use utcnow() from this module everywhere instead of datetime.utcnow().
The returned datetime is always timezone-aware (tzinfo=UTC).

For scheduling against a Campaign.timezone, use campaign_local_now().
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional
import logging

logger = logging.getLogger("mailer_agent.utils.datetime_utils")

# Re-export for convenient importing
UTC = timezone.utc


def utcnow() -> datetime:
    """
    Return the current time as a timezone-aware UTC datetime.

    Drop-in replacement for datetime.utcnow() that includes tzinfo=UTC.
    SQLAlchemy stores this correctly in both SQLite and PostgreSQL.
    """
    return datetime.now(tz=UTC)


def as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Ensure a datetime is timezone-aware (UTC).

    - If dt is None, returns None.
    - If dt is already aware (has tzinfo), converts to UTC.
    - If dt is naive, ASSUMES it is UTC and attaches tzinfo=UTC.
      (This handles existing DB values stored as naive UTC datetimes.)
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(UTC)
    # Naive — treat as UTC (backward-compat with existing DB rows).
    return dt.replace(tzinfo=UTC)


def campaign_local_now(timezone_str: str) -> datetime:
    """
    Return the current time in the campaign's configured timezone.

    Used for business-hours checks (e.g., "don't send at 2 AM prospect time").

    Args:
        timezone_str: IANA timezone string from Campaign.timezone,
                      e.g. "America/New_York", "Asia/Kolkata", "UTC".

    Returns:
        timezone-aware datetime in the campaign's local timezone.
        Falls back to UTC if the timezone string is invalid.
    """
    try:
        # zoneinfo is stdlib in Python 3.9+
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            tz = ZoneInfo(timezone_str)
        except (ZoneInfoNotFoundError, KeyError):
            logger.warning(
                "Unknown timezone %r in campaign config, falling back to UTC", timezone_str
            )
            tz = UTC
    except ImportError:
        # Python < 3.9: fall back to UTC (acceptable degradation)
        logger.warning(
            "zoneinfo not available (Python < 3.9), ignoring campaign timezone %r, using UTC",
            timezone_str,
        )
        tz = UTC

    return datetime.now(tz=tz)


def is_business_hours(
    dt: datetime,
    start_hour: int = 8,
    end_hour: int = 18,
    exclude_weekends: bool = True,
) -> bool:
    """
    Check if a (timezone-aware) datetime falls within business hours.

    Args:
        dt: A timezone-aware datetime (typically from campaign_local_now()).
        start_hour: First hour of business day (default 8 = 08:00).
        end_hour: Last hour of business day exclusive (default 18 = 18:00).
        exclude_weekends: If True, Saturdays and Sundays return False.

    Returns:
        True if dt is within business hours.
    """
    if dt.tzinfo is None:
        logger.warning("is_business_hours called with naive datetime — assuming UTC")
        dt = dt.replace(tzinfo=UTC)

    if exclude_weekends and dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False

    return start_hour <= dt.hour < end_hour


def next_business_hour_utc(campaign_timezone: str, start_hour: int = 8) -> datetime:
    """
    Return the next business-hours start time (in UTC) for the given campaign timezone.

    If it's already business hours, returns utcnow().

    Used by the scheduler to defer sends until the prospect's local morning
    rather than sending at 3 AM their time.
    """
    local_now = campaign_local_now(campaign_timezone)

    if is_business_hours(local_now):
        return utcnow()

    # Advance to next weekday 08:00 local
    candidate = local_now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    if candidate <= local_now:
        candidate += timedelta(days=1)

    # Skip weekends
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)

    # Convert back to UTC
    return candidate.astimezone(UTC)
