"""Prompt facts every agent kind shares (ARCHIE-293)."""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def with_current_datetime(state: dict) -> dict:
    """Add the current date/time/weekday to a copy of `state`, computed fresh
    in the user's timezone at render time.

    Never source these from persisted state: a Redis-cached UserState snapshot
    goes stale the moment it's written, silently freezing the agent's clock.
    """
    try:
        tz = ZoneInfo(state.get("user_timezone") or "UTC")
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")
    now = datetime.now(tz)
    return {
        **state,
        "current_date": now.strftime("%d.%m.%Y"),
        "current_time": now.strftime("%H:%M"),
        "current_weekday": now.strftime("%A"),
    }


def datetime_note(state: dict) -> str:
    """Return the current date/time line for the volatile prompt tail.

    Kept out of the cacheable prefix so minute-level changes don't break the
    prompt cache (ARCHIE-180).
    """
    dt = with_current_datetime(state)
    return (
        f"Current date/time: {dt['current_weekday']}, "
        f"{dt['current_date']}, {dt['current_time']} "
        f"({state.get('user_timezone') or 'UTC'})"
    )
