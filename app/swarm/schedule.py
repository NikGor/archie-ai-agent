"""Time rules of the spontaneous swarm checks (ARCHIE-238): pure functions."""

import random
from datetime import datetime, time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from archie_shared.swarm.models import SwarmSettings, SwarmWatchTask
from ..config import settings as app_settings


def local_now(timezone: str | None) -> datetime:
    try:
        zone = ZoneInfo(timezone or app_settings.default_timezone)
    except ZoneInfoNotFoundError:
        zone = ZoneInfo("UTC")
    return datetime.now(zone)


def parse_clock(value: str) -> time:
    hours, minutes = value.split(":")
    return time(int(hours), int(minutes))


def in_window(now: datetime, start: str, end: str) -> bool:
    """True if `now` is in [start, end). end 00:00 means midnight; end before
    start means the window goes over midnight."""
    current, first, last = now.time(), parse_clock(start), parse_clock(end)
    if first == last:
        return True
    if first < last:
        return first <= current < last
    return current >= first or current < last


def is_active(now: datetime, swarm_settings: SwarmSettings) -> bool:
    return in_window(now, swarm_settings.active_from, swarm_settings.active_until)


def voice_allowed(now: datetime, swarm_settings: SwarmSettings) -> bool:
    start = (
        swarm_settings.voice_from_weekend
        if now.weekday() >= 5
        else swarm_settings.voice_from_weekday
    )
    return in_window(now, start, swarm_settings.voice_until)


def pick_delay_seconds(swarm_settings: SwarmSettings) -> float:
    """Random pause, so the checks do not run on a fixed schedule."""
    return 60 * random.uniform(
        swarm_settings.interval_min_minutes, swarm_settings.interval_max_minutes
    )


def parse_quiet_until(value: str, now: datetime) -> datetime | None:
    """HH:MM later today in the user's time zone, else None."""
    try:
        until = datetime.combine(now.date(), parse_clock(value.strip()), now.tzinfo)
    except ValueError:
        return None
    return until if until > now else None


def due_tasks(
    swarm_settings: SwarmSettings,
    now: datetime,
    quiet_until: dict[str, datetime],
) -> list[SwarmWatchTask]:
    """Enabled checks that are not waiting for a later time. Football needs
    at least one team."""
    return [
        task
        for task in swarm_settings.tasks
        if not (task in quiet_until and quiet_until[task] > now)
        and (task != "football" or swarm_settings.football_teams)
    ]


def pick_task(
    tasks: list[SwarmWatchTask], last_runs: dict[str, float]
) -> SwarmWatchTask | None:
    """The check that waited longest; a random one among never-run checks."""
    if not tasks:
        return None
    never = [task for task in tasks if task not in last_runs]
    if never:
        return random.choice(never)
    return min(tasks, key=lambda task: last_runs[task])
