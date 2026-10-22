"""Swarm control state and memory in Redis.

- mode and settings of the spontaneous checks (ARCHIE-238);
- long-term memory: lasting facts and preferences, shown in every prompt;
- day memory: what each check saw and reported today, so the swarm does not
  repeat itself; one Redis list per local date, expires after two days;
- per-check state: when a check ran last and until when it has nothing to do.
"""

import json
import logging
import uuid
from datetime import UTC, date, datetime
from archie_shared.swarm.models import (
    SWARM_MEMORY_FACT_MAX_LENGTH,
    SwarmMemoryFact,
    SwarmMode,
    SwarmRunInfo,
    SwarmSettings,
    SwarmWatchTask,
)
from pydantic import BaseModel, Field
from ..backend.redis_factory import get_async_redis


logger = logging.getLogger(__name__)

_MODE_KEY = "archie:swarm:mode"
_SETTINGS_KEY = "archie:swarm:settings"
_MEMORY_KEY = "archie:swarm:memory"
_DAY_KEY = "archie:swarm:day:{day}"
_LAST_TASK_KEY = "archie:swarm:last_task"
_QUIET_KEY = "archie:swarm:quiet_until"
_LAST_RUN_KEY = "archie:swarm:last_run"

_DAY_TTL_SECONDS = 2 * 24 * 3600
MEMORY_MAX_FACTS = 30


class DayNote(BaseModel):
    """One line of the day memory."""

    time: str = Field(description="Local time HH:MM")
    task: SwarmWatchTask = Field(description="The check that wrote the note")
    note: str = Field(description="What the check saw")
    reported: str | None = Field(
        default=None, description="The alert text, if the check reported"
    )


class SwarmMemory:
    def __init__(self) -> None:
        self.redis = get_async_redis()

    async def get_mode(self) -> SwarmMode:
        return await self.redis.get(_MODE_KEY) or "running"

    async def set_mode(self, mode: SwarmMode) -> None:
        await self.redis.set(_MODE_KEY, mode)
        logger.info(f"swarm_memory_001: Mode is \033[35m{mode}\033[0m")

    async def get_settings(self) -> SwarmSettings:
        raw = await self.redis.get(_SETTINGS_KEY)
        return SwarmSettings.model_validate_json(raw) if raw else SwarmSettings()

    async def set_settings(self, settings: SwarmSettings) -> None:
        await self.redis.set(_SETTINGS_KEY, settings.model_dump_json())
        logger.info(f"swarm_memory_002: Settings saved: {settings.model_dump()}")

    async def list_facts(self) -> list[SwarmMemoryFact]:
        raw = await self.redis.hvals(_MEMORY_KEY)
        facts = [SwarmMemoryFact.model_validate_json(item) for item in raw]
        return sorted(facts, key=lambda fact: fact.created_at)

    async def add_fact(self, text: str) -> SwarmMemoryFact | None:
        text = " ".join(text.split())[:SWARM_MEMORY_FACT_MAX_LENGTH]
        facts = await self.list_facts()
        if not text or any(f.text.lower() == text.lower() for f in facts):
            return None
        if len(facts) >= MEMORY_MAX_FACTS:
            # The oldest fact gives way to the new one.
            await self.redis.hdel(_MEMORY_KEY, facts[0].fact_id)
        fact = SwarmMemoryFact(
            fact_id=uuid.uuid4().hex[:12],
            text=text,
            created_at=datetime.now(UTC).isoformat(),
        )
        await self.redis.hset(_MEMORY_KEY, fact.fact_id, fact.model_dump_json())
        logger.info(f"swarm_memory_003: Remembered: \033[33m{text}\033[0m")
        return fact

    async def forget_fact(self, fact_id: str) -> None:
        await self.redis.hdel(_MEMORY_KEY, fact_id)

    async def add_day_note(self, day: date, note: DayNote) -> None:
        key = _DAY_KEY.format(day=day.isoformat())
        await self.redis.rpush(key, note.model_dump_json())
        await self.redis.expire(key, _DAY_TTL_SECONDS)

    async def get_day_notes(self, day: date) -> list[DayNote]:
        raw = await self.redis.lrange(_DAY_KEY.format(day=day.isoformat()), 0, -1)
        return [DayNote.model_validate_json(item) for item in raw]

    async def get_last_task_runs(self) -> dict[str, float]:
        raw = await self.redis.hgetall(_LAST_TASK_KEY)
        return {task: float(ts) for task, ts in raw.items()}

    async def mark_task_run(self, task: SwarmWatchTask, timestamp: float) -> None:
        await self.redis.hset(_LAST_TASK_KEY, task, str(timestamp))

    async def get_quiet_until(self) -> dict[str, datetime]:
        raw = await self.redis.hgetall(_QUIET_KEY)
        return {task: datetime.fromisoformat(value) for task, value in raw.items()}

    async def set_quiet_until(
        self, task: SwarmWatchTask, until: datetime | None
    ) -> None:
        if until is None:
            await self.redis.hdel(_QUIET_KEY, task)
        else:
            await self.redis.hset(_QUIET_KEY, task, until.isoformat())

    async def get_last_run(self) -> SwarmRunInfo | None:
        raw = await self.redis.get(_LAST_RUN_KEY)
        return SwarmRunInfo.model_validate(json.loads(raw)) if raw else None

    async def set_last_run(self, run: SwarmRunInfo) -> None:
        await self.redis.set(_LAST_RUN_KEY, run.model_dump_json())
