"""Runs of agent systems (ARCHIE-339): one record per system run with its
steps; every step is an ordinary agent run (`runs.py`) whose parent is the
system run, so the full evidence stays in the agent run records."""

import time
import uuid
from typing import Literal, Protocol
from pydantic import BaseModel, Field, JsonValue
from app.platform.runtime.runs import RUN_TTL_SECONDS


SystemRunState = Literal[
    "running", "waiting_for_approval", "completed", "failed", "cancelled"
]
SYSTEM_TERMINAL_STATES: frozenset[str] = frozenset(
    {"waiting_for_approval", "completed", "failed", "cancelled"}
)
SystemEventKind = Literal[
    "system_started",
    "node_configured",
    "step_started",
    "step_finished",
    "system_completed",
    "system_failed",
    "system_cancelled",
]


class SystemEvent(BaseModel):
    kind: SystemEventKind = Field(description="What happened")
    at: float = Field(description="Unix time")
    detail: dict[str, JsonValue] = Field(default_factory=dict, description="Facts")


class SystemStep(BaseModel):
    node_id: str = Field(description="Node of the system")
    phase: str = Field(description="plan, work, synthesis, round_0, aggregate, step")
    run_id: str = Field(description="Agent run of this step")
    state: str = Field(description="State of the agent run")
    stop_reason: str | None = Field(None, description="Why the agent run stopped")
    cost_usd: float = Field(0.0, description="Cost of the agent run")
    seconds: float | None = Field(None, description="Duration of the agent run")
    summary: str = Field("", description="Summary of the agent result")


class SystemRunRecord(BaseModel):
    system_run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str = Field(description="System name")
    task: str = Field(description="Task of this run")
    trigger: Literal["manual", "schedule"] = Field("manual")
    demo_mode: bool = Field(False, description="Tools returned demo data")
    spec: dict[str, JsonValue] = Field(description="Validated AgentSystemSpec")
    configured: dict[str, JsonValue] = Field(
        default_factory=dict, description="Specs the configurator built (dynamic)"
    )
    state: SystemRunState = "running"
    stop_reason: str | None = None
    detail: str = ""
    steps: list[SystemStep] = Field(default_factory=list)
    events: list[SystemEvent] = Field(default_factory=list)
    result: dict[str, JsonValue] | None = Field(None, description="Final reply")
    proposed_actions: list[dict[str, JsonValue]] = Field(default_factory=list)
    cost_usd: float = 0.0
    created_at: float = Field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return round(self.finished_at - self.created_at, 3)

    def event(self, kind: SystemEventKind, **detail: JsonValue) -> None:
        self.events.append(SystemEvent(kind=kind, at=time.time(), detail=detail))

    def finish(self, state: SystemRunState, reason: str | None = None) -> None:
        self.state = state
        self.stop_reason = reason
        self.finished_at = time.time()

    def summary(self) -> dict[str, JsonValue]:
        return {
            "system_run_id": self.system_run_id,
            "name": self.name,
            "task": self.task[:200],
            "trigger": self.trigger,
            "demo_mode": self.demo_mode,
            "topology": self.spec.get("topology"),
            "state": self.state,
            "stop_reason": self.stop_reason,
            "steps": len(self.steps),
            "cost_usd": round(self.cost_usd, 5),
            "seconds": self.seconds,
            "created_at": self.created_at,
        }


class SystemRunStore(Protocol):
    async def save(self, record: SystemRunRecord) -> None: ...

    async def get(self, system_run_id: str) -> SystemRunRecord | None: ...

    async def recent(self, limit: int) -> list[SystemRunRecord]: ...


class MemorySystemRunStore:
    def __init__(self) -> None:
        self.records: dict[str, SystemRunRecord] = {}

    async def save(self, record: SystemRunRecord) -> None:
        self.records[record.system_run_id] = record.model_copy(deep=True)

    async def get(self, system_run_id: str) -> SystemRunRecord | None:
        return self.records.get(system_run_id)

    async def recent(self, limit: int) -> list[SystemRunRecord]:
        return sorted(self.records.values(), key=lambda r: -r.created_at)[:limit]


class RedisSystemRunStore:
    """Record per key with TTL and an index by time, like agent runs."""

    def __init__(self, redis, prefix: str = "archie:agent:system_run:") -> None:
        self.redis = redis
        self.prefix = prefix
        self.index = f"{prefix}index"

    async def save(self, record: SystemRunRecord) -> None:
        await self.redis.set(
            self.prefix + record.system_run_id,
            record.model_dump_json(),
            ex=RUN_TTL_SECONDS,
        )
        await self.redis.zadd(self.index, {record.system_run_id: record.created_at})
        await self.redis.zremrangebyscore(self.index, 0, time.time() - RUN_TTL_SECONDS)

    async def get(self, system_run_id: str) -> SystemRunRecord | None:
        raw = await self.redis.get(self.prefix + system_run_id)
        return SystemRunRecord.model_validate_json(raw) if raw else None

    async def recent(self, limit: int) -> list[SystemRunRecord]:
        ids = await self.redis.zrevrange(self.index, 0, limit - 1)
        records = []
        for run_id in ids:
            record = await self.get(
                run_id.decode() if isinstance(run_id, bytes) else run_id
            )
            if record:
                records.append(record)
        return records
