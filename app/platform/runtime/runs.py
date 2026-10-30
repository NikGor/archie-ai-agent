"""Runs of AgentSpec agents: record, events, store and manager (ARCHIE-321, ARCHIE-325).

A run moves created -> configured -> running -> waiting_for_approval |
completed | failed | cancelled. The record holds what was asked, how it ran
and what it cost; the parent gets a summary and the run_id, the full evidence
stays in the record (no retelling between agents).
"""

import asyncio
import hashlib
import json
import logging
import statistics
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol
from pydantic import BaseModel, Field, JsonValue


logger = logging.getLogger(__name__)

RunState = Literal[
    "created",
    "configured",
    "running",
    "waiting_for_approval",
    "completed",
    "failed",
    "cancelled",
]
TERMINAL_STATES: frozenset[str] = frozenset(
    {"waiting_for_approval", "completed", "failed", "cancelled"}
)
EventKind = Literal[
    "task_created",
    "agent_configured",
    "agent_started",
    "tool_invoked",
    "approval_requested",
    "agent_completed",
    "agent_failed",
    "agent_cancelled",
    "budget_exhausted",
    "model_escalated",
    "policy_violation",
    "loop_guard",
]
RUN_TTL_SECONDS = 7 * 86400
RUN_STREAM = "archie:agent:runs"


class RunEvent(BaseModel):
    kind: EventKind = Field(description="What happened")
    at: float = Field(description="Unix time")
    detail: dict[str, JsonValue] = Field(default_factory=dict, description="Facts")


class AgentRunRecord(BaseModel):
    """Everything about one run (§15 of ARCHIE-319)."""

    run_id: str = Field(description="Run identity")
    parent_run_id: str = Field(description="Run or chat turn that asked for it")
    trace_id: str | None = Field(None, description="Langfuse trace of the run")
    idempotency_key: str = Field(description="hash(parent, task, context)")
    source: str = Field(description="Trusted caller identity")
    task: str = Field(description="Original task")
    state: RunState = Field("created", description="Lifecycle state")
    stop_reason: str | None = Field(None, description="Why a run stopped early")
    detail: str = Field("", description="Human-readable stop detail")
    configuration: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Configurator decision, reason, model and cost",
    )
    spec: dict[str, JsonValue] | None = Field(None, description="Validated AgentSpec")
    spec_version: str | None = Field(None, description="AgentSpec format version")
    spec_fingerprint: str | None = Field(None, description="Hash of the spec")
    prompt_version: str | None = Field(None, description="Base prompt + spec hash")
    adjustments: list[str] = Field(default_factory=list, description="Validator caps")
    model: str | None = Field(None, description="Model of the last attempt")
    effort: str | None = Field(None, description="Provider effort of the last attempt")
    model_reason: str = Field("", description="Why this model was selected")
    escalations: list[dict[str, JsonValue]] = Field(
        default_factory=list, description="Profile escalations with reasons"
    )
    events: list[RunEvent] = Field(default_factory=list, description="Run events")
    tools_used: list[str] = Field(default_factory=list, description="Executed tools")
    llm_calls: int = Field(0, description="LLM calls of the agent")
    tool_calls: int = Field(0, description="Executed tool calls")
    iterations: int = Field(0, description="Tool rounds")
    input_tokens: int = Field(0, description="Input tokens incl. configurator")
    output_tokens: int = Field(0, description="Output tokens incl. configurator")
    cost_usd: float = Field(0.0, description="Total USD incl. configurator")
    configurator_cost_usd: float = Field(0.0, description="USD of the configurator")
    retries: int = Field(0, description="Repeated invalid replies")
    policy_violations: list[str] = Field(default_factory=list)
    loop_guards: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    result: dict[str, JsonValue] | None = Field(None, description="Final reply")
    evidence: list[dict[str, JsonValue]] = Field(default_factory=list)
    verification: dict[str, JsonValue] | None = Field(
        None, description="Check of the completion criteria"
    )
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return round(self.finished_at - self.created_at, 3)

    def summary(self) -> dict[str, JsonValue]:
        """What the parent sees: no full evidence, a link by run_id."""
        result = self.result or {}
        return {
            "run_id": self.run_id,
            "state": self.state,
            "stop_reason": self.stop_reason,
            "summary": result.get("summary") or self.detail,
            "findings": result.get("findings") or [],
            "answer": result.get("answer") or "",
            "proposed_actions": result.get("proposed_actions") or [],
            "cost_usd": round(self.cost_usd, 5),
            "evidence_items": len(self.evidence),
        }


class RunStore(Protocol):
    async def save(self, record: AgentRunRecord) -> None: ...
    async def get(self, run_id: str) -> AgentRunRecord | None: ...
    async def recent(self, limit: int) -> list[AgentRunRecord]: ...


class MemoryRunStore:
    def __init__(self) -> None:
        self.records: dict[str, AgentRunRecord] = {}

    async def save(self, record: AgentRunRecord) -> None:
        self.records[record.run_id] = record.model_copy(deep=True)

    async def get(self, run_id: str) -> AgentRunRecord | None:
        return self.records.get(run_id)

    async def recent(self, limit: int) -> list[AgentRunRecord]:
        ordered = sorted(self.records.values(), key=lambda r: -r.created_at)
        return ordered[:limit]


class RedisRunStore:
    """Record per key with TTL, an index by time and the stream of finished runs."""

    def __init__(self, redis, prefix: str = "archie:agent:run:") -> None:
        self.redis = redis
        self.prefix = prefix
        self.index = f"{prefix}index"

    async def save(self, record: AgentRunRecord) -> None:
        data = record.model_dump_json()
        await self.redis.set(self.prefix + record.run_id, data, ex=RUN_TTL_SECONDS)
        await self.redis.zadd(self.index, {record.run_id: record.created_at})
        await self.redis.zremrangebyscore(self.index, 0, time.time() - RUN_TTL_SECONDS)
        if record.state in TERMINAL_STATES:
            await self.redis.xadd(
                RUN_STREAM,
                {"data": data},
                minid=str(int((time.time() - RUN_TTL_SECONDS) * 1000)),
                approximate=True,
            )

    async def get(self, run_id: str) -> AgentRunRecord | None:
        raw = await self.redis.get(self.prefix + run_id)
        return AgentRunRecord.model_validate_json(raw) if raw else None

    async def recent(self, limit: int) -> list[AgentRunRecord]:
        ids = await self.redis.zrevrange(self.index, 0, limit - 1)
        records = []
        for run_id in ids:
            record = await self.get(
                run_id.decode() if isinstance(run_id, bytes) else run_id
            )
            if record:
                records.append(record)
        return records


EventListener = Callable[[RunEvent], Awaitable[None]]


class RunRecorder:
    """Writes events and state of one run; the engine never touches the store."""

    def __init__(self, record: AgentRunRecord, store: RunStore) -> None:
        self.record = record
        self.store = store
        self.listener: EventListener | None = None
        self._pending: set[asyncio.Task] = set()

    def event(self, kind: EventKind, **detail: JsonValue) -> None:
        event = RunEvent(kind=kind, at=time.time(), detail=detail)
        self.record.events.append(event)
        if self.listener is not None:
            # Progress for the caller (WS status); never blocks or fails the run
            task = asyncio.get_running_loop().create_task(self._notify(event))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        logger.info(
            f"agent_run_001: \033[36m{self.record.run_id}\033[0m {kind} {json.dumps(detail, ensure_ascii=False, default=str)[:300]}"
        )

    async def _notify(self, event: RunEvent) -> None:
        try:
            await self.listener(event)
        except Exception as exc:
            logger.warning(f"agent_run_warning_001: Listener failed: {exc}")

    def state(self, state: RunState) -> None:
        self.record.state = state
        if state == "running" and self.record.started_at is None:
            self.record.started_at = time.time()
        if state in TERMINAL_STATES:
            self.record.finished_at = time.time()

    async def save(self) -> None:
        try:
            await self.store.save(self.record)
        except Exception as exc:
            logger.error(
                f"agent_run_error_001: Could not save run {self.record.run_id}: {exc}"
            )


def idempotency_key(parent_run_id: str, task: str, context: str) -> str:
    raw = json.dumps([parent_run_id, task, context], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


RunBody = Callable[[RunRecorder], Awaitable[None]]


class RunManager:
    """Runs of this process: start, idempotent repeat, status, cancel, stats."""

    def __init__(self, store: RunStore | None = None, keep: int = 200) -> None:
        self.store = store or MemoryRunStore()
        self.keep = keep
        self.recorders: dict[str, RunRecorder] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.keys: dict[str, str] = {}

    def existing(self, key: str) -> RunRecorder | None:
        run_id = self.keys.get(key)
        return self.recorders.get(run_id) if run_id else None

    def create(
        self, task: str, parent_run_id: str, source: str, context: str = ""
    ) -> tuple[RunRecorder, bool]:
        """A new recorder, or the existing one for the same parent, task and context."""
        key = idempotency_key(parent_run_id, task, context)
        found = self.existing(key)
        if found is not None:
            logger.info(
                f"agent_run_002: Repeated delegation returns {found.record.run_id}"
            )
            return found, True
        record = AgentRunRecord(
            run_id=uuid.uuid4().hex[:16],
            parent_run_id=parent_run_id,
            idempotency_key=key,
            source=source,
            task=task,
        )
        recorder = RunRecorder(record, self.store)
        recorder.event("task_created", source=source, parent_run_id=parent_run_id)
        self.recorders[record.run_id] = recorder
        self.keys[key] = record.run_id
        self._trim()
        return recorder, False

    def _trim(self) -> None:
        while len(self.recorders) > self.keep:
            run_id = next(iter(self.recorders))
            if run_id in self.tasks:
                break
            recorder = self.recorders.pop(run_id)
            self.keys.pop(recorder.record.idempotency_key, None)

    async def run(self, recorder: RunRecorder, body: RunBody) -> AgentRunRecord:
        """Run `body` as its own task; cancelling the waiter cancels the run."""
        run_id = recorder.record.run_id
        task = self.tasks.get(run_id)
        if task is None and recorder.record.state not in TERMINAL_STATES:
            task = asyncio.create_task(
                self._guard(recorder, body), name=f"run-{run_id}"
            )
            self.tasks[run_id] = task
        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise
        return recorder.record

    async def _guard(self, recorder: RunRecorder, body: RunBody) -> None:
        try:
            await body(recorder)
        except asyncio.CancelledError:
            if recorder.record.state not in TERMINAL_STATES:
                recorder.state("cancelled")
                recorder.record.stop_reason = "cancelled"
                recorder.event("agent_cancelled")
        except Exception as exc:
            logger.exception(
                f"agent_run_error_002: Run {recorder.record.run_id}: {exc}"
            )
            recorder.record.errors.append(f"{type(exc).__name__}: {exc}")
            recorder.record.stop_reason = "error"
            recorder.record.detail = f"{type(exc).__name__}: {exc}"
            recorder.state("failed")
            recorder.event("agent_failed", reason="error")
        finally:
            self.tasks.pop(recorder.record.run_id, None)
            await recorder.save()

    async def status(self, run_id: str) -> AgentRunRecord | None:
        recorder = self.recorders.get(run_id)
        return recorder.record if recorder else await self.store.get(run_id)

    async def cancel(self, run_id: str) -> bool:
        task = self.tasks.get(run_id)
        if task is None:
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def recent(self, limit: int = 50) -> list[AgentRunRecord]:
        local = {r.record.run_id: r.record for r in self.recorders.values()}
        stored = {r.run_id: r for r in await self.store.recent(limit)}
        merged = {**stored, **local}
        return sorted(merged.values(), key=lambda r: -r.created_at)[:limit]

    def stats(self) -> dict[str, JsonValue]:
        """Aggregates over the runs this process holds (ARCHIE-325)."""
        records = [
            r.record
            for r in self.recorders.values()
            if r.record.state in TERMINAL_STATES
        ]
        succeeded = [
            r for r in records if r.state in ("completed", "waiting_for_approval")
        ]
        latencies = sorted(r.seconds for r in records if r.seconds is not None)

        def percentile(share: float) -> float | None:
            if not latencies:
                return None
            return latencies[min(len(latencies) - 1, int(share * len(latencies)))]

        return {
            "runs": len(records),
            "active": len(self.tasks),
            "success_rate": round(len(succeeded) / len(records), 3)
            if records
            else None,
            "cost_per_success_usd": round(
                sum(r.cost_usd for r in records) / len(succeeded), 5
            )
            if succeeded
            else None,
            "latency_p50_seconds": statistics.median(latencies) if latencies else None,
            "latency_p95_seconds": percentile(0.95),
            "policy_violation_share": round(
                sum(1 for r in records if r.policy_violations) / len(records), 3
            )
            if records
            else None,
        }


_manager: RunManager | None = None


def register_run_manager(manager: RunManager) -> None:
    """The composition root registers the process RunManager for the API."""
    global _manager
    _manager = manager


def run_manager() -> RunManager | None:
    return _manager
