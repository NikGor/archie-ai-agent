"""Runtime state of the process (ARCHIE-298): services under the supervisor and
today's LLM spend of background kinds."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, JsonValue
from app.platform.runtime.runs import AgentRunRecord, RunManager, run_manager
from app.platform.runtime.status import agent_status
from .backend.redis_factory import get_async_redis
from .platform.runtime.budget import daily_spend
from .platform.runtime.supervisor import ServiceStatus, Supervisor


router = APIRouter()
supervisor = Supervisor()

# The runtime process writes its status here (ARCHIE-299)
RUNTIME_STATUS_KEY = "archie:runtime:status"
RUNTIME_STATUS_TTL_SECONDS = 120


class RuntimeStatus(BaseModel):
    agent_kinds: dict[str, dict[str, JsonValue]] = Field(
        default_factory=dict,
        description="State and persistent budgets of background kinds",
    )
    services: list[ServiceStatus] = Field(description="Services under the supervisor")
    daily_spend_usd: dict[str, float] = Field(description="Kind -> LLM spend today")
    runtime_process: "RuntimeStatus | None" = Field(
        default=None,
        description="Status of the archie-runtime process (RUNTIME_MODE=api), if it reports",
    )


def local_status() -> RuntimeStatus:
    return RuntimeStatus(
        services=supervisor.status(),
        daily_spend_usd={
            kind: round(cost, 4) for kind, cost in daily_spend.by_kind.items()
        },
    )


async def publish_status(status: RuntimeStatus) -> None:
    redis = get_async_redis()
    try:
        await redis.set(
            RUNTIME_STATUS_KEY, status.model_dump_json(), ex=RUNTIME_STATUS_TTL_SECONDS
        )
    finally:
        await redis.aclose()


@router.get("/runtime/status", response_model=RuntimeStatus)
async def runtime_status() -> RuntimeStatus:
    status = local_status()
    status.agent_kinds = await agent_status()
    redis = get_async_redis()
    try:
        raw = await redis.get(RUNTIME_STATUS_KEY)
    finally:
        await redis.aclose()
    if raw:
        status.runtime_process = RuntimeStatus.model_validate_json(raw)
    return status


class RunList(BaseModel):
    runs: list[dict[str, JsonValue]] = Field(description="Run summaries, newest first")


def _manager() -> RunManager:
    manager = run_manager()
    if manager is None:
        raise HTTPException(503, "Dynamic agents are not configured in this process")
    return manager


@router.get("/runtime/runs", response_model=RunList)
async def list_runs(limit: int = 50) -> RunList:
    """Runs of dynamic agents (ARCHIE-325): summary, model, cost, state."""
    records = await _manager().recent(min(max(limit, 1), 200))
    return RunList(
        runs=[
            {
                **record.summary(),
                "task": record.task,
                "parent_run_id": record.parent_run_id,
                "trace_id": record.trace_id,
                "model": record.model,
                "created_at": record.created_at,
                "seconds": record.seconds,
            }
            for record in records
        ]
    )


@router.get("/runtime/runs/{run_id}", response_model=AgentRunRecord)
async def get_run(run_id: str) -> AgentRunRecord:
    record = await _manager().status(run_id)
    if record is None:
        raise HTTPException(404, f"Unknown run {run_id}")
    return record


@router.post("/runtime/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, JsonValue]:
    manager = _manager()
    if await manager.status(run_id) is None:
        raise HTTPException(404, f"Unknown run {run_id}")
    cancelled = await manager.cancel(run_id)
    record = await manager.status(run_id)
    return {
        "run_id": run_id,
        "cancelled": cancelled,
        "state": record.state if record else None,
    }
