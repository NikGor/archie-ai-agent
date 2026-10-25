"""Runtime state of the process (ARCHIE-298): services under the supervisor and
today's LLM spend of background kinds."""

from fastapi import APIRouter
from pydantic import BaseModel, Field, JsonValue
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
