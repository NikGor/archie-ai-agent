"""Runtime state of the process (ARCHIE-298): services under the supervisor and
today's LLM spend of background kinds."""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from .platform.runtime.budget import daily_spend
from .platform.runtime.supervisor import ServiceStatus, Supervisor


router = APIRouter()
supervisor = Supervisor()


class RuntimeStatus(BaseModel):
    services: list[ServiceStatus] = Field(description="Services under the supervisor")
    daily_spend_usd: dict[str, float] = Field(description="Kind -> LLM spend today")


@router.get("/runtime/status", response_model=RuntimeStatus)
async def runtime_status() -> RuntimeStatus:
    return RuntimeStatus(
        services=supervisor.status(),
        daily_spend_usd={kind: round(cost, 4) for kind, cost in daily_spend.by_kind.items()},
    )
