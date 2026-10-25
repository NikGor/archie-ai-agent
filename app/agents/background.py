"""Composition of background kinds; individual kinds do not import each other."""

import json
import time
from archie_shared.swarm.models import SwarmUsage
from pydantic import BaseModel
from app.agents.ephemeral.models import EphemeralResult, SpawnRequest
from app.agents.ephemeral.runner import Completion, complete, estimate_call
from app.agents.ephemeral.service import EphemeralService
from app.agents.swarm.service import swarm_service
from app.backend.redis_factory import get_async_redis
from app.config import settings
from app.platform.runtime.daily_budget import DailyBudget
from app.platform.runtime.spawn import register_spawn_handler
from app.platform.runtime.status import register_status


async def publish_ephemeral(request: SpawnRequest, result: EphemeralResult) -> None:
    redis = swarm_service.bus.redis
    await redis.xadd(
        "archie:ephemeral:runs",
        {
            "data": json.dumps(
                {
                    "request": request.model_dump(mode="json"),
                    "result": result.model_dump(mode="json"),
                },
                ensure_ascii=False,
            )
        },
        minid=str(int((time.time() - 8 * 86400) * 1000)),
        approximate=True,
    )
    await swarm_service.bus.post(
        sender="ephemeral",
        kind="system",
        mentions=[],
        paired=False,
        text=f"Диагностика ({result.status}): {result.summary}\n"
        + "\n".join(result.findings)
        + "\n"
        + result.limitation,
        run_id=result.parent_run_id,
        usage=SwarmUsage(
            model=settings.ephemeral_model,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_cost=result.usage.total_cost,
        ),
    )


def configure_ephemeral() -> EphemeralService:
    ledger = DailyBudget(
        get_async_redis(),
        "ephemeral",
        settings.default_timezone,
        settings.ephemeral_daily_cost_usd,
    )

    async def bounded_call(
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        model: str,
        max_output_tokens: int,
    ) -> Completion:
        _, cost_ceiling = estimate_call(messages, schema, model, max_output_tokens)
        reservation = await ledger.reserve(cost_ceiling, ttl=120)
        result = await complete(messages, schema, model, max_output_tokens)
        await ledger.settle(reservation, result.trace.total_cost or 0.0)
        return result

    service = EphemeralService(
        publisher=publish_ephemeral, call=bounded_call, model=settings.ephemeral_model
    )
    register_spawn_handler(service.spawn)

    async def status() -> dict:
        return {
            "active": service.gate.active,
            "max_concurrent": service.gate.max_concurrent,
            "daily_spend_usd": await ledger.spent(),
            "daily_limit_usd": settings.ephemeral_daily_cost_usd,
            "ha_logs_available": False,
        }

    register_status("ephemeral", status)
    return service
