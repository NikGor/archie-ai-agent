"""HTTP API of the agent constructor (ARCHIE-341); Django proxies it."""

import logging
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, JsonValue
from app.agents.ephemeral.composer import CompositionResult
from app.agents.systems import ScheduleInfo, system_service
from app.platform.runtime.system import SystemCheck, SystemValidationError
from app.platform.runtime.system_runs import SystemRunRecord


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent-systems", tags=["agent-systems"])


class SpecRequest(BaseModel):
    spec: dict[str, JsonValue] = Field(description="AgentSystemSpec")


class ComposeRequest(BaseModel):
    task: str = Field(min_length=1, max_length=2000)
    topology: (
        Literal["centralized", "workflow", "hierarchical", "decentralized"] | None
    ) = None


class RunRequest(SpecRequest):
    task: str = Field("", max_length=2000, description="Empty: the fixed task")
    demo_mode: bool = Field(True, description="Tools return demo data")


class ScheduleRequest(SpecRequest):
    demo_mode: bool = Field(False, description="Tools return demo data")


@router.get("/catalog")
async def catalog() -> dict[str, JsonValue]:
    return system_service().catalog()


@router.post("/validate")
async def validate(request: SpecRequest) -> SystemCheck:
    return system_service().check(request.spec)


@router.post("/compose")
async def compose(request: ComposeRequest) -> CompositionResult:
    return await system_service().compose(request.task, request.topology)


@router.post("/runs")
async def start_run(request: RunRequest) -> dict[str, JsonValue]:
    try:
        record = system_service().start(request.spec, request.task, request.demo_mode)
    except SystemValidationError as exc:
        raise HTTPException(422, {"errors": exc.errors}) from exc
    except ValueError as exc:
        raise HTTPException(422, {"errors": [str(exc)]}) from exc
    return record.summary()


@router.get("/runs")
async def list_runs(limit: int = 30) -> list[dict[str, JsonValue]]:
    records = await system_service().runtime.recent(min(limit, 100))
    return [r.summary() for r in records]


@router.get("/runs/{system_run_id}")
async def get_run(system_run_id: str) -> SystemRunRecord:
    record = await system_service().runtime.status(system_run_id)
    if record is None:
        raise HTTPException(404, "system run not found")
    return record


@router.post("/runs/{system_run_id}/cancel")
async def cancel_run(system_run_id: str) -> dict[str, bool]:
    return {"cancelled": await system_service().runtime.cancel(system_run_id)}


@router.post("/schedules")
async def create_schedule(request: ScheduleRequest) -> ScheduleInfo:
    try:
        return await system_service().schedule(request.spec, request.demo_mode)
    except SystemValidationError as exc:
        raise HTTPException(422, {"errors": exc.errors}) from exc
    except ValueError as exc:
        raise HTTPException(422, {"errors": [str(exc)]}) from exc


@router.get("/schedules/{job_id}")
async def get_schedule(job_id: str) -> ScheduleInfo:
    try:
        return await system_service().schedule_info(job_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.delete("/schedules/{job_id}")
async def delete_schedule(job_id: str) -> dict[str, bool]:
    try:
        await system_service().unschedule(job_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"deleted": True}
