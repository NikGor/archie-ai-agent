"""Chat monitoring endpoints (ARCHIE-213); Django starts runs and stores results."""

from archie_shared.monitoring.models import MonitorRunInfo, MonitorRunRequest
from fastapi import APIRouter, HTTPException
from .service import (
    MonitorBusyError,
    MonitorConfigError,
    MonitorRunNotFoundError,
    monitor_service,
)


router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.post("/runs", status_code=202)
async def start_run(request: MonitorRunRequest) -> MonitorRunInfo:
    """Judge pending turns in the background; progress goes to Django."""
    try:
        return await monitor_service.start(request)
    except MonitorBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except MonitorConfigError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/runs")
async def list_runs() -> list[MonitorRunInfo]:
    """Runs kept in agent memory, newest first; Django keeps the full history."""
    return monitor_service.list_runs()


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> MonitorRunInfo:
    try:
        return monitor_service.get(run_id)
    except MonitorRunNotFoundError as exc:
        raise HTTPException(404, f"Run {run_id} not found") from exc
