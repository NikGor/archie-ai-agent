"""Prompt eval endpoints: case catalog, run control and live progress (ARCHIE-208)."""

import contextlib
import logging
from archie_shared.evals.models import (
    EvalCaseInfo,
    EvalRunInfo,
    EvalRunRequest,
    EvalSnapshotEvent,
)
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from app.config import MODEL_PROVIDERS, settings
from .catalog import CatalogError, get_catalog
from .service import EvalBusyError, EvalRunNotFoundError, eval_service


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evals", tags=["evals"])
ws_router = APIRouter()

_KNOWN_MODELS = {model for models in MODEL_PROVIDERS.values() for model in models}


@router.get("/cases")
async def list_cases() -> list[EvalCaseInfo]:
    return list(get_catalog())


@router.get("/models")
async def list_models() -> dict[str, list[str] | str]:
    """Models a run can use, and the default one."""
    return {"models": sorted(_KNOWN_MODELS), "default": settings.eval_default_model}


@router.post("/runs", status_code=202)
async def start_run(request: EvalRunRequest) -> EvalRunInfo:
    unknown_models = set(request.models) - _KNOWN_MODELS
    if unknown_models:
        raise HTTPException(422, f"Unknown models: {sorted(unknown_models)}")
    try:
        return await eval_service.start(request)
    except EvalBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except CatalogError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/runs")
async def list_runs() -> list[EvalRunInfo]:
    """Runs kept in agent memory, newest first; Django keeps the full history."""
    return eval_service.list_runs()


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> EvalSnapshotEvent:
    try:
        return eval_service.get(run_id).snapshot()
    except EvalRunNotFoundError as exc:
        raise HTTPException(404, f"Run {run_id} not found") from exc


@router.delete("/runs/{run_id}")
async def cancel_run(run_id: str) -> EvalRunInfo:
    try:
        return await eval_service.cancel(run_id)
    except EvalRunNotFoundError as exc:
        raise HTTPException(404, f"Run {run_id} not found") from exc


@ws_router.websocket("/ws_evals/{run_id}")
async def ws_evals(websocket: WebSocket, run_id: str) -> None:
    """Sends a snapshot, then live events until the run finishes."""
    await websocket.accept()
    try:
        queue = eval_service.subscribe(run_id)
    except EvalRunNotFoundError:
        await websocket.send_json(
            {"type": "error", "message": f"Run {run_id} not found"}
        )
        await websocket.close()
        return
    logger.info(f"eval_ws_001: Client subscribed to run \033[36m{run_id}\033[0m")
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event.model_dump(mode="json"))
            if event.type == "run_finished" or (
                event.type == "snapshot" and event.run.finished_at is not None
            ):
                break
        await websocket.close()
    except WebSocketDisconnect:
        logger.info(f"eval_ws_002: Client left run \033[36m{run_id}\033[0m")
    finally:
        eval_service.unsubscribe(run_id, queue)
        with contextlib.suppress(RuntimeError):
            await websocket.close()
