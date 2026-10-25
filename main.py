#!/usr/bin/env python3
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv


load_dotenv()

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from app.backend.cron_scheduler import cron_scheduler  # noqa: E402
from app.config import settings  # noqa: E402
from app.endpoints import router  # noqa: E402
from app.evals.endpoints import router as evals_router  # noqa: E402
from app.evals.endpoints import ws_router as evals_ws_router  # noqa: E402
from app.monitoring.endpoints import router as monitoring_router  # noqa: E402
from app.agents.swarm.endpoints import router as swarm_router  # noqa: E402
from app.agents.swarm.service import swarm_service  # noqa: E402
from app.utils.tracing import init_tracing, shutdown_tracing  # noqa: E402
from app.ws_docs import router as ws_docs_router  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("=== STEP 1: App Init ===")
init_tracing()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await cron_scheduler.start()
    if settings.swarm_enabled:
        await swarm_service.start()
    try:
        yield
    finally:
        if settings.swarm_enabled:
            await swarm_service.stop()
        await cron_scheduler.stop()
        shutdown_tracing()


app = FastAPI(
    title="Archie AI Agent",
    description="3-stage orchestration pipeline with Schema-Guided Reasoning (SGR)",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
app.include_router(ws_docs_router)
app.include_router(swarm_router)
app.include_router(evals_router)
app.include_router(evals_ws_router)
app.include_router(monitoring_router)
logger.info("main_001: FastAPI ready")

if __name__ == "__main__":
    logger.info("main_002: Starting server on \033[36m0.0.0.0:8005\033[0m")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
    )
