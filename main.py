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
from app.agents.chat.service import ChatService  # noqa: E402
from app.agents.swarm.bridge import SwarmHubBridge  # noqa: E402
from app.agents.registry import load_agent_kinds  # noqa: E402
from app.runtime_endpoints import router as runtime_router  # noqa: E402
from app.runtime_endpoints import supervisor  # noqa: E402
from app.ws_docs import router as ws_docs_router  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("=== STEP 1: App Init ===")
init_tracing()


# Composition root (ARCHIE-298): the supervisor starts, watches and stops the
# long-lived services of this process
supervisor.register(ChatService())
if settings.runtime_mode == "api":
    # Swarm and cron run in the runtime process (python -m app.runtime_main);
    # this process relays swarm events and commands through Redis (ARCHIE-299)
    swarm_service.bridge = SwarmHubBridge(swarm_service.hub, role="api")
    supervisor.register(swarm_service.bridge)
else:
    supervisor.register(cron_scheduler)
    if settings.swarm_enabled:
        supervisor.register(swarm_service)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_agent_kinds()
    await supervisor.start()
    try:
        yield
    finally:
        await supervisor.stop()
        if settings.swarm_enabled or settings.runtime_mode == "api":
            await swarm_service.close()
        await cron_scheduler.close()
        shutdown_tracing()


app = FastAPI(
    title="Archie AI Agent",
    description="3-stage orchestration pipeline with Schema-Guided Reasoning (SGR)",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
app.include_router(runtime_router)
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
