#!/usr/bin/env python3
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv


load_dotenv()

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from app.backend.cron_scheduler import cron_scheduler  # noqa: E402
from app.endpoints import router  # noqa: E402
from app.ws_docs import router as ws_docs_router  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("=== STEP 1: App Init ===")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await cron_scheduler.start()
    try:
        yield
    finally:
        await cron_scheduler.stop()


app = FastAPI(
    title="Archie AI Agent",
    description="3-stage orchestration pipeline with Schema-Guided Reasoning (SGR)",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
app.include_router(ws_docs_router)
logger.info("main_001: FastAPI ready")

if __name__ == "__main__":
    logger.info("main_002: Starting server on \033[36m0.0.0.0:8005\033[0m")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
    )
