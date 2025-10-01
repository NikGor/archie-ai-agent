#!/usr/bin/env python3
import logging

import uvicorn
from fastapi import FastAPI

from app.endpoints import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("=== STEP 1: App Init ===")
app = FastAPI()
app.include_router(router)
logger.info("main_001: FastAPI ready")

if __name__ == "__main__":
    logger.info("main_002: Starting server on \033[36m0.0.0.0:8005\033[0m")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
    )
