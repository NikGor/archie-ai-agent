#!/usr/bin/env python3
"""
Main entry point for the Archie AI Agent application.
"""

import logging
from fastapi import FastAPI
from app.endpoints import router
from app.config import setup_logging

# Setup logging before anything else
setup_logging(level="INFO")

logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(router)

logger.info("Archie AI Agent application started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
