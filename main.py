#!/usr/bin/env python3
import logging
import uvicorn
from fastapi import FastAPI
from app.endpoints import router

logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(router)

logger.info("Archie AI Agent application started")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
