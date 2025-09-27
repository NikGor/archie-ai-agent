#!/usr/bin/env python3
"""
Main entry point for the Archie AI Agent application.
"""

from fastapi import FastAPI
from app.endpoints import router

app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
