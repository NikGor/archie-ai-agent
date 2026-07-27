"""Shared Redis client construction.

state_service.py, tool_result_store.py and light_control_tool.py each built
their own Redis client from the same settings.redis_host/port/db — one
factory replaces the copy-pasted connection blocks.
"""

import redis
import redis.asyncio as aioredis

from .. import config


def get_async_redis() -> aioredis.Redis:
    """Build an async Redis client from the app's Redis settings."""
    return aioredis.Redis(
        host=config.settings.redis_host,
        port=config.settings.redis_port,
        db=config.settings.redis_db,
        decode_responses=True,
    )


def get_sync_redis() -> redis.Redis:
    """Build a sync Redis client from the app's Redis settings."""
    return redis.Redis(
        host=config.settings.redis_host,
        port=config.settings.redis_port,
        db=config.settings.redis_db,
        decode_responses=True,
    )
