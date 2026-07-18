"""Session-scoped store for tool results in Redis (ARCHIE-157).

Persists ToolResult objects per conversation so the agent can reuse results
of earlier tool calls across separate requests within the same dialog.
"""

import json
import logging
import redis
import redis.asyncio as aioredis
from ..config import settings
from ..models.tool_models import ToolResult


logger = logging.getLogger(__name__)


class ToolResultStore:
    """Persist and load tool results per conversation for cross-request context."""

    def __init__(self) -> None:
        self.enabled = settings.tool_result_cache_enabled
        self.ttl = settings.tool_result_cache_ttl
        self.max_items = settings.tool_result_cache_max_items
        self.redis_client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )

    @staticmethod
    def _build_key(conversation_id: str | None, user_name: str | None) -> str | None:
        """Prefer conversation scope, fall back to user scope."""
        if conversation_id:
            return f"tool_results:conversation:{conversation_id}"
        if user_name:
            return f"tool_results:user:{user_name}"
        return None

    def _dedupe(self, results: list[ToolResult]) -> list[ToolResult]:
        """Drop duplicate results and keep only the most recent max_items.

        Deduping by (tool_name, output) makes replays idempotent: re-running the
        same request produces the same results, which collapse to one entry.
        """
        seen: set[str] = set()
        unique: list[ToolResult] = []
        for result in results:
            fingerprint = (
                f"{result.tool_name}:"
                f"{json.dumps(result.output, sort_keys=True, ensure_ascii=False)}"
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append(result)
        return unique[-self.max_items :]

    async def load(
        self, conversation_id: str | None, user_name: str | None
    ) -> list[ToolResult]:
        """Load persisted results for the session. Returns [] when disabled/empty."""
        key = self._build_key(conversation_id, user_name)
        if not self.enabled or not key:
            return []
        try:
            raw = await self.redis_client.get(key)
            if not raw:
                return []
            results = [ToolResult(**item) for item in json.loads(raw)]
            logger.info(
                f"tool_result_store_001: Loaded \033[33m{len(results)}\033[0m "
                f"results from \033[36m{key}\033[0m"
            )
            return results
        except (redis.RedisError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"tool_result_store_error_001: Load failed: \033[31m{e}\033[0m")
            return []

    async def save(
        self,
        conversation_id: str | None,
        user_name: str | None,
        results: list[ToolResult],
    ) -> None:
        """Persist results for the session with TTL. No-op when disabled/empty."""
        key = self._build_key(conversation_id, user_name)
        if not self.enabled or not key or not results:
            return
        try:
            deduped = self._dedupe(results)
            payload = json.dumps(
                [result.model_dump() for result in deduped], ensure_ascii=False
            )
            await self.redis_client.set(key, payload, ex=self.ttl)
            logger.info(
                f"tool_result_store_002: Saved \033[33m{len(deduped)}\033[0m "
                f"results to \033[36m{key}\033[0m (ttl=\033[33m{self.ttl}\033[0ms)"
            )
        except (redis.RedisError, TypeError) as e:
            logger.error(f"tool_result_store_error_002: Save failed: \033[31m{e}\033[0m")
