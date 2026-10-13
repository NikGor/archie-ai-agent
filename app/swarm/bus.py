"""Swarm message bus.

Redis Stream `archie:swarm:stream` is the durable chat history (survives
restarts and WS reconnects). SwarmHub fans live events out to in-process
listeners: the dispatcher and every open /ws_swarm connection.
"""

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from archie_shared.swarm.models import (
    SWARM_TEXT_MAX_LENGTH,
    SwarmActionInfo,
    SwarmMessage,
    SwarmMessageEvent,
    SwarmMessageKind,
    SwarmServerEvent,
    SwarmUsage,
)
from ..backend.redis_factory import get_async_redis
from ..config import settings
from .agents import SWARM_AGENTS, SWARM_GROUPS


logger = logging.getLogger(__name__)

_STREAM_KEY = "archie:swarm:stream"
_MENTION_PATTERN = re.compile(r"@([a-z][a-z0-9_]*)")


def parse_mentions(text: str) -> list[str]:
    """Return known agent and group IDs mentioned in `text`, in order, without duplicates."""
    found = [
        m
        for m in _MENTION_PATTERN.findall(text.lower())
        if m in SWARM_AGENTS or m in SWARM_GROUPS
    ]
    return list(dict.fromkeys(found))


def clip_text(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= SWARM_TEXT_MAX_LENGTH:
        return text
    return text[: SWARM_TEXT_MAX_LENGTH - 1] + "…"


class SwarmHub:
    """In-process pub/sub of swarm server events."""

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[SwarmServerEvent]] = set()

    def subscribe(self) -> asyncio.Queue[SwarmServerEvent]:
        queue: asyncio.Queue[SwarmServerEvent] = asyncio.Queue()
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[SwarmServerEvent]) -> None:
        self._queues.discard(queue)

    def publish(self, event: SwarmServerEvent) -> None:
        for queue in self._queues:
            queue.put_nowait(event)


class SwarmBus:
    """Writes messages to the Redis stream and announces them on the hub."""

    def __init__(self, hub: SwarmHub) -> None:
        self.hub = hub
        self.redis = get_async_redis()

    async def post(
        self,
        sender: str,
        kind: SwarmMessageKind,
        text: str,
        reply_to: str | None = None,
        depth: int = 0,
        actions: list[SwarmActionInfo] | None = None,
        usage: SwarmUsage | None = None,
        mentions: list[str] | None = None,
    ) -> SwarmMessage:
        """`mentions` overrides the mentions parsed from the text (council delegations)."""
        text = clip_text(text)
        message = SwarmMessage(
            message_id="",
            created_at=datetime.now(UTC).isoformat(),
            sender=sender,
            kind=kind,
            text=text,
            mentions=[
                m
                for m in (mentions if mentions is not None else parse_mentions(text))
                if m != sender
            ],
            reply_to=reply_to,
            depth=depth,
            actions=actions or [],
            usage=usage,
        )
        data = message.model_dump_json(exclude={"message_id"})
        message_id = await self.redis.xadd(
            _STREAM_KEY,
            {"data": data},
            maxlen=settings.swarm_stream_max_length,
            approximate=True,
        )
        message.message_id = message_id
        logger.info(
            f"swarm_bus_001: \033[36m{sender}\033[0m [{kind}] depth={depth} "
            f"mentions={message.mentions}: {text[:80]}"
        )
        self.hub.publish(SwarmMessageEvent(message=message))
        return message

    async def get_history(self, count: int) -> list[SwarmMessage]:
        """Return the last `count` messages, oldest first."""
        entries = await self.redis.xrevrange(_STREAM_KEY, count=count)
        return [
            SwarmMessage.model_validate(
                {**json.loads(fields["data"]), "message_id": entry_id}
            )
            for entry_id, fields in reversed(entries)
        ]
