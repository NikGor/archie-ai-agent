"""Swarm message bus.

Redis Stream `archie:swarm:stream` is the durable chat history (survives
restarts and WS reconnects). SwarmHub fans live events out to in-process
listeners: the dispatcher and every open /ws_swarm connection.
"""

import asyncio
import json
import logging
import re
import uuid
from datetime import UTC, datetime
from archie_shared.swarm.models import (
    SWARM_TEXT_MAX_LENGTH,
    SwarmActionInfo,
    SwarmImportance,
    SwarmMessage,
    SwarmMessageEvent,
    SwarmMessageKind,
    SwarmServerEvent,
    SwarmUsage,
)
from app.agents.swarm.agents import SWARM_AGENTS, SWARM_GROUPS, add_paired_outputs
from app.backend.redis_factory import get_async_redis
from app.config import settings


logger = logging.getLogger(__name__)

_STREAM_KEY = "archie:swarm:stream"
# Extra entries read past check lines: about a day of checks
_CHECK_LINES_SCAN = 120
_MENTION_PATTERN = re.compile(r"@([a-z][a-z0-9_]*)")


def parse_mentions(text: str) -> list[str]:
    """Return known agent and group IDs mentioned in `text`, in order, without duplicates."""
    found = [
        m
        for m in _MENTION_PATTERN.findall(text.lower())
        if m in SWARM_AGENTS or m in SWARM_GROUPS
    ]
    return list(dict.fromkeys(found))


def new_run_id() -> str:
    """ID of one swarm run: a trigger and every message it causes (ARCHIE-281)."""
    return uuid.uuid4().hex[:16]


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
        importance: SwarmImportance | None = None,
        paired: bool = True,
        round_number: int | None = None,
        run_id: str | None = None,
    ) -> SwarmMessage:
        """`mentions` overrides the mentions parsed from the text (council
        delegations). `paired` adds the paired outputs (voice -> display).
        A message without `run_id` starts a new run; replies pass the run of
        their trigger."""
        if mentions is None:
            mentions = parse_mentions(text)
        text = clip_text(text)
        message = SwarmMessage(
            message_id="",
            run_id=run_id or new_run_id(),
            created_at=datetime.now(UTC).isoformat(),
            sender=sender,
            kind=kind,
            text=text,
            mentions=[
                m
                for m in (add_paired_outputs(mentions) if paired else mentions)
                if m != sender
            ],
            reply_to=reply_to,
            depth=depth,
            actions=actions or [],
            importance=importance,
            round=round_number,
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

    async def clear(self) -> None:
        """Delete the whole chat history (Stop, ARCHIE-238)."""
        await self.redis.delete(_STREAM_KEY)
        logger.info("swarm_bus_002: Chat history cleared")

    async def get_history(
        self, count: int, with_checks: bool = False
    ) -> list[SwarmMessage]:
        """Return the last `count` messages, oldest first. Silent check lines
        are for the user only: agents get the chat without them (ARCHIE-242)."""
        scan = count if with_checks else count + _CHECK_LINES_SCAN
        entries = await self.redis.xrevrange(_STREAM_KEY, count=scan)
        messages = [
            SwarmMessage.model_validate(
                {**json.loads(fields["data"]), "message_id": entry_id}
            )
            for entry_id, fields in entries
        ]
        if not with_checks:
            messages = [m for m in messages if m.kind != "check"]
        return list(reversed(messages[:count]))
