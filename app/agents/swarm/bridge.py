"""Swarm events and commands across processes (ARCHIE-299).

With RUNTIME_MODE=api / runtime the swarm runs in its own process. Every event
one process publishes on its SwarmHub goes to the other through a Redis
pub/sub channel; the API process sends the commands that need the ticker to
the runtime process. The runtime keeps its last control event in Redis for
newly connected clients.
"""

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Literal
from archie_shared.swarm.models import (
    SwarmClientCommand,
    SwarmControlEvent,
    SwarmServerEvent,
    SwarmWatchTask,
)
from pydantic import BaseModel, Field, TypeAdapter
from app.agents.swarm.bus import SwarmHub
from app.backend.redis_factory import get_async_redis


logger = logging.getLogger(__name__)

EVENTS_CHANNEL = "archie:swarm:events"
COMMANDS_CHANNEL = "archie:swarm:commands"
CONTROL_KEY = "archie:swarm:control"

_events: TypeAdapter[SwarmServerEvent] = TypeAdapter(SwarmServerEvent)
_client_commands: TypeAdapter[SwarmClientCommand] = TypeAdapter(SwarmClientCommand)


class RunCheckCommand(BaseModel):
    """Run one spontaneous check now (POST /swarm/checks/{task})."""

    type: Literal["run_check"] = Field("run_check", description="Command type")
    task: SwarmWatchTask = Field(description="Check to run")


RuntimeCommand = SwarmClientCommand | RunCheckCommand
CommandHandler = Callable[[RuntimeCommand], Awaitable[None]]


class SwarmHubBridge:
    name = "swarm-bridge"

    def __init__(
        self,
        hub: SwarmHub,
        role: Literal["api", "runtime"],
        on_command: CommandHandler | None = None,
    ) -> None:
        self.hub = hub
        self.role = role
        self.on_command = on_command
        self.origin = uuid.uuid4().hex
        self.redis = get_async_redis()
        self._listener: asyncio.Task[None] | None = None
        self._sends: set[asyncio.Task[None]] = set()
        hub.relays.append(self._relay)

    async def start(self) -> None:
        pubsub = self.redis.pubsub()
        channels = [EVENTS_CHANNEL] + ([COMMANDS_CHANNEL] if self.role == "runtime" else [])
        await pubsub.subscribe(*channels)
        self._listener = asyncio.create_task(self._listen(pubsub), name="swarm-bridge")
        logger.info(f"swarm_bridge_001: \033[36m{self.role}\033[0m bridge on {channels}")

    async def stop(self) -> None:
        if self._listener:
            self._listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener
            self._listener = None

    def healthy(self) -> bool:
        return self._listener is not None and not self._listener.done()

    def _relay(self, event: SwarmServerEvent) -> None:
        task = asyncio.create_task(self._send_event(event))
        self._sends.add(task)
        task.add_done_callback(self._sends.discard)

    async def _send_event(self, event: SwarmServerEvent) -> None:
        payload = json.dumps(
            {"origin": self.origin, "event": event.model_dump(mode="json")},
            ensure_ascii=False,
        )
        try:
            await self.redis.publish(EVENTS_CHANNEL, payload)
            if self.role == "runtime" and isinstance(event, SwarmControlEvent):
                await self.redis.set(CONTROL_KEY, event.model_dump_json())
        except Exception as exc:
            logger.error(f"swarm_bridge_error_001: Event not relayed: {exc}")

    async def send_command(self, command: RuntimeCommand) -> None:
        await self.redis.publish(COMMANDS_CHANNEL, command.model_dump_json())

    async def last_control(self) -> SwarmControlEvent | None:
        raw = await self.redis.get(CONTROL_KEY)
        return SwarmControlEvent.model_validate_json(raw) if raw else None

    async def _listen(self, pubsub) -> None:  # noqa: ANN001  redis PubSub
        async for item in pubsub.listen():
            if item.get("type") != "message":
                continue
            try:
                await self._handle(item["channel"], item["data"])
            except Exception as exc:
                logger.error(f"swarm_bridge_error_002: Bad bridge message: {exc}")

    async def _handle(self, channel: str, data: str) -> None:
        if channel == EVENTS_CHANNEL:
            payload = json.loads(data)
            if payload["origin"] != self.origin:
                self.hub.deliver(_events.validate_python(payload["event"]))
            return
        if channel == COMMANDS_CHANNEL and self.on_command:
            raw = json.loads(data)
            command: RuntimeCommand = (
                RunCheckCommand.model_validate(raw)
                if raw.get("type") == "run_check"
                else _client_commands.validate_python(raw)
            )
            await self.on_command(command)
