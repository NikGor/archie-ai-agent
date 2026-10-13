"""Swarm dispatcher: wakes agents for new chat messages, with loop guards.

Guards: hop depth limit, no self-wake, one running turn per agent, and a
cooldown between agent-to-agent wake-ups.
"""

import asyncio
import contextlib
import logging
import time
from archie_shared.swarm.models import SwarmMessage, SwarmMessageEvent, SwarmTypingEvent
from ..config import settings
from .agents import SWARM_AGENTS, SwarmRoster
from .bus import SwarmBus, SwarmHub
from .gate import SwarmGate
from .runner import SwarmAgentRunner


logger = logging.getLogger(__name__)

_AGENT_SENDERS = set(SWARM_AGENTS)


class SwarmDispatcher:
    def __init__(
        self,
        hub: SwarmHub,
        bus: SwarmBus,
        roster: SwarmRoster,
        runner: SwarmAgentRunner,
        gate: SwarmGate,
    ) -> None:
        self.hub = hub
        self.bus = bus
        self.roster = roster
        self.runner = runner
        self.gate = gate
        self._running: set[str] = set()
        self._last_run: dict[str, float] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        queue = self.hub.subscribe()
        self._worker = asyncio.create_task(self._listen(queue), name="archie-swarm")
        logger.info("swarm_dispatcher_001: Dispatcher started")

    async def stop(self) -> None:
        for task in [self._worker, *self._tasks]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        logger.info("swarm_dispatcher_002: Dispatcher stopped")

    async def _listen(self, queue: asyncio.Queue) -> None:
        while True:
            event = await queue.get()
            if not isinstance(event, SwarmMessageEvent):
                continue
            try:
                await self.handle(event.message)
            except Exception:
                logger.exception("swarm_dispatcher_error_001: Failed to handle message")

    async def handle(self, message: SwarmMessage) -> None:
        for agent_id in await self.select_targets(message):
            self._running.add(agent_id)
            task = asyncio.create_task(self._run_agent(agent_id, message))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def select_targets(self, message: SwarmMessage) -> list[str]:
        if message.depth >= settings.swarm_max_depth:
            logger.info(
                f"swarm_dispatcher_003: Depth limit reached at {message.message_id}"
            )
            return []
        enabled = await self.roster.get_enabled_ids()
        if message.mentions:
            candidates = message.mentions
        elif message.sender == "user":
            candidates = await self.gate.route(
                message,
                [
                    SWARM_AGENTS[agent_id]
                    for agent_id in SWARM_AGENTS
                    if agent_id in enabled
                ],
            )
        else:
            return []
        from_agent = message.sender in _AGENT_SENDERS
        return [
            agent_id
            for agent_id in candidates
            if agent_id in enabled
            and agent_id != message.sender
            and agent_id not in self._running
            and not (from_agent and self._in_cooldown(agent_id))
        ]

    def _in_cooldown(self, agent_id: str) -> bool:
        last = self._last_run.get(agent_id)
        cooling = (
            last is not None
            and time.monotonic() - last < settings.swarm_cooldown_seconds
        )
        if cooling:
            logger.info(
                f"swarm_dispatcher_004: \033[36m{agent_id}\033[0m is in cooldown"
            )
        return cooling

    async def _run_agent(self, agent_id: str, trigger: SwarmMessage) -> None:
        self.hub.publish(SwarmTypingEvent(agent_id=agent_id, active=True))
        try:
            history = await self.bus.get_history(settings.swarm_history_size)
            draft = await self.runner.run(SWARM_AGENTS[agent_id], trigger, history)
            if draft:
                await self.bus.post(
                    sender=agent_id,
                    kind=(
                        "action"
                        if any(a.status == "success" for a in draft.actions)
                        else "report"
                    ),
                    text=draft.text,
                    reply_to=trigger.message_id,
                    depth=trigger.depth + 1,
                    actions=draft.actions,
                    usage=draft.usage,
                )
        except Exception as exc:
            logger.exception(f"swarm_dispatcher_error_002: Agent {agent_id} failed")
            # No "@" here: a mention would wake the failing agent again.
            await self.bus.post(
                sender="system",
                kind="system",
                text=f"{agent_id} failed: {exc}",
                reply_to=trigger.message_id,
                depth=settings.swarm_max_depth,
            )
        finally:
            self._last_run[agent_id] = time.monotonic()
            self._running.discard(agent_id)
            self.hub.publish(SwarmTypingEvent(agent_id=agent_id, active=False))
