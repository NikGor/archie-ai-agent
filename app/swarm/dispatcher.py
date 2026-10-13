"""Swarm dispatcher: wakes agents for new chat messages, with loop guards.

Guards: hop depth limit, no self-wake, one running turn per agent, and a
cooldown between agent-to-agent wake-ups. '@group' mentions wake the members
one after another; listens_to subscribers hear their senders without mentions.
"""

import asyncio
import contextlib
import logging
import time
from archie_shared.swarm.models import SwarmMessage, SwarmMessageEvent, SwarmTypingEvent
from ..config import settings
from .agents import SWARM_AGENTS, SWARM_GROUPS, SwarmRoster, get_group_members
from .bus import SwarmBus, SwarmHub
from .gate import SwarmGate
from .models import SwarmRouteOption
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
        self._pending: dict[str, SwarmMessage] = {}
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
        enabled = await self.roster.get_enabled_ids()
        queues = await self.select_targets(message, enabled)
        for queue in queues:
            self._start(queue, message)
        queued = {agent_id for queue in queues for agent_id in queue}
        for agent_id in self.select_listeners(message, enabled):
            if agent_id in queued:
                continue
            if agent_id in self._running:
                # Keep the latest event for a busy listener instead of dropping it.
                self._pending[agent_id] = message
            else:
                self._start([agent_id], message)

    async def select_targets(
        self, message: SwarmMessage, enabled: set[str]
    ) -> list[list[str]]:
        """Return wake queues: agents in one queue speak one after another
        (a group), separate queues run in parallel."""
        if message.depth >= settings.swarm_max_depth:
            logger.info(
                f"swarm_dispatcher_003: Depth limit reached at {message.message_id}"
            )
            return []
        if message.mentions:
            candidates = message.mentions
        elif message.sender == "user":
            candidates = await self.gate.route(message, self._route_options(enabled))
        else:
            return []
        from_agent = message.sender in _AGENT_SENDERS
        queues: list[list[str]] = []
        seen: set[str] = set()
        for candidate in candidates:
            members = (
                get_group_members(candidate)
                if candidate in SWARM_GROUPS
                else [candidate]
            )
            queue = [
                agent_id
                for agent_id in members
                if agent_id not in seen
                and agent_id in enabled
                and agent_id != message.sender
                and agent_id not in self._running
                and not (from_agent and self._in_cooldown(agent_id))
            ]
            seen.update(queue)
            if queue:
                queues.append(queue)
        return queues

    @staticmethod
    def select_listeners(message: SwarmMessage, enabled: set[str]) -> list[str]:
        """Agents subscribed to the sender via listens_to; no depth limit or cooldown."""
        return [
            profile.agent_id
            for profile in SWARM_AGENTS.values()
            if message.sender in profile.listens_to
            and profile.agent_id in enabled
            and profile.agent_id != message.sender
        ]

    @staticmethod
    def _route_options(enabled: set[str]) -> list[SwarmRouteOption]:
        agents = [
            SwarmRouteOption(agent_id=profile.agent_id, description=profile.description)
            for profile in SWARM_AGENTS.values()
            if profile.agent_id in enabled and not profile.groups
        ]
        groups = [
            SwarmRouteOption(agent_id=group, description=description)
            for group, description in SWARM_GROUPS.items()
            if any(agent_id in enabled for agent_id in get_group_members(group))
        ]
        return agents + groups

    def _start(self, queue: list[str], trigger: SwarmMessage) -> None:
        self._running.update(queue)
        task = asyncio.create_task(self._run_queue(queue, trigger))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_queue(self, queue: list[str], trigger: SwarmMessage) -> None:
        for agent_id in queue:
            await self._run_agent(agent_id, trigger)
            pending = self._pending.pop(agent_id, None)
            if pending:
                self._start([agent_id], pending)

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
