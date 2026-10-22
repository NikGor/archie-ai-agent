"""Swarm dispatcher: wakes agents for new chat messages, with loop guards.

Routing: a user message without @mentions, or any '@council' mention, goes to
the council (see council.py). Other @mentions wake the named agents directly.
listens_to subscribers hear their senders without mentions. Council opinions
wake nobody.

Guards: hop depth limit, no self-wake, one running turn per agent, and a
cooldown between agent-to-agent wake-ups.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Coroutine
from typing import Any
from archie_shared.swarm.models import SwarmMessage, SwarmMessageEvent, SwarmTypingEvent
from ..config import settings
from .agents import SWARM_AGENTS, SWARM_GROUPS, SwarmRoster, get_group_members
from .bus import SwarmBus, SwarmHub
from .council import COUNCIL_GROUP, SwarmCouncil
from .memory import SwarmMemory
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
        council: SwarmCouncil,
        memory: SwarmMemory,
    ) -> None:
        self.hub = hub
        self.memory = memory
        self.bus = bus
        self.roster = roster
        self.runner = runner
        self.council = council
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

    async def reset(self) -> None:
        """Cancel all running turns and drop queued ones (Stop, ARCHIE-238)."""
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._pending.clear()
        self._running.clear()
        self._last_run.clear()
        for agent_id in SWARM_AGENTS:
            self.hub.publish(SwarmTypingEvent(agent_id=agent_id, active=False))
        logger.info("swarm_dispatcher_007: All turns cancelled")

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
        if await self.memory.get_mode() == "stopped":
            logger.info("swarm_dispatcher_006: Swarm is stopped, message ignored")
            return
        enabled = await self.roster.get_enabled_ids()
        if self.wants_council(message):
            await self._start_council(message, enabled)
        relays = [a for a in self.council.pop_relay(message) if a in enabled]
        for agent_id in relays:
            # The data for a display or voice task arrived (ARCHIE-240).
            relayed = message.model_copy(update={"mentions": [agent_id]})
            if agent_id in self._running:
                self._pending[agent_id] = relayed
            else:
                self._start(agent_id, relayed)
        targets = self.select_targets(message, enabled)
        if self.is_human_intent(message):
            # A user command or council decision must not be lost to a busy agent.
            for agent_id in self.select_busy(message, enabled):
                self._pending[agent_id] = message
        for agent_id in targets:
            self._start(agent_id, message)
        # A relayed agent already got this message: no "busy" note for it.
        await self._report_skipped(message, enabled, targets + relays)
        for agent_id in self.select_listeners(message, enabled):
            if agent_id in targets:
                continue
            if agent_id in self._running:
                # Keep the latest event for a busy listener instead of dropping it.
                self._pending[agent_id] = message
            else:
                self._start(agent_id, message)

    @staticmethod
    def wants_council(message: SwarmMessage) -> bool:
        if message.kind == "opinion" or message.depth >= settings.swarm_max_depth:
            return False
        if COUNCIL_GROUP in message.mentions:
            return True
        return message.sender == "user" and not message.mentions

    @staticmethod
    def is_human_intent(message: SwarmMessage) -> bool:
        """User messages and council decisions carry the user's will: no cooldown."""
        return message.sender == "user" or message.kind == "decision"

    def _mentioned(self, message: SwarmMessage, enabled: set[str]) -> list[str]:
        if message.kind == "opinion":
            return []
        if message.depth >= settings.swarm_max_depth:
            logger.info(
                f"swarm_dispatcher_003: Depth limit reached at {message.message_id}"
            )
            return []
        return [
            agent_id
            for agent_id in message.mentions
            if agent_id not in SWARM_GROUPS
            and agent_id in enabled
            and agent_id != message.sender
        ]

    def select_targets(self, message: SwarmMessage, enabled: set[str]) -> list[str]:
        """Agents named directly by @mention, after the loop guards. Cooldown
        applies only to agent-to-agent mentions (ARCHIE-204)."""
        cooled = message.sender in _AGENT_SENDERS and not self.is_human_intent(message)
        return [
            agent_id
            for agent_id in self._mentioned(message, enabled)
            if agent_id not in self._running
            and not (cooled and self._in_cooldown(agent_id))
        ]

    def select_busy(self, message: SwarmMessage, enabled: set[str]) -> list[str]:
        """Mentioned agents that are running now."""
        return [a for a in self._mentioned(message, enabled) if a in self._running]

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

    async def _start_council(self, message: SwarmMessage, enabled: set[str]) -> None:
        members = [m for m in get_group_members(COUNCIL_GROUP) if m in enabled]
        if not members:
            # No "@" here: a mention would route back to the council.
            await self.bus.post(
                sender="system",
                kind="system",
                text="The council is asleep. Wake a council member or mention an agent.",
                reply_to=message.message_id,
                depth=settings.swarm_max_depth,
            )
            return
        if self._running & set(members):
            # The council is busy: answer the latest message when it is free.
            self._pending[COUNCIL_GROUP] = message
            return
        self._running.update(members)
        # Only the user can force a discussion; agents get a chair triage.
        force = message.sender == "user" and COUNCIL_GROUP in message.mentions
        self._track(self._run_council(message, members, force))

    async def _run_council(
        self, trigger: SwarmMessage, members: list[str], force: bool
    ) -> None:
        try:
            await self.council.run(trigger, members, force)
        except Exception as exc:
            logger.exception("swarm_dispatcher_error_003: Council failed")
            await self._post_failure(COUNCIL_GROUP, trigger, exc)
        finally:
            self._running.difference_update(members)
        pending = self._pending.pop(COUNCIL_GROUP, None)
        if pending:
            await self._start_council(pending, await self.roster.get_enabled_ids())

    def _start(self, agent_id: str, trigger: SwarmMessage) -> None:
        self._running.add(agent_id)
        self._track(self._run_single(agent_id, trigger))

    def _track(self, coroutine: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_single(self, agent_id: str, trigger: SwarmMessage) -> None:
        await self._run_agent(agent_id, trigger)
        pending = self._pending.pop(agent_id, None)
        if pending:
            self._start(agent_id, pending)

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
            if not draft and self.is_human_intent(trigger):
                # A task from the user or the council must not end in silence.
                await self._post_note(f"{agent_id} gave no answer", trigger)
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
            await self._post_failure(agent_id, trigger, exc)
        finally:
            self._last_run[agent_id] = time.monotonic()
            self._running.discard(agent_id)
            self.hub.publish(SwarmTypingEvent(agent_id=agent_id, active=False))

    async def _report_skipped(
        self, message: SwarmMessage, enabled: set[str], targets: list[str]
    ) -> None:
        """Say in the chat why a mentioned agent does not answer, instead of a
        silent drop (ARCHIE-206)."""
        if message.kind == "opinion":
            return
        for agent_id in message.mentions:
            if (
                agent_id in SWARM_GROUPS
                or agent_id not in enabled
                or agent_id == message.sender
                or agent_id in targets
                or self._pending.get(agent_id) is message
            ):
                continue
            if message.depth >= settings.swarm_max_depth:
                reason = f"chain limit of {settings.swarm_max_depth} steps reached"
            elif agent_id in self._running:
                reason = "busy with another task"
            else:
                reason = "asked again too soon by an agent"
            logger.info(
                f"swarm_dispatcher_005: \033[36m{agent_id}\033[0m skipped: {reason}"
            )
            await self._post_note(f"{agent_id} skipped: {reason}", message)

    async def _post_note(self, text: str, trigger: SwarmMessage) -> None:
        # No "@" in notes: a mention would wake the agent again.
        await self.bus.post(
            sender="system",
            kind="system",
            text=text,
            reply_to=trigger.message_id,
            depth=settings.swarm_max_depth,
        )

    async def _post_failure(
        self, name: str, trigger: SwarmMessage, exc: Exception
    ) -> None:
        # No "@" here: a mention would wake the failing agent again.
        await self.bus.post(
            sender="system",
            kind="system",
            text=f"{name} failed: {exc}",
            reply_to=trigger.message_id,
            depth=settings.swarm_max_depth,
        )
