"""Spontaneous swarm checks (ARCHIE-238).

While the swarm runs, the ticker wakes up after a random pause in
[interval_min, interval_max] minutes, inside the active hours. Each wake-up
runs one check (home, mail, calendar or football) with its watcher agent.
A check that finds nothing new stays silent. An alert goes to the chat; its
importance decides who else hears it: normal -> @display, high -> @display
and @voice (only in the voice hours).

Play/Pause/Stop: Pause stops the checks, the chat still works. Stop also
clears the chat and the dispatcher ignores all messages.
"""

import asyncio
import contextlib
import logging
import time
from datetime import UTC, date, datetime, timedelta
from archie_shared.swarm.models import (
    SwarmControlEvent,
    SwarmHistoryEvent,
    SwarmImportance,
    SwarmMessage,
    SwarmMode,
    SwarmRunInfo,
    SwarmSettings,
    SwarmTypingEvent,
    SwarmUsage,
    SwarmWatchTask,
)
from ..config import settings as app_settings
from ..utils.general_utils import local_now
from ..utils.tracing import trace_swarm_run
from .agents import SWARM_AGENTS, SwarmRoster
from .bus import SwarmBus, SwarmHub, new_run_id
from .dispatcher import SwarmDispatcher
from .memory import DayNote, SwarmMemory
from .runner import SwarmAgentRunner
from .schedule import (
    due_tasks,
    is_active,
    parse_quiet_until,
    plan_delay_seconds,
    pick_task,
    voice_allowed,
)


logger = logging.getLogger(__name__)

# Check -> watcher agent
WATCH_AGENTS: dict[SwarmWatchTask, str] = {
    "home": "home",
    "mail": "mail",
    "calendar": "calendar",
    "football": "football",
    "weather": "weather",
}
# Checks whose next event has a known time; home and mail can change any time.
QUIET_TASKS: frozenset[str] = frozenset({"calendar", "football"})
# Chat lines a watcher sees: it works from its notes, the chat is background.
WATCH_HISTORY_SIZE = 6


def select_outputs(
    importance: SwarmImportance, voice_ok: bool, enabled: set[str]
) -> list[str]:
    """Agents that must also deliver an alert of this importance."""
    outputs = []
    if importance in ("normal", "high"):
        outputs.append("display")
    if importance == "high" and voice_ok:
        outputs.append("voice")
    return [agent_id for agent_id in outputs if agent_id in enabled]


class SwarmTicker:
    def __init__(
        self,
        hub: SwarmHub,
        bus: SwarmBus,
        roster: SwarmRoster,
        runner: SwarmAgentRunner,
        memory: SwarmMemory,
        dispatcher: SwarmDispatcher,
    ) -> None:
        self.hub = hub
        self.bus = bus
        self.roster = roster
        self.runner = runner
        self.memory = memory
        self.dispatcher = dispatcher
        self.next_run_at: datetime | None = None
        self._changed = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._check: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._worker = asyncio.create_task(self._loop(), name="archie-swarm-ticker")
        logger.info("swarm_ticker_001: Ticker started")

    async def stop(self) -> None:
        for task in [self._check, self._worker]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def set_mode(self, mode: SwarmMode) -> None:
        await self.memory.set_mode(mode)
        if mode == "stopped":
            await self._cancel_check()
            await self.dispatcher.reset()
            await self.bus.clear()
            self.hub.publish(
                SwarmHistoryEvent(messages=[], agents=await self.roster.list_agents())
            )
        self._changed.set()
        await self.publish_control()

    async def set_settings(self, swarm_settings: SwarmSettings) -> None:
        await self.memory.set_settings(swarm_settings)
        self._changed.set()
        await self.publish_control()

    async def forget(self, fact_id: str) -> None:
        await self.memory.forget_fact(fact_id)
        await self.publish_control()

    async def build_control_event(self) -> SwarmControlEvent:
        return SwarmControlEvent(
            mode=await self.memory.get_mode(),
            settings=await self.memory.get_settings(),
            next_run_at=self.next_run_at.isoformat() if self.next_run_at else None,
            last_run=await self.memory.get_last_run(),
            memory=await self.memory.list_facts(),
            today=await self.memory.get_today_facts(await self.local_date()),
        )

    async def local_date(self) -> date:
        state = await self.runner.get_user_state()
        return local_now(state.user_timezone).date()

    async def publish_control(self) -> None:
        self.hub.publish(await self.build_control_event())

    async def _loop(self) -> None:
        while True:
            self._changed.clear()
            delay = None
            if await self.memory.get_mode() == "running":
                state = await self.runner.get_user_state()
                delay = plan_delay_seconds(
                    local_now(state.user_timezone), await self.memory.get_settings()
                )
                self.next_run_at = datetime.now(UTC) + timedelta(seconds=delay)
            else:
                self.next_run_at = None
            await self.publish_control()
            try:
                # A mode or settings change plans the next check again.
                await asyncio.wait_for(self._changed.wait(), timeout=delay)
                continue
            except TimeoutError:
                pass
            self._check = asyncio.create_task(self.run_check())
            with contextlib.suppress(asyncio.CancelledError):
                await self._check
            self._check = None

    async def _cancel_check(self) -> None:
        if self._check and not self._check.done():
            self._check.cancel()
            logger.info("swarm_ticker_002: Running check cancelled")

    async def run_check(self, task: SwarmWatchTask | None = None) -> None:
        """Run one check: the given one, or the one that waited longest."""
        run_id = new_run_id()
        try:
            swarm_settings = await self.memory.get_settings()
            state = await self.runner.get_user_state()
            now = local_now(state.user_timezone)
            if task is None:
                if not is_active(now, swarm_settings):
                    logger.info("swarm_ticker_003: Outside the active hours")
                    return
                enabled = await self.roster.get_enabled_ids()
                task = pick_task(
                    [
                        t
                        for t in due_tasks(
                            swarm_settings, now, await self.memory.get_quiet_until()
                        )
                        if WATCH_AGENTS[t] in enabled
                    ],
                    await self.memory.get_last_task_runs(),
                )
                if task is None:
                    logger.info("swarm_ticker_004: No check is due")
                    return
            await self._watch(
                task,
                swarm_settings,
                now,
                state.language or app_settings.default_language,
                run_id,
            )
        except Exception as exc:
            logger.exception("swarm_ticker_error_001: Check failed")
            if task:
                # No "@" here: a mention would wake an agent.
                await self.bus.post(
                    sender="system",
                    kind="system",
                    text=f"{task} check failed: {exc}",
                    depth=app_settings.swarm_max_depth,
                    run_id=run_id,
                )
                await self._finish(task, "failed")

    async def _watch(
        self,
        task: SwarmWatchTask,
        swarm_settings: SwarmSettings,
        now: datetime,
        language: str,
        run_id: str | None = None,
    ) -> None:
        run_id = run_id or new_run_id()
        agent_id = WATCH_AGENTS[task]
        logger.info(f"swarm_ticker_005: Check \033[36m{task}\033[0m run={run_id}")
        await self.memory.mark_task_run(task, time.time())
        self.hub.publish(SwarmTypingEvent(agent_id=agent_id, active=True))
        try:
            with trace_swarm_run(
                "swarm-check",
                run_id,
                agent_id,
                now.date(),
                f"{task} check",
                task=task,
            ) as root:
                draft = await self.runner.watch(
                    SWARM_AGENTS[agent_id],
                    task,
                    await self.bus.get_history(WATCH_HISTORY_SIZE),
                    self.runner.render_task(
                        "swarm_watch",
                        task=task,
                        teams=swarm_settings.football_teams,
                        language=language,
                    ),
                )
                root.update(output=draft.outcome.model_dump(mode="json"))
        finally:
            self.hub.publish(SwarmTypingEvent(agent_id=agent_id, active=False))
        outcome = draft.outcome
        if task in QUIET_TASKS:
            await self.memory.set_quiet_until(
                task, parse_quiet_until(outcome.quiet_until, now)
            )
        alert = outcome.reply.replace("@", "").strip()
        # The check itself decides what the swarm keeps for today (ARCHIE-246)
        await self.memory.add_today_facts(now.date(), outcome.remember_today)
        if outcome.today_note.strip() or alert:
            await self.memory.add_day_note(
                now.date(),
                DayNote(
                    time=now.strftime("%H:%M"),
                    task=task,
                    note=outcome.today_note.strip(),
                    reported=alert or None,
                ),
            )
        logger.info(
            f"swarm_ticker_006: \033[36m{task}\033[0m importance={outcome.importance} "
            f"alert=\033[33m{alert or '-'}\033[0m note: {outcome.today_note}"
        )
        if not alert:
            # Every check shows in the chat; agents do not read "check" lines (ARCHIE-242).
            await self.bus.post(
                sender=agent_id,
                kind="check",
                text=outcome.today_note.strip() or "ничего нового",
                usage=draft.usage,
                mentions=[],
                run_id=run_id,
            )
            await self._finish(task, "silent")
            return
        await self.post_alert(
            agent_id, alert, outcome.importance, draft.usage, now, run_id
        )
        await self._finish(task, "reported")

    async def post_alert(
        self,
        sender: str,
        text: str,
        importance: SwarmImportance,
        usage: SwarmUsage | None = None,
        now: datetime | None = None,
        run_id: str | None = None,
    ) -> SwarmMessage:
        """Post an alert; its importance decides who else delivers it.
        Without `run_id` (a cron alert) the alert starts a new run."""
        swarm_settings = await self.memory.get_settings()
        if now is None:
            now = local_now((await self.runner.get_user_state()).user_timezone)
        outputs = select_outputs(
            importance,
            voice_allowed(now, swarm_settings),
            await self.roster.get_enabled_ids(),
        )
        return await self.bus.post(
            sender=sender,
            kind="alert",
            text=text,
            usage=usage,
            mentions=outputs,
            importance=importance,
            run_id=run_id,
        )

    async def _finish(self, task: SwarmWatchTask, outcome: str) -> None:
        await self.memory.set_last_run(
            SwarmRunInfo(
                task=task, finished_at=datetime.now(UTC).isoformat(), outcome=outcome
            )
        )
        await self.publish_control()
