"""Council: the main contact of the swarm, a discussion led by a consul (ARCHIE-249).

A user message without @mentions, an @council mention or a check alert starts
a discussion:
1. Every member except the consul gets the message at the same time and on its
   own model; each posts an opinion as soon as it is ready. In a round the
   members do not see each other's opinions of that round.
2. The consul reads the round. He can end the discussion, or post a remark and
   start one more round (up to swarm_council_max_rounds).
3. The consul posts the single decision with a plan. The plan steps give tasks
   to home agents one after another; a step can wait for the agent's reply,
   so that the next step gets its data.
Opinions and the decision never wake anyone; only plan steps do.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from archie_shared.swarm.models import (
    SwarmMessage,
    SwarmMessageEvent,
    SwarmMessageKind,
    SwarmTypingEvent,
    SwarmUsage,
)
from pydantic import BaseModel, Field
from ..config import settings
from ..utils.general_utils import local_now
from .agents import SWARM_AGENTS, add_paired_outputs
from .bus import SwarmBus, SwarmHub
from .models import ConsulDecision, ConsulTurn, CouncilStep, SwarmReply
from .runner import SwarmAgentRunner, format_chat_line


logger = logging.getLogger(__name__)

COUNCIL_GROUP = "council"
# Agents that deliver data to the user
OUTPUT_AGENTS = frozenset({"display", "voice", "chat"})


def merge_usage(first: SwarmUsage | None, second: SwarmUsage) -> SwarmUsage:
    if first is None:
        return second
    return SwarmUsage(
        model=second.model,
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        total_cost=first.total_cost + second.total_cost,
    )


class CouncilPlan(BaseModel):
    """The consul's decision and the steps that the home agents still must run."""

    consul: str = Field(description="Agent ID of the consul")
    decision: SwarmMessage = Field(description="The posted decision")
    steps: list[CouncilStep] = Field(description="Steps in the order they run")
    delivered: list[str] = Field(
        default_factory=list,
        description="Output agents that already got the trigger alert",
    )


class SwarmCouncil:
    def __init__(self, hub: SwarmHub, bus: SwarmBus, runner: SwarmAgentRunner) -> None:
        self.hub = hub
        self.bus = bus
        self.runner = runner
        # Called after the council saves long-term memory (the UI shows it).
        self.on_memory_change: Callable[[], Awaitable[None]] | None = None

    @staticmethod
    def choose_consul(members: list[str]) -> str:
        """The chair leads; if the chair is asleep, the first awake member does."""
        return next((m for m in members if SWARM_AGENTS[m].chair), members[0])

    async def discuss(
        self, trigger: SwarmMessage, members: list[str]
    ) -> CouncilPlan | None:
        """Discussion rounds, then the decision. Returns the plan to execute."""
        consul = self.choose_consul(members)
        speakers = [m for m in members if m != consul]
        context = {"line": format_chat_line(trigger), "alert": trigger.kind == "alert"}
        usage: SwarmUsage | None = None
        max_rounds = max(1, settings.swarm_council_max_rounds)
        for round_number in range(1, max_rounds + 1):
            await self._round(trigger, speakers, round_number, context)
            last = round_number == max_rounds or not speakers
            turn, turn_usage = await self._ask(
                consul,
                trigger,
                "consul.jinja2",
                ConsulDecision if last else ConsulTurn,
                round=round_number,
                last=last,
                **context,
            )
            usage = merge_usage(usage, turn_usage)
            more = isinstance(turn, ConsulTurn) and turn.next_round
            logger.info(
                f"swarm_council_001: consul=\033[36m{consul}\033[0m round={round_number} "
                f"next_round=\033[35m{more}\033[0m: \033[33m{turn.reasoning}\033[0m"
            )
            if more:
                if turn.remark.strip():
                    await self._post(
                        consul,
                        "opinion",
                        turn.remark.replace("@", ""),
                        trigger,
                        turn_usage,
                        round_number=round_number,
                    )
                continue
            decision = ConsulDecision.model_validate(turn.model_dump())
            return await self._post_decision(consul, decision, trigger, usage)
        return None

    async def _round(
        self,
        trigger: SwarmMessage,
        speakers: list[str],
        round_number: int,
        context: dict[str, Any],
    ) -> None:
        """All members speak at once on the same chat; each posts when ready."""
        history = await self.bus.get_history(settings.swarm_history_size)

        async def speak(member: str) -> None:
            opinion, usage = await self._ask(
                member,
                trigger,
                "council_opinion.jinja2",
                SwarmReply,
                history=history,
                round=round_number,
                **context,
            )
            if opinion.reply.strip():
                await self._post(
                    member,
                    "opinion",
                    opinion.reply.replace("@", ""),
                    trigger,
                    usage,
                    round_number=round_number,
                )

        results = await asyncio.gather(
            *(speak(m) for m in speakers), return_exceptions=True
        )
        for member, result in zip(speakers, results, strict=True):
            if isinstance(result, Exception):
                # One failed member must not stop the discussion.
                logger.error(
                    f"swarm_council_error_001: \033[36m{member}\033[0m failed: "
                    f"\033[31m{result}\033[0m"
                )

    async def execute(self, plan: CouncilPlan) -> None:
        """Give the steps to the agents one by one. A waiting step holds the
        next one until its agent replies, or the step timeout ends."""
        total = len(plan.steps)
        for number, step in enumerate(plan.steps, start=1):
            queue = self.hub.subscribe()
            try:
                task = await self.bus.post(
                    sender=plan.consul,
                    kind="task",
                    text=f"{number}/{total} @{step.agent_id}: {step.task}",
                    reply_to=plan.decision.message_id,
                    depth=plan.decision.depth + 1,
                    mentions=[
                        a
                        for a in add_paired_outputs([step.agent_id])
                        if a == step.agent_id or a not in plan.delivered
                    ],
                    paired=False,
                )
                if step.wait and number < total:
                    await self._wait_reply(queue, task, step.agent_id)
            finally:
                self.hub.unsubscribe(queue)

    async def _wait_reply(
        self, queue: asyncio.Queue, task: SwarmMessage, agent_id: str
    ) -> None:
        async def first_reply() -> None:
            while True:
                event = await queue.get()
                if (
                    isinstance(event, SwarmMessageEvent)
                    and event.message.reply_to == task.message_id
                    and event.message.sender in (agent_id, "system")
                ):
                    return

        try:
            await asyncio.wait_for(first_reply(), settings.swarm_step_timeout_seconds)
        except TimeoutError:
            logger.warning(
                f"swarm_council_warning_001: \033[36m{agent_id}\033[0m gave no reply "
                f"to {task.message_id} in time"
            )
            await self.bus.post(
                sender="system",
                kind="system",
                text=f"{agent_id} did not answer in time, the next step starts",
                reply_to=task.message_id,
                depth=settings.swarm_max_depth,
            )

    async def _ask(
        self,
        agent_id: str,
        trigger: SwarmMessage,
        template: str,
        schema: type[BaseModel],
        history: list[SwarmMessage] | None = None,
        **context: Any,
    ) -> tuple[Any, SwarmUsage]:
        self.hub.publish(SwarmTypingEvent(agent_id=agent_id, active=True))
        try:
            if history is None:
                history = await self.bus.get_history(settings.swarm_history_size)
            return await self.runner.ask(
                SWARM_AGENTS[agent_id],
                trigger,
                history,
                self.runner.render_task(template, **context),
                schema,
            )
        finally:
            self.hub.publish(SwarmTypingEvent(agent_id=agent_id, active=False))

    async def _post_decision(
        self,
        consul: str,
        decision: ConsulDecision,
        trigger: SwarmMessage,
        usage: SwarmUsage,
    ) -> CouncilPlan | None:
        """Only plan steps wake agents; an "@" in the reply is removed, so
        offering an action never runs it (ARCHIE-204)."""
        # An alert already went to its outputs: do not deliver it there again.
        delivered = set(trigger.mentions) & OUTPUT_AGENTS if trigger.kind == "alert" else set()
        steps = [
            s
            for s in decision.steps
            if s.agent_id in SWARM_AGENTS
            and not SWARM_AGENTS[s.agent_id].groups
            and s.agent_id not in delivered
        ]
        await self._remember(decision)
        reply = decision.reply.replace("@", "").strip()
        if not reply and not steps:
            return None
        plan_line = " → ".join(f"@{s.agent_id}" for s in steps)
        message = await self._post(
            consul,
            "decision",
            "\n".join(line for line in [reply, plan_line] if line),
            trigger,
            usage,
            mentions=[],
            paired=False,
        )
        return CouncilPlan(
            consul=consul, decision=message, steps=steps, delivered=sorted(delivered)
        )

    async def _remember(self, decision: ConsulDecision) -> None:
        """Long-term facts and facts for today, as the council chose them."""
        memory = self.runner.memory
        saved = [f for f in decision.remember if await memory.add_fact(f)]
        if decision.remember_today:
            state = await self.runner.get_user_state()
            today = local_now(state.user_timezone).date()
            if await memory.add_today_facts(today, decision.remember_today):
                saved.append("today")
        if saved and self.on_memory_change:
            await self.on_memory_change()

    async def _post(
        self,
        agent_id: str,
        kind: SwarmMessageKind,
        text: str,
        trigger: SwarmMessage,
        usage: SwarmUsage,
        mentions: list[str] | None = None,
        paired: bool = True,
        round_number: int | None = None,
    ) -> SwarmMessage:
        return await self.bus.post(
            sender=agent_id,
            kind=kind,
            text=text,
            reply_to=trigger.message_id,
            depth=trigger.depth + 1,
            usage=usage,
            # Opinions never wake anyone
            mentions=[] if kind == "opinion" else mentions,
            paired=paired,
            round_number=round_number,
        )
