"""Council: the main contact of the swarm.

The chair (the council member with the user's persona) answers every user
message without @mentions. A simple request gets the decision at once
(one call). A choice, advice or plan - or an explicit @council - is discussed
first: the other members post one opinion each, then the chair posts the
single council decision. A decision delegates work to home agents with
@mentions; opinions never wake anyone.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any
from archie_shared.swarm.models import (
    SwarmMessage,
    SwarmMessageKind,
    SwarmTypingEvent,
    SwarmUsage,
)
from pydantic import BaseModel
from ..config import settings
from .agents import SWARM_AGENTS, add_paired_outputs
from .bus import SwarmBus, SwarmHub
from .models import CouncilAlertDecision, CouncilDecision, CouncilTriage, SwarmReply
from .runner import SwarmAgentRunner, format_chat_line
from .schedule import local_now


logger = logging.getLogger(__name__)

COUNCIL_GROUP = "council"
DEFAULT_CHAIR = "business"
# Agents that deliver data to the user; they wait for the agents that find it.
OUTPUT_AGENTS = frozenset({"display", "voice", "chat"})
RELAYS_MAX = 20


def merge_usage(first: SwarmUsage, second: SwarmUsage) -> SwarmUsage:
    return SwarmUsage(
        model=second.model,
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        total_cost=first.total_cost + second.total_cost,
    )


class SwarmCouncil:
    def __init__(self, hub: SwarmHub, bus: SwarmBus, runner: SwarmAgentRunner) -> None:
        self.hub = hub
        self.bus = bus
        self.runner = runner
        # Called after the council saves long-term memory (the UI shows it).
        self.on_memory_change: Callable[[], Awaitable[None]] | None = None
        # Decision ID -> output agents that wait for the first data reply to it
        self.relays: dict[str, list[str]] = {}

    def pop_relay(self, message: SwarmMessage) -> list[str]:
        """Output agents waiting for this message: the first reply of a home
        agent to a decision that also gave them a task (ARCHIE-240)."""
        if message.reply_to not in self.relays:
            return []
        if message.sender not in SWARM_AGENTS or message.sender in OUTPUT_AGENTS:
            return []
        if message.kind not in ("report", "action"):
            return []
        return self.relays.pop(message.reply_to)

    async def choose_chair(self, members: list[str]) -> str:
        """The member whose persona the user picked chairs the council."""
        state = await self.runner.get_user_state()
        persona = (state.persona or "").lower()
        if persona in members:
            return persona
        return DEFAULT_CHAIR if DEFAULT_CHAIR in members else members[0]

    async def run(self, trigger: SwarmMessage, members: list[str], force: bool) -> None:
        chair = await self.choose_chair(members)
        line = format_chat_line(trigger)
        if trigger.kind == "alert":
            await self._review_alert(chair, trigger, line)
            return
        triage, triage_usage = await self._ask(
            chair,
            trigger,
            "council_triage.jinja2",
            CouncilTriage,
            line=line,
            force=force,
        )
        logger.info(
            f"swarm_council_001: chair=\033[36m{chair}\033[0m convene="
            f"\033[35m{triage.convene or force}\033[0m: \033[33m{triage.reasoning}\033[0m"
        )
        if not (triage.convene or force) and triage.reply.strip():
            await self._post_decision(chair, triage, trigger, triage_usage)
            return
        for member in members:
            if member == chair:
                continue
            opinion, usage = await self._ask(
                member, trigger, "council_opinion.jinja2", SwarmReply, line=line
            )
            if opinion.reply.strip():
                await self._post(member, "opinion", opinion.reply, trigger, usage)
        decision, usage = await self._ask(
            chair, trigger, "council_decision.jinja2", CouncilDecision, line=line
        )
        await self._post_decision(
            chair, decision, trigger, merge_usage(triage_usage, usage)
        )

    async def _review_alert(self, chair: str, trigger: SwarmMessage, line: str) -> None:
        """The chair decides if the home acts on an alert by itself (ARCHIE-245)."""
        decision, usage = await self._ask(
            chair, trigger, "council_alert.jinja2", CouncilAlertDecision, line=line
        )
        logger.info(
            f"swarm_council_002: alert from \033[36m{trigger.sender}\033[0m: "
            f"delegations={[d.agent_id for d in decision.delegations]} "
            f"reasoning: \033[33m{decision.reasoning}\033[0m"
        )
        if decision.delegations:
            await self._post_decision(chair, decision, trigger, usage)
        else:
            await self._remember(decision)

    async def _ask(
        self,
        agent_id: str,
        trigger: SwarmMessage,
        template: str,
        schema: type[BaseModel],
        **context: Any,
    ) -> tuple[Any, SwarmUsage]:
        self.hub.publish(SwarmTypingEvent(agent_id=agent_id, active=True))
        try:
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
        chair: str,
        decision: CouncilDecision,
        trigger: SwarmMessage,
        usage: SwarmUsage,
    ) -> None:
        """Only structured delegations wake agents; an "@" in the reply text is
        removed, so offering an action never runs it (ARCHIE-204)."""
        delegations = [
            d
            for d in decision.delegations
            if d.agent_id in SWARM_AGENTS and not SWARM_AGENTS[d.agent_id].groups
        ]
        await self._remember(decision)
        lines = [decision.reply.replace("@", "").strip()]
        lines += [f"→ @{d.agent_id}: {d.task}" for d in delegations]
        # An alert already went to its outputs: do not deliver it there again.
        delivered = set(trigger.mentions) if trigger.kind == "alert" else set()
        mentions = [
            a
            for a in add_paired_outputs(
                list(dict.fromkeys(d.agent_id for d in delegations))
            )
            if a not in delivered or a not in OUTPUT_AGENTS
        ]
        outputs = [a for a in mentions if a in OUTPUT_AGENTS]
        sources = [a for a in mentions if a not in OUTPUT_AGENTS]
        message = await self._post(
            chair,
            "decision",
            "\n".join(line for line in lines if line),
            trigger,
            usage,
            # Display and voice start after the data arrives, not before it.
            mentions=sources if outputs and sources else mentions,
            paired=False,
        )
        if outputs and sources:
            self.relays[message.message_id] = outputs
            while len(self.relays) > RELAYS_MAX:
                self.relays.pop(next(iter(self.relays)))

    async def _remember(self, decision: CouncilDecision) -> None:
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
    ) -> SwarmMessage:
        return await self.bus.post(
            sender=agent_id,
            kind=kind,
            text=text,
            reply_to=trigger.message_id,
            depth=trigger.depth + 1,
            usage=usage,
            mentions=mentions,
            paired=paired,
        )
