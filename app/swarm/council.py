"""Council: the main contact of the swarm.

The chair (the council member with the user's persona) answers every user
message without @mentions. A simple request gets the decision at once
(one call). A choice, advice or plan - or an explicit @council - is discussed
first: the other members post one opinion each, then the chair posts the
single council decision. A decision delegates work to home agents with
@mentions; opinions never wake anyone.
"""

import logging
from typing import Any
from archie_shared.swarm.models import (
    SwarmMessage,
    SwarmMessageKind,
    SwarmTypingEvent,
    SwarmUsage,
)
from pydantic import BaseModel
from ..config import settings
from .agents import SWARM_AGENTS
from .bus import SwarmBus, SwarmHub
from .models import CouncilTriage, SwarmReply
from .runner import SwarmAgentRunner, format_chat_line


logger = logging.getLogger(__name__)

COUNCIL_GROUP = "council"
DEFAULT_CHAIR = "business"


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

    async def choose_chair(self, members: list[str]) -> str:
        """The member whose persona the user picked chairs the council."""
        state = await self.runner.state_service.get_user_state(
            user_name=settings.swarm_user_name
        )
        persona = (state.persona or "").lower()
        if persona in members:
            return persona
        return DEFAULT_CHAIR if DEFAULT_CHAIR in members else members[0]

    async def run(self, trigger: SwarmMessage, members: list[str], force: bool) -> None:
        chair = await self.choose_chair(members)
        line = format_chat_line(trigger)
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
            await self._post(chair, "decision", triage.reply, trigger, triage_usage)
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
            chair, trigger, "council_decision.jinja2", SwarmReply, line=line
        )
        await self._post(
            chair, "decision", decision.reply, trigger, merge_usage(triage_usage, usage)
        )

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

    async def _post(
        self,
        agent_id: str,
        kind: SwarmMessageKind,
        text: str,
        trigger: SwarmMessage,
        usage: SwarmUsage,
    ) -> None:
        await self.bus.post(
            sender=agent_id,
            kind=kind,
            text=text,
            reply_to=trigger.message_id,
            depth=trigger.depth + 1,
            usage=usage,
        )
