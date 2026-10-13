"""Wires the swarm components together; started from the FastAPI lifespan."""

from .agents import SwarmRoster
from .bus import SwarmBus, SwarmHub
from .dispatcher import SwarmDispatcher
from .gate import LlmGate
from .runner import SwarmAgentRunner


class SwarmService:
    def __init__(self) -> None:
        self.hub = SwarmHub()
        self.bus = SwarmBus(self.hub)
        self.roster = SwarmRoster()
        self.dispatcher = SwarmDispatcher(
            hub=self.hub,
            bus=self.bus,
            roster=self.roster,
            runner=SwarmAgentRunner(),
            gate=LlmGate(),
        )

    async def start(self) -> None:
        await self.dispatcher.start()

    async def stop(self) -> None:
        await self.dispatcher.stop()
        await self.bus.redis.aclose()
        await self.roster.redis.aclose()


swarm_service = SwarmService()
