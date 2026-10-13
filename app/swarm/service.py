"""Wires the swarm components together; started from the FastAPI lifespan."""

from .agents import SwarmRoster
from .bus import SwarmBus, SwarmHub
from .council import SwarmCouncil
from .dispatcher import SwarmDispatcher
from .runner import SwarmAgentRunner


class SwarmService:
    def __init__(self) -> None:
        self.hub = SwarmHub()
        self.bus = SwarmBus(self.hub)
        self.roster = SwarmRoster()
        runner = SwarmAgentRunner()
        self.dispatcher = SwarmDispatcher(
            hub=self.hub,
            bus=self.bus,
            roster=self.roster,
            runner=runner,
            council=SwarmCouncil(hub=self.hub, bus=self.bus, runner=runner),
        )

    async def start(self) -> None:
        await self.dispatcher.start()

    async def stop(self) -> None:
        await self.dispatcher.stop()
        await self.bus.redis.aclose()
        await self.roster.redis.aclose()


swarm_service = SwarmService()
