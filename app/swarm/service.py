"""Wires the swarm components together; started from the FastAPI lifespan."""

from .agents import SwarmRoster
from .bus import SwarmBus, SwarmHub
from .council import SwarmCouncil
from .dispatcher import SwarmDispatcher
from .journal import SwarmJournal
from .memory import SwarmMemory
from .runner import SwarmAgentRunner
from .ticker import SwarmTicker


class SwarmService:
    def __init__(self) -> None:
        self.hub = SwarmHub()
        self.bus = SwarmBus(self.hub)
        self.journal = SwarmJournal(self.hub)
        self.roster = SwarmRoster()
        self.memory = SwarmMemory()
        runner = SwarmAgentRunner(roster=self.roster, memory=self.memory)
        council = SwarmCouncil(hub=self.hub, bus=self.bus, runner=runner)
        self.dispatcher = SwarmDispatcher(
            hub=self.hub,
            bus=self.bus,
            roster=self.roster,
            runner=runner,
            council=council,
            memory=self.memory,
        )
        self.ticker = SwarmTicker(
            hub=self.hub,
            bus=self.bus,
            roster=self.roster,
            runner=runner,
            memory=self.memory,
            dispatcher=self.dispatcher,
        )
        council.on_memory_change = self.ticker.publish_control

    async def start(self) -> None:
        # The journal subscribes first, so it sees every message
        await self.journal.start()
        await self.dispatcher.start()
        await self.ticker.start()

    async def stop(self) -> None:
        await self.ticker.stop()
        await self.dispatcher.stop()
        await self.journal.stop()
        await self.bus.redis.aclose()
        await self.roster.redis.aclose()
        await self.memory.redis.aclose()


swarm_service = SwarmService()
