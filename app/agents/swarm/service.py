"""Wires the swarm components together; started from the FastAPI lifespan."""

from typing import TYPE_CHECKING
from app.agents.swarm.agents import SwarmRoster
from app.agents.swarm.bus import SwarmBus, SwarmHub
from app.agents.swarm.council import SwarmCouncil
from app.agents.swarm.dispatcher import SwarmDispatcher
from app.agents.swarm.journal import SwarmJournal
from app.agents.swarm.memory import SwarmMemory
from app.agents.swarm.runner import SwarmAgentRunner
from app.agents.swarm.ticker import SwarmTicker


if TYPE_CHECKING:
    from app.agents.swarm.bridge import SwarmHubBridge


class SwarmService:
    name = "swarm"

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
        # Set by the composition root in RUNTIME_MODE api / runtime (ARCHIE-299)
        self.bridge: "SwarmHubBridge | None" = None

    async def start(self) -> None:
        # The journal subscribes first, so it sees every message
        await self.journal.start()
        await self.dispatcher.start()
        await self.ticker.start()

    async def stop(self) -> None:
        await self.ticker.stop()
        await self.dispatcher.stop()
        await self.journal.stop()

    def healthy(self) -> bool:
        """Every swarm loop is alive (the supervisor restarts the swarm, ARCHIE-298)."""
        workers = [self.ticker._worker, self.dispatcher._worker]
        if self.journal.is_configured():
            workers.append(self.journal._worker)
        return all(worker is not None and not worker.done() for worker in workers)

    async def close(self) -> None:
        """Close Redis at process exit; stop() keeps it for a restart."""
        await self.bus.redis.aclose()
        await self.roster.redis.aclose()
        await self.memory.redis.aclose()


swarm_service = SwarmService()
