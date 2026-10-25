"""Capacity and depth policy of the runtime's on-demand agents (ARCHIE-300)."""

from app.platform.runtime.spawn import SpawnContext


class SpawnDeniedError(PermissionError):
    """A caller does not have permission or capacity to create another run."""


class SpawnGate:
    """One gate per runtime process; acquire/release contain no await points.

    ARCHIE-300 runs only in the single archie-runtime process. Sharing the gate
    on the registered service makes the capacity bound atomic for its asyncio
    tasks; multiple runtime replicas will need a distributed lease instead.
    """

    def __init__(
        self,
        sources: frozenset[str],
        tools: frozenset[str],
        max_concurrent: int,
        max_depth: int,
    ) -> None:
        if max_concurrent < 1 or max_depth < 1:
            raise ValueError("Spawn capacity and maximum depth must be positive")
        self.sources = sources
        self.tools = tools
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self.active = 0

    def acquire(self, caller: SpawnContext, tools: tuple[str, ...]) -> None:
        if caller.source not in self.sources:
            raise SpawnDeniedError(f"{caller.source} may not spawn agents")
        if caller.depth >= self.max_depth:
            raise SpawnDeniedError("Maximum spawn depth reached")
        if not set(tools) <= self.tools:
            raise SpawnDeniedError("Only approved read tools are allowed")
        if self.active >= self.max_concurrent:
            raise SpawnDeniedError("Concurrent spawn capacity reached")
        self.active += 1

    def release(self) -> None:
        self.active -= 1
