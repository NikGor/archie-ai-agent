"""Trusted spawn origins and dispatch without dependencies on agent packages."""

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pydantic import BaseModel, Field


class SpawnContext(BaseModel):
    source: str = Field(
        description="Caller identity set by the runtime, never by an LLM"
    )
    parent_run_id: str = Field(min_length=1, description="Run that owns the request")
    depth: int = Field(0, ge=0, description="Current parent's spawn depth")
    user_timezone: str = Field("UTC", description="Timezone passed to tools")
    demo_mode: bool = Field(
        False, description="Use fixed sample data instead of real tools"
    )


SpawnHandler = Callable[[str, str, SpawnContext, str], Awaitable[BaseModel]]
_CURRENT: ContextVar[SpawnContext | None] = ContextVar("spawn_context", default=None)
_handler: SpawnHandler | None = None


@contextmanager
def spawn_scope(context: SpawnContext) -> Iterator[None]:
    token = _CURRENT.set(context)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def register_spawn_handler(handler: SpawnHandler) -> None:
    global _handler
    _handler = handler


async def spawn(goal: str, context: str = "", publish_to: str = "swarm") -> BaseModel:
    caller = _CURRENT.get()
    if caller is None:
        raise PermissionError("Spawning requires a trusted runtime caller context")
    if _handler is None:
        raise RuntimeError("The ephemeral agent kind is not registered")
    return await _handler(goal, context, caller, publish_to)
