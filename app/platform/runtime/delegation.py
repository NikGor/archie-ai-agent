"""Delegation from a trusted caller to a configured agent (ARCHIE-324).

The chat sets the caller (who it is, which tools it has) around tool execution;
`delegate_agent_tool` calls `delegate`, which reaches the handler registered by
the composition root. The LLM never sets the caller or its tools.
"""

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from app.platform.runtime.spawn import SpawnContext


StepSink = Callable[[str], Awaitable[None]]


class DelegationCaller(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    caller: SpawnContext = Field(description="Trusted caller identity")
    tools: tuple[str, ...] = Field(description="Tools the caller itself may use")
    on_step: StepSink | None = Field(None, description="Progress of the agent")


DelegateHandler = Callable[
    [str, str, DelegationCaller], Awaitable[dict[str, JsonValue]]
]
_CALLER: ContextVar[DelegationCaller | None] = ContextVar(
    "delegation_caller", default=None
)
_handler: DelegateHandler | None = None


@contextmanager
def delegation_scope(caller: DelegationCaller) -> Iterator[None]:
    token = _CALLER.set(caller)
    try:
        yield
    finally:
        _CALLER.reset(token)


def register_delegate_handler(handler: DelegateHandler) -> None:
    global _handler
    _handler = handler


async def delegate(task: str, context: str = "") -> dict[str, JsonValue]:
    caller = _CALLER.get()
    if caller is None:
        raise PermissionError("Delegation requires a trusted caller context")
    if _handler is None:
        raise RuntimeError("Dynamic agents are not configured in this process")
    return await _handler(task, context, caller)
