"""Status providers registered by the process composition root."""

from collections.abc import Awaitable, Callable
from pydantic import JsonValue


StatusProvider = Callable[[], Awaitable[dict[str, JsonValue]]]
_PROVIDERS: dict[str, StatusProvider] = {}


def register_status(name: str, provider: StatusProvider) -> None:
    _PROVIDERS[name] = provider


async def agent_status() -> dict[str, dict[str, JsonValue]]:
    return {name: await provider() for name, provider in _PROVIDERS.items()}
