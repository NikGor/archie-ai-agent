"""WebSocket models for status updates and streaming."""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

# Callback called with each streaming text chunk (plain/voice formats)
StreamCallback = Callable[[str], Awaitable[None]] | None


class StatusUpdate(BaseModel):
    """Status update sent via WebSocket during request processing."""

    step: str
    status: str
    message: str
    detail: str | None = Field(
        default=None,
        description="Human-readable detail of the action, e.g. 'Ищу в Google: лучшие рестораны'",
    )
