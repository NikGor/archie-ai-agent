"""WebSocket models for status updates and streaming."""

from collections.abc import Awaitable, Callable
from pydantic import BaseModel, Field


# Callback called with each streaming text chunk (plain/voice formats)
StreamCallback = Callable[[str], Awaitable[None]] | None

# Callback for generic stream events: (event_type, optional_text_chunk)
# Used for stream_placeholder and stream_reasoning events (UI formats)
StreamEventCallback = Callable[[str, str | None], Awaitable[None]] | None


class StatusUpdate(BaseModel):
    """Status update sent via WebSocket during request processing."""

    step: str
    status: str
    message: str
    detail: str | None = Field(
        default=None,
        description="Human-readable detail of the action, e.g. 'Ищу в Google: лучшие рестораны'",
    )
