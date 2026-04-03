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
        description="Human-readable detail of the action, e.g. 'Google: best restaurants'",
    )


StatusCallback = Callable[[StatusUpdate], Awaitable[None]] | None


class StatusNotifier:
    """Thin wrapper that eliminates repetitive `if on_status` checks in the pipeline."""

    def __init__(self, on_status: StatusCallback = None):
        self._cb = on_status

    async def emit(
        self, step: str, status: str, message: str, detail: str | None = None
    ) -> None:
        if self._cb:
            await self._cb(StatusUpdate(step=step, status=status, message=message, detail=detail))
