"""WebSocket models for status updates."""

from pydantic import BaseModel, Field


class StatusUpdate(BaseModel):
    """Status update sent via WebSocket during request processing."""

    step: str
    status: str
    message: str
    detail: str | None = Field(
        default=None,
        description="Human-readable detail of the action, e.g. 'Ищу в Google: лучшие рестораны'",
    )
