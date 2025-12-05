"""WebSocket models for status updates."""

from pydantic import BaseModel


class StatusUpdate(BaseModel):
    """Status update sent via WebSocket during request processing."""
    step: str
    status: str
    message: str
