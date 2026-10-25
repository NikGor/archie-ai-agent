"""Which tools the chat pipeline shows to Stage 1 (ARCHIE-292)."""

from app.agents.chat.formats import resolve_format
from app.tools.catalog import tool_registry


def chat_tool_names(response_format: str) -> list[str]:
    """Tool names for a response format; exclusive groups (swarm) are left out."""
    return tool_registry.names(
        resolve_format(response_format).tool_groups, include_exclusive=False
    )
