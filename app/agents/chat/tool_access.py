"""Which tools the chat pipeline shows to Stage 1 (ARCHIE-292)."""

from app.tools.catalog import tool_registry


# Dashboard tiles and widgets only read and change the smart home
_FORMAT_GROUPS: dict[str, tuple[str, ...]] = {
    "dashboard": ("smarthome",),
    "widget": ("smarthome",),
}


def chat_tool_names(response_format: str) -> list[str]:
    """Tool names for a response format; exclusive groups (swarm) are left out."""
    return tool_registry.names(
        _FORMAT_GROUPS.get(response_format), include_exclusive=False
    )
