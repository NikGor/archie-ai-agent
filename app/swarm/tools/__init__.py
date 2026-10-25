"""Tools only swarm agents get; the chat does not see them (ARCHIE-246, ARCHIE-292)."""

from ...platform.tools.registry import ToolGroup
from ...tools.catalog import tool_registry


SWARM_TOOL_GROUP = ToolGroup(
    name="swarm",
    exclusive=True,
    tools={
        "weather_tool": "app.tools.weather_tool",
        "alert_chat_tool": "app.swarm.tools.alert_chat_tool",
        "swarm_alert_tool": "app.swarm.tools.swarm_alert_tool",
    },
)

tool_registry.add_group(SWARM_TOOL_GROUP)
