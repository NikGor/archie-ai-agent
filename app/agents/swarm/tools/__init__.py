"""Tools only swarm agents get; the chat does not see them (ARCHIE-246, ARCHIE-292)."""

from app.platform.tools.registry import READ, WRITE, ToolGroup
from app.tools.catalog import tool_registry


SWARM_TOOL_GROUP = ToolGroup(
    name="swarm",
    exclusive=True,
    tools={
        "spawn_agent_tool": "app.agents.swarm.tools.spawn_agent_tool",
        "weather_tool": "app.tools.weather_tool",
        "alert_chat_tool": "app.agents.swarm.tools.alert_chat_tool",
        "swarm_alert_tool": "app.agents.swarm.tools.swarm_alert_tool",
    },
    capabilities={
        "spawn_agent_tool": READ,
        "weather_tool": READ,
        "alert_chat_tool": WRITE,
        "swarm_alert_tool": WRITE,
    },
)

tool_registry.add_group(SWARM_TOOL_GROUP)
