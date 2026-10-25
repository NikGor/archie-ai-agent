"""Tools every agent kind may use, registered in the process tool registry
(ARCHIE-292). Packages with their own tools add exclusive groups (app.swarm.tools)."""

from ..platform.tools.registry import ToolGroup, ToolRegistry


CORE_TOOL_GROUPS = [
    ToolGroup(
        name="smarthome",
        tools={
            "smarthome_control_tool": "app.tools.smarthome_control_tool",
            "smarthome_status_tool": "app.tools.smarthome_status_tool",
            "spotify_tool": "app.tools.spotify_tool",
        },
    ),
    ToolGroup(name="tv", tools={"tv_tool": "app.tools.tv_tool"}),
    ToolGroup(
        name="internet_search",
        tools={
            "google_search_tool": "app.tools.google_search_tool",
            "google_places_search_tool": "app.tools.google_places_search_tool",
        },
    ),
    ToolGroup(
        name="productivity",
        tools={
            "task_tool": "app.tools.task_tool",
            "notes_tool": "app.tools.notes_tool",
            "events_tool": "app.tools.events_tool",
            "gmail_tool": "app.tools.gmail_tool",
            "cron_tool": "app.tools.cron_tool",
            "announce_tool": "app.tools.announce_tool",
            "awtrix_tool": "app.tools.awtrix_tool",
            "agent_task_tool": "app.agents.chat.tools.agent_task_tool",
        },
    ),
    ToolGroup(
        name="knowledge",
        tools={"document_search_tool": "app.tools.document_search_tool"},
    ),
    ToolGroup(
        name="skills", tools={"skill_loader_tool": "app.tools.skill_loader_tool"}
    ),
]

tool_registry = ToolRegistry(CORE_TOOL_GROUPS)
