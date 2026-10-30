"""Tools every agent kind may use, registered in the process tool registry
(ARCHIE-292). Packages with their own tools add exclusive groups (app.agents.swarm.tools)."""

from ..platform.tools.registry import (
    READ,
    WRITE,
    ToolCapability,
    ToolGroup,
    ToolRegistry,
)


CORE_TOOL_GROUPS = [
    ToolGroup(
        name="smarthome",
        tools={
            "smarthome_control_tool": "app.tools.smarthome_control_tool",
            "smarthome_status_tool": "app.tools.smarthome_status_tool",
            "spotify_tool": "app.tools.spotify_tool",
        },
        capabilities={
            "smarthome_control_tool": WRITE,
            "smarthome_status_tool": READ,
            "spotify_tool": WRITE,
        },
    ),
    ToolGroup(
        name="tv",
        tools={"tv_tool": "app.tools.tv_tool"},
        capabilities={"tv_tool": WRITE},
    ),
    ToolGroup(
        name="internet_search",
        tools={
            "google_search_tool": "app.tools.google_search_tool",
            "google_places_search_tool": "app.tools.google_places_search_tool",
        },
        capabilities={"google_search_tool": READ, "google_places_search_tool": READ},
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
        capabilities={
            "task_tool": WRITE,
            "notes_tool": WRITE,
            "events_tool": ToolCapability(
                effect="write",
                read_actions=("list", "read", "today", "upcoming"),
                approval=True,
            ),
            "gmail_tool": READ,
            "cron_tool": ToolCapability(
                effect="write", read_actions=("list", "get"), approval=True
            ),
            "announce_tool": WRITE,
            "awtrix_tool": WRITE,
            "agent_task_tool": WRITE,
        },
    ),
    ToolGroup(
        name="knowledge",
        tools={"document_search_tool": "app.tools.document_search_tool"},
        capabilities={"document_search_tool": READ},
    ),
    ToolGroup(
        name="agents",
        tools={"delegate_agent_tool": "app.agents.chat.tools.delegate_agent_tool"},
        capabilities={"delegate_agent_tool": READ},
    ),
    ToolGroup(
        name="skills",
        tools={"skill_loader_tool": "app.tools.skill_loader_tool"},
        capabilities={"skill_loader_tool": READ},
    ),
]

tool_registry = ToolRegistry(CORE_TOOL_GROUPS)
