"""Tool contracts shared by every agent kind (ARCHIE-292): tool names, the
neutral call, its result and the per-call context."""

from typing import Any, Literal
from pydantic import BaseModel, Field


ToolName = Literal[
    "smarthome_control_tool",
    "smarthome_status_tool",
    "spotify_tool",
    "tv_tool",
    "google_search_tool",
    "google_places_search_tool",
    "task_tool",
    "notes_tool",
    "events_tool",
    "gmail_tool",
    "cron_tool",
    "announce_tool",
    "awtrix_tool",
    "weather_tool",
    "spawn_agent_tool",
    "alert_chat_tool",
    "swarm_alert_tool",
    "agent_task_tool",
    "document_search_tool",
    "skill_loader_tool",
]


class Parameter(BaseModel):
    name: str = Field(description="Parameter name")
    value: str = Field(description="Parameter value")


class ToolCall(BaseModel):
    """One tool call as the executor runs it, whatever agent kind asked for it.

    Stage 1 of the chat asks with ToolCallRequest (with confirmation fields for
    the LLM); the swarm builds ToolCall directly.
    """

    tool_name: ToolName = Field(description="Name of the tool to call")
    arguments: list[Parameter] = Field(description="Arguments with their values")
    reason: str = Field(default="", description="Why the tool is called")


class ToolResult(BaseModel):
    """Result from tool execution."""

    tool_name: str = Field(description="Name of the executed tool")
    output: dict[str, Any] = Field(description="Tool output data")
    success: bool = Field(
        default=True, description="Whether tool executed successfully"
    )
    error: str | None = Field(default=None, description="Error message if failed")


class ToolContext(BaseModel):
    """Facts of the request that the executor gives to every tool that declares
    them as parameters. They stay out of the tool schema (ARCHIE-257).

    `demo_mode` is the execution mode of this call (ARCHIE-292, ARCHIE-171):
    demo data for the chat demo and evals, no side effects for a dry-run swarm.
    """

    user_timezone: str = Field(
        description="IANA timezone of the user, e.g. Europe/Berlin"
    )
    demo_mode: bool = Field(
        default=False,
        description="Return demo data and make no side effects",
    )
