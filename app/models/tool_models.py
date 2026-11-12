"""Pydantic models for tool execution: results and request context."""

from typing import Any
from pydantic import BaseModel, Field


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
    them as parameters. They stay out of the tool schema, like demo_mode (ARCHIE-257).
    """

    user_timezone: str = Field(
        description="IANA timezone of the user, e.g. Europe/Berlin"
    )
