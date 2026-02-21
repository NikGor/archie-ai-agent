"""Pydantic models for tool execution results."""

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
