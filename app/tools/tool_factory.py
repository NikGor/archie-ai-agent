"""Factory for managing and executing tools."""

import logging
from typing import Any, Callable
from pydantic import BaseModel, Field
from .weather_tool import get_weather
from .smarthome_tool import control_lights, control_climate


logger = logging.getLogger(__name__)


class GetWeather(BaseModel):
    """Schema for weather tool."""

    location: str


class ControlLights(BaseModel):
    """Schema for light control tool."""

    action: str = Field(
        description="Action to perform: 'turn_on', 'turn_off', 'set_scene', 'dim', 'brighten'"
    )
    target: str = Field(
        default="all",
        description="Target device or group: 'all', 'living_room', 'bedroom', 'kitchen', or specific device name",
    )


class ControlClimate(BaseModel):
    """Schema for climate control tool."""

    action: str = Field(
        description="Action to perform: 'set_temperature', 'increase', 'decrease', 'turn_on', 'turn_off'"
    )
    temperature: float | None = Field(
        default=None,
        description="Target temperature in Celsius (only for 'set_temperature' action)",
    )


class ToolFactory:
    """Factory for registering and executing agent tools."""

    def __init__(self):
        self.tools: dict[str, Callable] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default tools."""
        self.register_tool("get_weather", get_weather)
        self.register_tool("control_lights", control_lights)
        self.register_tool("control_climate", control_climate)
        self.register_tool("ControlLights", control_lights)
        self.register_tool("ControlClimate", control_climate)
        logger.info("tool_factory_001: Registered default tools")

    def register_tool(self, name: str, func: Callable) -> None:
        """Register a tool with the factory."""
        self.tools[name] = func
        logger.info(f"tool_factory_002: Registered tool: \033[36m{name}\033[0m")

    async def execute_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> Any:
        """Execute a tool by name with given arguments."""
        logger.info(
            f"tool_factory_003: Executing \033[36m{tool_name}\033[0m with args: {tool_arguments}"
        )
        try:
            if tool_name not in self.tools:
                logger.error(
                    f"tool_factory_error_001: Unknown tool: \033[31m{tool_name}\033[0m"
                )
                return {"error": f"Unknown tool: {tool_name}"}
            tool_func = self.tools[tool_name]
            if tool_name == "get_weather":
                location = tool_arguments.get("location")
                result = await tool_func(city_name=location)
            elif tool_name in ["ControlLights", "ControlClimate"]:
                result = await tool_func(**tool_arguments)
            else:
                result = await tool_func(**tool_arguments)
            logger.info(
                f"tool_factory_004: Tool \033[36m{tool_name}\033[0m executed successfully"
            )
            return result
        except Exception as e:
            logger.error(
                f"tool_factory_error_002: Tool execution failed: \033[31m{e!s}\033[0m"
            )
            return {"error": f"Tool execution failed: {e!s}"}

    def get_tool_definitions(self, for_dashboard: bool = False) -> list[type[BaseModel]]:
        """Get tool definitions for OpenAI API."""
        smarthome_tools = [ControlLights, ControlClimate]
        if for_dashboard:
            return smarthome_tools
        return [GetWeather]
