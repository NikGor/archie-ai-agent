"""Smart home status/query tool — reads real-time device and sensor state via Home Assistant MCP."""

import logging
from typing import Any
from app.backend.homeassistant_mcp_client import HomeAssistantMCPError, call_tool


logger = logging.getLogger(__name__)


async def smarthome_status_tool(
    query: str,
    name: str | None = None,
    domain: str | None = None,
    area: str | None = None,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Get the CURRENT real-time state of smart home devices and sensors: temperature, humidity,
    whether a light/climate device is on or off, brightness, motion, window/door state, etc.
    Use this tool for ANY question about current conditions in the home, e.g.
    "what's the temperature in the living room", "is the kitchen light on", "how humid is the bedroom".
    Do NOT use smarthome_control_tool for these queries — that tool only changes state,
    it cannot read it.

    Args:
        query: Original user question, used for logging/context only
        name: Filter by entity name or alias (case-insensitive), e.g. "Гостиная термостат"
        domain: Filter by domain: light, climate, sensor, binary_sensor, todo
        area: Filter by area/room name, e.g. "Гостиная"

    Returns:
        dict[str, str]: Status text describing matching devices/sensors and their current state
    """
    logger.info(
        f"smarthome_status_001: Query: \033[36m{query}\033[0m, "
        f"name: \033[33m{name}\033[0m, domain: \033[33m{domain}\033[0m, area: \033[33m{area}\033[0m"
    )
    if demo_mode:
        return {
            "status": "demo",
            "message": "[DEMO] Would fetch live smart home context",
            "context": "Living room thermostat: 22°C, humidity 45%. Kitchen light: on.",
        }

    arguments: dict[str, Any] = {}
    if name:
        arguments["name"] = name
    if domain:
        arguments["domain"] = domain
    if area:
        arguments["area"] = area

    try:
        result = await call_tool("GetLiveContext", arguments)
    except HomeAssistantMCPError as e:
        logger.error(f"smarthome_status_error_001: \033[31m{e}\033[0m")
        return {
            "status": "error",
            "message": f"Home Assistant error: {e}",
        }

    logger.info("smarthome_status_002: Fetched live context successfully")
    return {
        "status": "success",
        "context": result.get("result", ""),
    }
