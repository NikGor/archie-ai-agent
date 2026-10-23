"""Smart home status/query tool — reads real-time device and sensor state via Home Assistant MCP."""

import logging
import re
from typing import Any
from app.backend.homeassistant_mcp_client import HomeAssistantMCPError, call_tool


logger = logging.getLogger(__name__)

_BRIGHTNESS_PATTERN = re.compile(
    r"^(?P<indent>\s*)brightness: '?(?P<value>\d+)'?$", re.MULTILINE
)


def add_brightness_percent(context: str) -> str:
    """Add brightness_pct next to the raw 0-255 light brightness.

    smarthome_control_tool takes brightness in percent, but older Home Assistant
    versions report the raw 0-255 attribute in GetLiveContext, so "brightness: '3'"
    (about 1%) was read as 3%. Newer HA adds brightness_pct itself. ARCHIE-204
    """
    if "brightness_pct" in context:
        return context

    def _with_percent(match: re.Match[str]) -> str:
        raw = int(match["value"])
        percent = max(round(raw / 255 * 100), 1) if raw > 0 else 0
        return f"{match[0]}\n{match['indent']}brightness_pct: '{percent}'"

    return _BRIGHTNESS_PATTERN.sub(_with_percent, context)


async def smarthome_status_tool(
    query: str,
    device_name: str | None = None,
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
        device_name: Filter by entity name or alias (case-insensitive), e.g. "Гостиная термостат"
        domain: Filter by domain: light, climate, sensor, binary_sensor, todo, vacuum
        area: Filter by area/room name, e.g. "Гостиная"

    Returns:
        dict[str, str]: Status text describing matching devices/sensors and their current state
    """
    logger.info(
        f"smarthome_status_001: Query: \033[36m{query}\033[0m, "
        f"device_name: \033[33m{device_name}\033[0m, domain: \033[33m{domain}\033[0m, area: \033[33m{area}\033[0m"
    )
    if demo_mode:
        return {
            "status": "demo",
            "message": "[DEMO] Would fetch live smart home context",
            # ARCHIE-208: covers every sensor type the eval cases ask about
            "context": (
                "Living room thermostat: 22°C, target 21°C, humidity 45%. "
                "Bedroom thermostat: 19°C, target 18°C, humidity 50%. "
                "Living room light sensor: 320 lx. "
                "Kitchen light: on. Living room floor lamp: off."
            ),
        }

    arguments: dict[str, Any] = {}
    if device_name:
        arguments["name"] = device_name
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
        "context": add_brightness_percent(result.get("result", "")),
    }
