"""Smart home control tool — turns devices on/off and changes their settings via Home Assistant MCP."""

import logging
import re
from typing import Any
from app.backend.homeassistant_mcp_client import HomeAssistantMCPError, call_tool
from app.utils.arg_coercion import coerce_float, coerce_int


logger = logging.getLogger(__name__)

_TEMPERATURE_STEP = 1.0  # degrees Celsius per increase_temperature/decrease_temperature
_CURRENT_TEMP_PATTERN = re.compile(r"current_temperature:\s*'?(-?\d+\.?\d*)")


def _extract_current_temperature(live_context: dict[str, Any]) -> float | None:
    """Pull the current_temperature attribute out of GetLiveContext's freeform text result."""
    text = live_context.get("result")
    if not isinstance(text, str):
        return None
    match = _CURRENT_TEMP_PATTERN.search(text)
    return float(match.group(1)) if match else None


async def smarthome_control_tool(  # noqa: PLR0911, PLR0912
    action: str,
    device_name: str | None = None,
    area: str | None = None,
    floor: str | None = None,
    brightness: int | str | None = None,
    color_temp: int | str | None = None,
    color: str | None = None,
    temperature: float | str | None = None,
    volume_percent: int | str | None = None,
    volume_step: str | None = None,
    search_query: str | None = None,
    media_class: str | None = None,
    message: str | None = None,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Control smart home devices via Home Assistant: lights, climate, media players, timers,
    and whole-home voice broadcasts. This tool only CHANGES state — it cannot answer
    "is the light on"/"what's the temperature" or similar questions, use smarthome_status_tool for those.
    IMPORTANT: Users don't know numeric values (brightness percentages, Kelvin temperatures, etc.).
    Choose appropriate values based on context and NEVER mention raw numbers to the user.

    Args:
        action: One of: turn_on, turn_off, set_light, set_temperature, increase_temperature,
            decrease_temperature, media_pause, media_unpause, media_next, media_previous,
            media_set_volume, media_volume_step, media_mute, media_unmute, media_search_and_play,
            broadcast, cancel_timers, activate_scene
        device_name: Entity name/alias to target, e.g. 'Floor Lamp Living Room', 'Living Room Thermostat'.
            For 'activate_scene', the scene name, e.g. 'Jazz Bar', 'Рождество'
        area: Area/room name to target instead of/in addition to name, e.g. 'Гостиная'
        floor: Floor name to target
        brightness: 0-100 percent, for 'set_light'. Interpret naturally: 'brighter'=+20, 'dim'=-20, 'max'=100
        color_temp: Color warmth in Kelvin, for 'set_light'. 2700=warm, 4000=neutral, 6500=cold
        color: Named color for 'set_light', e.g. 'red', 'orange', 'pink', 'purple'.
            MUST be a plain color name — Home Assistant rejects hex codes like '#FF0000'.
        temperature: Target temperature in Celsius, required for 'set_temperature'
        volume_percent: Volume 0-100, required for 'media_set_volume'
        volume_step: 'up' or 'down', required for 'media_volume_step'
        search_query: What to search for, required for 'media_search_and_play'
        media_class: Media type for 'media_search_and_play', e.g. 'music', 'playlist', 'movie'
        message: Message to announce, required for 'broadcast'

    Returns:
        dict[str, Any]: Status and Home Assistant MCP result
    """
    brightness = coerce_int(brightness)
    color_temp = coerce_int(color_temp)
    temperature = coerce_float(temperature)
    volume_percent = coerce_int(volume_percent)

    logger.info(
        f"smarthome_control_001: Action \033[36m{action}\033[0m, device_name: \033[33m{device_name}\033[0m, "
        f"area: \033[33m{area}\033[0m, demo_mode: \033[35m{demo_mode}\033[0m"
    )

    if demo_mode:
        return {
            "status": "demo",
            "message": f"[DEMO] {action} would be executed",
            "applied_settings": {
                "action": action,
                "device_name": device_name,
                "area": area,
                "brightness": brightness,
                "color_temp": color_temp,
                "color": color,
                "temperature": temperature,
            },
        }

    target: dict[str, Any] = {}
    if device_name:
        target["name"] = device_name
    if area:
        target["area"] = area
    if floor:
        target["floor"] = floor

    try:
        if action == "turn_on":
            ha_result = await call_tool("HassTurnOn", target)
        elif action == "turn_off":
            ha_result = await call_tool("HassTurnOff", target)
        elif action == "set_light":
            set_args = dict(target)
            if brightness is not None:
                set_args["brightness"] = brightness
            if color_temp is not None:
                set_args["temperature"] = color_temp
            if color is not None:
                set_args["color"] = color
            ha_result = await call_tool("HassLightSet", set_args)
        elif action == "set_temperature":
            if temperature is None:
                return {
                    "status": "error",
                    "message": "temperature is required for action=set_temperature",
                }
            ha_result = await call_tool(
                "HassClimateSetTemperature", {**target, "temperature": temperature}
            )
        elif action in ("increase_temperature", "decrease_temperature"):
            live_context = await call_tool(
                "GetLiveContext", {**target, "domain": "climate"}
            )
            current_temp = _extract_current_temperature(live_context)
            if current_temp is None:
                return {
                    "status": "error",
                    "message": f"Could not read current temperature for {device_name or area}",
                }
            temperature = current_temp + (
                _TEMPERATURE_STEP
                if action == "increase_temperature"
                else -_TEMPERATURE_STEP
            )
            ha_result = await call_tool(
                "HassClimateSetTemperature", {**target, "temperature": temperature}
            )
        elif action == "media_pause":
            ha_result = await call_tool("HassMediaPause", target)
        elif action == "media_unpause":
            ha_result = await call_tool("HassMediaUnpause", target)
        elif action == "media_next":
            ha_result = await call_tool("HassMediaNext", target)
        elif action == "media_previous":
            ha_result = await call_tool("HassMediaPrevious", target)
        elif action == "media_set_volume":
            if volume_percent is None:
                return {
                    "status": "error",
                    "message": "volume_percent is required for action=media_set_volume",
                }
            ha_result = await call_tool(
                "HassSetVolume", {**target, "volume_level": volume_percent}
            )
        elif action == "media_volume_step":
            if not volume_step:
                return {
                    "status": "error",
                    "message": "volume_step is required for action=media_volume_step",
                }
            ha_result = await call_tool(
                "HassSetVolumeRelative", {**target, "volume_step": volume_step}
            )
        elif action == "media_mute":
            ha_result = await call_tool(
                "HassMediaPlayerMute", {**target, "is_volume_muted": "true"}
            )
        elif action == "media_unmute":
            ha_result = await call_tool(
                "HassMediaPlayerUnmute", {**target, "is_volume_muted": "false"}
            )
        elif action == "media_search_and_play":
            if not search_query:
                return {
                    "status": "error",
                    "message": "search_query is required for action=media_search_and_play",
                }
            search_args = {**target, "search_query": search_query}
            if media_class:
                search_args["media_class"] = media_class
            ha_result = await call_tool("HassMediaSearchAndPlay", search_args)
        elif action == "broadcast":
            if not message:
                return {
                    "status": "error",
                    "message": "message is required for action=broadcast",
                }
            ha_result = await call_tool("HassBroadcast", {"message": message})
        elif action == "cancel_timers":
            ha_result = await call_tool("HassCancelAllTimers", target)
        elif action == "activate_scene":
            if not device_name:
                return {
                    "status": "error",
                    "message": "device_name is required for action=activate_scene",
                }
            ha_result = await call_tool("HassTurnOn", target)
        else:
            return {
                "status": "error",
                "message": f"Unknown action: {action}",
            }
    except HomeAssistantMCPError as e:
        logger.error(f"smarthome_control_error_001: \033[31m{e}\033[0m")
        return {
            "status": "error",
            "message": f"Home Assistant error: {e}",
        }

    logger.info(f"smarthome_control_002: Action \033[36m{action}\033[0m succeeded")
    return {
        "status": "success",
        "message": f"{action} executed successfully",
        "applied_settings": {
            "action": action,
            "device_name": device_name,
            "area": area,
            "brightness": brightness,
            "color_temp": color_temp,
            "color": color,
            "temperature": temperature,
        },
        "ha_result": ha_result,
    }
