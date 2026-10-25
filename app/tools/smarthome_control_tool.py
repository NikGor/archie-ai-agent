"""Smart home control tool — turns devices on/off and changes their settings via Home Assistant MCP."""

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any
from pydantic import BaseModel, Field
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


async def _search_and_play(
    target: dict[str, Any], search_query: str, media_class: str | None
) -> dict[str, Any]:
    """Search a media player and play the best result (HA has no single search-and-play tool). # ARCHIE-214"""
    player = {f"player_{key}": value for key, value in target.items()}
    search = await call_tool(
        "media_player__search_media", {**player, "search_query": search_query}
    )
    results = search.get("results", [])
    if not results:
        raise HomeAssistantMCPError(f"Nothing found for '{search_query}'")
    matching = [item for item in results if item.get("media_class") == media_class]
    item = (matching or results)[0]
    return await call_tool(
        "media_player__play_media",
        {
            **player,
            "media_content_id": item["media_content_id"],
            "media_content_type": item["media_content_type"],
        },
    )


class ControlRequest(BaseModel):
    """Arguments of one smarthome_control_tool call after coercion (ARCHIE-304)."""

    device_name: str | None = Field(default=None, description="Entity name or scene")
    area: str | None = Field(default=None, description="Area name")
    floor: str | None = Field(default=None, description="Floor name")
    brightness: int | None = Field(default=None, description="0-100 percent")
    color_temp: int | None = Field(default=None, description="Kelvin")
    color: str | None = Field(default=None, description="Named color")
    temperature: float | None = Field(
        default=None, description="Target or set temperature"
    )
    volume_percent: int | None = Field(default=None, description="Volume 0-100")
    volume_step: str | None = Field(default=None, description="up | down")
    search_query: str | None = Field(default=None, description="Media search")
    media_class: str | None = Field(default=None, description="Media type")
    message: str | None = Field(default=None, description="Broadcast text")

    def target(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in (
                ("name", self.device_name),
                ("area", self.area),
                ("floor", self.floor),
            )
            if value
        }


class ControlArgumentError(ValueError):
    """A required argument of the action is missing."""


def _require(value: object, name: str, action: str) -> None:
    if value is None or value == "":
        raise ControlArgumentError(f"{name} is required for action={action}")


# Actions that call one Home Assistant intent with the plain target
_TARGET_INTENTS = {
    "turn_on": "HassTurnOn",
    "turn_off": "HassTurnOff",
    "media_pause": "HassMediaPause",
    "media_unpause": "HassMediaUnpause",
    "media_next": "HassMediaNext",
    "media_previous": "HassMediaPrevious",
    "cancel_timers": "HassCancelAllTimers",
}


async def _set_light(_action: str, request: ControlRequest) -> Any:
    set_args = request.target()
    if request.brightness is not None:
        set_args["brightness"] = request.brightness
    if request.color_temp is not None:
        set_args["temperature"] = request.color_temp
    if request.color is not None:
        set_args["color"] = request.color
    return await call_tool("HassLightSet", set_args)


async def _set_temperature(action: str, request: ControlRequest) -> Any:
    _require(request.temperature, "temperature", action)
    return await call_tool(
        "HassClimateSetTemperature",
        {**request.target(), "temperature": request.temperature},
    )


async def _step_temperature(action: str, request: ControlRequest) -> Any:
    target = request.target()
    live_context = await call_tool("GetLiveContext", {**target, "domain": "climate"})
    current_temp = _extract_current_temperature(live_context)
    if current_temp is None:
        raise ControlArgumentError(
            f"Could not read current temperature for {request.device_name or request.area}"
        )
    step = _TEMPERATURE_STEP if action == "increase_temperature" else -_TEMPERATURE_STEP
    request.temperature = current_temp + step
    return await call_tool(
        "HassClimateSetTemperature", {**target, "temperature": request.temperature}
    )


async def _set_volume(action: str, request: ControlRequest) -> Any:
    _require(request.volume_percent, "volume_percent", action)
    return await call_tool(
        "HassSetVolume", {**request.target(), "volume_level": request.volume_percent}
    )


async def _volume_step(action: str, request: ControlRequest) -> Any:
    _require(request.volume_step, "volume_step", action)
    return await call_tool(
        "HassSetVolumeRelative",
        {**request.target(), "volume_step": request.volume_step},
    )


async def _mute(action: str, request: ControlRequest) -> Any:
    if action == "media_mute":
        return await call_tool(
            "HassMediaPlayerMute", {**request.target(), "is_volume_muted": "true"}
        )
    return await call_tool(
        "HassMediaPlayerUnmute", {**request.target(), "is_volume_muted": "false"}
    )


async def _media_search_and_play(action: str, request: ControlRequest) -> Any:
    _require(request.search_query, "search_query", action)
    assert request.search_query is not None
    return await _search_and_play(
        request.target(), request.search_query, request.media_class
    )


async def _broadcast(action: str, request: ControlRequest) -> Any:
    _require(request.message, "message", action)
    return await call_tool("HassBroadcast", {"message": request.message})


async def _activate_scene(action: str, request: ControlRequest) -> Any:
    _require(request.device_name, "device_name", action)
    return await call_tool("HassTurnOn", request.target())


async def _vacuum(action: str, request: ControlRequest) -> Any:
    # Vacuums have no area and no TURN_ON feature, so HassTurnOn can't start them. # ARCHIE-244
    vacuum_args: dict[str, Any] = {"domain": ["vacuum"]}
    if request.device_name:
        vacuum_args["name"] = request.device_name
    intent = "HassVacuumStart" if action == "vacuum_start" else "HassVacuumReturnToBase"
    return await call_tool(intent, vacuum_args)


async def _vacuum_clean_area(action: str, request: ControlRequest) -> Any:
    _require(request.area, "area", action)
    clean_args: dict[str, Any] = {"area": request.area}
    if request.device_name:
        clean_args["name"] = request.device_name
    return await call_tool("HassVacuumCleanArea", clean_args)


ControlHandler = Callable[[str, ControlRequest], Awaitable[Any]]

_ACTIONS: dict[str, ControlHandler] = {
    "set_light": _set_light,
    "set_temperature": _set_temperature,
    "increase_temperature": _step_temperature,
    "decrease_temperature": _step_temperature,
    "media_set_volume": _set_volume,
    "media_volume_step": _volume_step,
    "media_mute": _mute,
    "media_unmute": _mute,
    "media_search_and_play": _media_search_and_play,
    "broadcast": _broadcast,
    "activate_scene": _activate_scene,
    "vacuum_start": _vacuum,
    "vacuum_return_to_base": _vacuum,
    "vacuum_clean_area": _vacuum_clean_area,
}


async def _run_action(action: str, request: ControlRequest) -> Any:
    if action in _TARGET_INTENTS:
        return await call_tool(_TARGET_INTENTS[action], request.target())
    handler = _ACTIONS.get(action)
    if handler is None:
        raise ControlArgumentError(f"Unknown action: {action}")
    return await handler(action, request)


async def smarthome_control_tool(
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
    robot vacuum, and whole-home voice broadcasts. This tool only CHANGES state — it cannot answer
    "is the light on"/"what's the temperature" or similar questions, use smarthome_status_tool for those.
    IMPORTANT: Users don't know numeric values (brightness percentages, Kelvin temperatures, etc.).
    Choose appropriate values based on context and NEVER mention raw numbers to the user.

    Args:
        action: One of: turn_on, turn_off, set_light, set_temperature, increase_temperature,
            decrease_temperature, media_pause, media_unpause, media_next, media_previous,
            media_set_volume, media_volume_step, media_mute, media_unmute, media_search_and_play,
            broadcast, cancel_timers, activate_scene, vacuum_start, vacuum_return_to_base,
            vacuum_clean_area
        device_name: Entity name/alias to target, e.g. 'Floor Lamp Living Room', 'Living Room Thermostat'.
            For 'activate_scene', the scene name, e.g. 'Jazz Bar', 'Рождество'.
            Optional for vacuum_* actions (the only robot vacuum is used by default)
        area: Area/room name to target instead of/in addition to name, e.g. 'Гостиная'.
            Required for 'vacuum_clean_area' (the room to clean)
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

    request = ControlRequest(
        device_name=device_name,
        area=area,
        floor=floor,
        brightness=brightness,
        color_temp=color_temp,
        color=color,
        temperature=temperature,
        volume_percent=volume_percent,
        volume_step=volume_step,
        search_query=search_query,
        media_class=media_class,
        message=message,
    )
    try:
        ha_result = await _run_action(action, request)
    except ControlArgumentError as e:
        return {"status": "error", "message": str(e)}
    except HomeAssistantMCPError as e:
        logger.error(f"smarthome_control_error_001: \033[31m{e}\033[0m")
        return {
            "status": "error",
            "message": f"Home Assistant error: {e}",
        }
    temperature = request.temperature

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
