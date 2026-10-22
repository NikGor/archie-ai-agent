"""Tool that shows short messages on the AWTRIX NG LED display (Ulanzi TC001).

Talks to the clock over its HTTP API. Meant to be scheduled by cron_tool or
called by the swarm display agent for proactive notices (a match starts, time
to air the room), but can also be called directly.
"""

import logging
import re
from typing import Any, Literal
import httpx
from app.config import settings


logger = logging.getLogger(__name__)

_APP_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
_BUILTIN_APPS = frozenset({"Time", "Date", "Temperature", "Humidity", "Battery"})


def _build_payload(
    text: str | None,
    icon: str | None,
    color: str | None,
    duration_seconds: int | None,
    duration_key: str,
) -> dict[str, Any]:
    if not text:
        raise ValueError("text is required")
    payload: dict[str, Any] = {"text": text}
    if icon:
        payload["icon"] = icon
    if color:
        if not _COLOR_PATTERN.match(color):
            raise ValueError("color must look like #RRGGBB")
        payload["textColor"] = color
    if duration_seconds:
        payload[duration_key] = duration_seconds * 1000
    return payload


def _validate_app_name(name: str | None) -> str:
    if not name or not _APP_NAME_PATTERN.match(name):
        raise ValueError("name must be 1-32 characters: A-Z a-z 0-9 _ -")
    if name in _BUILTIN_APPS:
        raise ValueError(f"name must not match a built-in app: {name}")
    return name


async def _send(method: str, path: str, payload: dict[str, Any] | None) -> int:
    url = f"http://{settings.awtrix_host}/api/v1{path}"
    async with httpx.AsyncClient(timeout=settings.awtrix_timeout) as client:
        response = await client.request(method, url, json=payload)
    return response.status_code


async def awtrix_tool(
    action: Literal["notify", "push", "remove"],
    text: str | None = None,
    name: str | None = None,
    icon: str | None = None,
    color: str | None = None,
    duration_seconds: int | None = None,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Show a very short message on the LED display in the user's home.

    Use it for proactive notices, usually scheduled through cron_tool — for
    example, a football match starting soon. The display is 32x8 pixels: keep
    the text under about 20 characters, numbers matter more than words.
    notify shows the text once on top of everything. push adds a named app to
    the display loop until removed. remove deletes a pushed app.

    Example (scheduled via cron_tool):
        tool_name="awtrix_tool",
        arguments_json='{"action":"notify","text":"Bayern 20:30","color":"#00FF00"}'

    Args:
        action: notify (one-time notice), push (named app in the loop) or remove
        text: Short text to show; required for notify and push
        name: App name for push and remove: 1-32 characters A-Z a-z 0-9 _ -
        icon: Optional icon ID already uploaded to the display
        color: Optional text color as #RRGGBB
        duration_seconds: notify - how long to show it; push - when the app removes itself

    Returns:
        dict[str, Any]: Delivery status
    """
    if demo_mode:
        return {"status": "demo", "message": f"[DEMO] AWTRIX {action} would be sent"}
    if not settings.awtrix_host:
        logger.error("awtrix_tool_error_001: AWTRIX_HOST is not set")
        return {"status": "error", "message": "AWTRIX_HOST is not configured"}

    try:
        if action == "notify":
            payload = _build_payload(text, icon, color, duration_seconds, "durationMs")
            code = await _send("POST", "/notifications", payload)
        elif action == "push":
            payload = _build_payload(text, icon, color, duration_seconds, "lifetimeMs")
            code = await _send(
                "PUT", f"/apps/pushed/{_validate_app_name(name)}", payload
            )
        else:
            code = await _send("DELETE", f"/apps/{_validate_app_name(name)}", None)
    except ValueError as exc:
        logger.warning(f"awtrix_tool_warning_001: Invalid arguments: {exc}")
        return {"status": "error", "message": str(exc)}
    except httpx.HTTPError as exc:
        logger.error(f"awtrix_tool_error_002: Display unreachable: {exc!r}")
        return {"status": "error", "message": f"Display unreachable: {exc!r}"}

    # Removing an app that is already gone is a success: the goal is reached.
    if code == httpx.codes.NOT_FOUND and action == "remove":
        code = httpx.codes.OK
    if code >= httpx.codes.BAD_REQUEST:
        logger.error(f"awtrix_tool_error_003: Display answered {code} to {action}")
        return {"status": "error", "message": f"Display answered {code}"}

    logger.info(f"awtrix_tool_001: \033[36m{action}\033[0m sent ({code})")
    return {"status": "success", "action": action}
