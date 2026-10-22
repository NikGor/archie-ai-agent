"""Tool that shows short messages and icons on the AWTRIX NG LED display (Ulanzi TC001).

Talks to the clock over its HTTP API. Meant to be scheduled by cron_tool or
called by the swarm display agent for proactive notices (a match starts, time
to air the room), but can also be called directly.

Icons come from two places: the library already stored on the display
(/ICONS, listed by `list_icons`) or a picture the agent draws itself as an
8x8 pixel map (`icon_pixels`). A drawn icon is sent inline for one message or
saved to the library with `save_icon`.
"""

import base64
import io
import logging
import re
from typing import Any, Literal
import httpx
from PIL import Image
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)
from app.config import settings


logger = logging.getLogger(__name__)

_APP_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
_BUILTIN_APPS = frozenset({"Time", "Date", "Temperature", "Humidity", "Battery"})

# The display font has no emoji: each one would show as "?".
_EMOJI_PATTERN = re.compile(
    "[\U0001f000-\U0001faff\u2300-\u23ff\u2600-\u27bf\u2b00-\u2bff"
    "\ufe0f\u200d\u20e3\U000e0020-\U000e007f]+"
)

_ICON_SIZE = 8
# ARCHIE-233: a notice is shown 3 times so the user does not miss it; the
# display queues notifications and plays them one after the other.
_NOTIFY_REPEAT = 3
# The clock sits on weak Wi-Fi and sometimes drops a request: try twice.
_SEND_ATTEMPTS = 2
_RETRY_DELAY_SECONDS = 1.0
_ICONS_DIR = "/ICONS"
_ICON_EXTENSIONS = (".gif", ".jpg")
# '.' is off: a dark pixel of the LED matrix. Bright values read well on LEDs.
_PALETTE: dict[str, tuple[int, int, int]] = {
    ".": (0, 0, 0),
    "R": (255, 40, 40),
    "O": (255, 140, 0),
    "Y": (255, 220, 0),
    "G": (0, 200, 60),
    "C": (0, 200, 220),
    "B": (40, 80, 255),
    "P": (150, 60, 220),
    "M": (255, 80, 180),
    "N": (140, 80, 30),
    "W": (255, 255, 255),
    "K": (110, 110, 110),
}


def _build_payload(
    text: str | None,
    icon: str | None,
    color: str | None,
    duration_seconds: int | None,
    duration_key: str,
) -> dict[str, Any]:
    text = " ".join(_EMOJI_PATTERN.sub(" ", text or "").split())
    if not text:
        raise ValueError("text is required (emoji are not shown on the display)")
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


def _validate_icon_name(name: str | None) -> str:
    if not name or not _APP_NAME_PATTERN.match(name):
        raise ValueError("name must be 1-32 characters: A-Z a-z 0-9 _ -")
    return name


def _render_icon(pixels: str) -> bytes:
    """Draw an 8x8 GIF from rows of palette letters separated by '/'."""
    rows = pixels.strip().split("/")
    if len(rows) != _ICON_SIZE or any(len(row) != _ICON_SIZE for row in rows):
        raise ValueError("icon_pixels must be 8 rows of 8 characters separated by '/'")
    unknown = sorted({char for row in rows for char in row} - set(_PALETTE))
    if unknown:
        raise ValueError(f"icon_pixels has unknown colors: {''.join(unknown)}")
    image = Image.new("RGB", (_ICON_SIZE, _ICON_SIZE))
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            image.putpixel((x, y), _PALETTE[char])
    buffer = io.BytesIO()
    image.save(buffer, format="GIF")
    return buffer.getvalue()


def _icon_data_url(pixels: str) -> str:
    encoded = base64.b64encode(_render_icon(pixels)).decode()
    return f"data:image/gif;base64,{encoded}"


def _resolve_icon(icon: str | None, icon_pixels: str | None) -> str | None:
    if icon and icon_pixels:
        raise ValueError("give either icon or icon_pixels, not both")
    return _icon_data_url(icon_pixels) if icon_pixels else icon


def _icon_ids(listing: dict[str, Any]) -> list[str]:
    names = [entry["name"] for entry in listing.get("files", [])]
    return sorted(
        name.rsplit(".", 1)[0] for name in names if name.endswith(_ICON_EXTENSIONS)
    )


async def _send(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
) -> httpx.Response:
    url = f"http://{settings.awtrix_host}/api/v1{path}"
    async with httpx.AsyncClient(timeout=settings.awtrix_timeout) as client:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(_SEND_ATTEMPTS),
            wait=wait_fixed(_RETRY_DELAY_SECONDS),
            retry=retry_if_exception_type(httpx.TransportError),
            before_sleep=lambda state: logger.warning(
                f"awtrix_tool_warning_002: Retrying after {state.outcome.exception()!r}"
            ),
            reraise=True,
        ):
            with attempt:
                return await client.request(method, url, json=payload, files=files)
    raise AssertionError("unreachable: AsyncRetrying always yields an attempt")


async def _run_action(
    action: str,
    text: str | None,
    name: str | None,
    icon: str | None,
    icon_pixels: str | None,
    color: str | None,
    duration_seconds: int | None,
    repeat: int,
) -> tuple[httpx.Response, dict[str, Any]]:
    """Send one action to the display; return its response and extra result fields."""
    if action == "notify":
        payload = _build_payload(
            text,
            _resolve_icon(icon, icon_pixels),
            color,
            duration_seconds,
            "durationMs",
        )
        for _ in range(max(1, repeat)):
            response = await _send("POST", "/notifications", payload)
            if not response.is_success:
                break
        return response, {}
    if action == "push":
        payload = _build_payload(
            text,
            _resolve_icon(icon, icon_pixels),
            color,
            duration_seconds,
            "lifetimeMs",
        )
        app_name = _validate_app_name(name)
        return await _send("PUT", f"/apps/pushed/{app_name}", payload), {}
    if action == "remove":
        return await _send("DELETE", f"/apps/{_validate_app_name(name)}"), {}
    if action == "list_icons":
        response = await _send("GET", f"/files?dir={_ICONS_DIR}")
        return response, {
            "icons": _icon_ids(response.json())
        } if response.is_success else {}
    icon_id = _validate_icon_name(name)
    if not icon_pixels:
        raise ValueError("icon_pixels is required for save_icon")
    gif = (f"{icon_id}.gif", _render_icon(icon_pixels), "image/gif")
    response = await _send("POST", f"/files?dir={_ICONS_DIR}", files={"file": gif})
    return response, {"icon": icon_id}


async def awtrix_tool(
    action: Literal["notify", "push", "remove", "list_icons", "save_icon"],
    text: str | None = None,
    name: str | None = None,
    icon: str | None = None,
    icon_pixels: str | None = None,
    color: str | None = None,
    duration_seconds: int | None = None,
    repeat: int = _NOTIFY_REPEAT,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Show a very short message with an optional icon on the LED display at home.

    Use it for proactive notices, usually scheduled through cron_tool, for
    example a football match starting soon. The display is 32x8 pixels: keep
    the text under about 20 characters, numbers matter more than words. It
    cannot show emoji: they are removed from the text, so pass an icon instead
    (see the awtrix skill).
    notify shows the text on top of everything, 3 times in a row by default:
    call it once, do not repeat the call. push adds a named app to
    the display loop until removed. remove deletes a pushed app.

    Icons (8x8). First call list_icons and reuse a fitting icon by its ID in
    icon. If none fits, draw one in icon_pixels: 8 rows of 8 characters joined
    by '/', each character a color: . off, R red, O orange, Y yellow, G green,
    C cyan, B blue, P purple, M pink, N brown, W white, K grey. Draw simple
    bold shapes with few colors. A drawn icon is used for this message only;
    save_icon stores it under name so later messages can reuse it by ID.

    Example heart: icon_pixels="......../.RR..RR./RRRRRRRR/RRRRRRRR/.RRRRRR./..RRRR../...RR.../........"
    Example (scheduled via cron_tool):
        tool_name="awtrix_tool",
        arguments_json='{"action":"notify","text":"Bayern 20:30","icon":"ball"}'

    Args:
        action: notify, push, remove, list_icons (IDs stored on the display) or save_icon
        text: Short text to show; required for notify and push
        name: App name for push and remove, icon ID for save_icon: 1-32 characters A-Z a-z 0-9 _ -
        icon: ID of an icon stored on the display
        icon_pixels: Drawn 8x8 icon, rows joined by '/'; required for save_icon
        color: Optional text color as #RRGGBB
        duration_seconds: notify - how long to show it; push - when the app removes itself
        repeat: notify only - how many times in a row the message is shown (default 3)

    Returns:
        dict[str, Any]: Delivery status; list_icons also returns the icon IDs
    """
    if demo_mode:
        return {"status": "demo", "message": f"[DEMO] AWTRIX {action} would be sent"}
    if not settings.awtrix_host:
        logger.error("awtrix_tool_error_001: AWTRIX_HOST is not set")
        return {"status": "error", "message": "AWTRIX_HOST is not configured"}

    try:
        response, extra = await _run_action(
            action, text, name, icon, icon_pixels, color, duration_seconds, repeat
        )
    except ValueError as exc:
        logger.warning(f"awtrix_tool_warning_001: Invalid arguments: {exc}")
        return {"status": "error", "message": str(exc)}
    except httpx.HTTPError as exc:
        logger.error(f"awtrix_tool_error_002: Display unreachable: {exc!r}")
        return {"status": "error", "message": f"Display unreachable: {exc!r}"}
    result: dict[str, Any] = {"status": "success", "action": action, **extra}

    # Removing an app that is already gone is a success: the goal is reached.
    gone = response.status_code == httpx.codes.NOT_FOUND and action == "remove"
    if not (response.is_success or gone):
        logger.error(
            f"awtrix_tool_error_003: Display answered {response.status_code} to {action}"
        )
        return {
            "status": "error",
            "message": f"Display answered {response.status_code}",
        }

    logger.info(
        f"awtrix_tool_001: \033[36m{action}\033[0m sent ({response.status_code})"
    )
    return result
