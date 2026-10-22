"""Tests for the AWTRIX LED display tool."""

import base64
import io
from unittest.mock import AsyncMock, patch
import httpx
import pytest
from PIL import Image
from app.tools.awtrix_tool import awtrix_tool


HEART = "......../.RR..RR./RRRRRRRR/RRRRRRRR/.RRRRRR./..RRRR../...RR.../........"


@pytest.fixture(autouse=True)
def awtrix_host():
    with patch("app.tools.awtrix_tool.settings") as mock_settings:
        mock_settings.awtrix_host = "clock.local"
        mock_settings.awtrix_timeout = 1.0
        yield mock_settings


def _patch_send(code: int = 200, body: dict | None = None) -> AsyncMock:
    response = httpx.Response(code, json=body or {})
    return patch("app.tools.awtrix_tool._send", new=AsyncMock(return_value=response))


def _decode_data_url(url: str) -> Image.Image:
    prefix = "data:image/gif;base64,"
    assert url.startswith(prefix)
    return Image.open(io.BytesIO(base64.b64decode(url[len(prefix) :]))).convert("RGB")


async def test_notify_posts_payload_with_color_and_duration():
    with _patch_send() as send:
        result = await awtrix_tool(
            "notify",
            text="Bayern 20:30",
            icon="ball",
            color="#00FF00",
            duration_seconds=6,
        )

    assert result == {"status": "success", "action": "notify"}
    send.assert_awaited_once_with(
        "POST",
        "/notifications",
        {
            "text": "Bayern 20:30",
            "icon": "ball",
            "textColor": "#00FF00",
            "durationMs": 6000,
        },
    )


async def test_push_puts_named_app_with_lifetime():
    with _patch_send() as send:
        result = await awtrix_tool(
            "push", name="match_live", text="1:0", duration_seconds=60
        )

    assert result["status"] == "success"
    send.assert_awaited_once_with(
        "PUT", "/apps/pushed/match_live", {"text": "1:0", "lifetimeMs": 60000}
    )


async def test_remove_deletes_app():
    with _patch_send() as send:
        result = await awtrix_tool("remove", name="match_live")

    assert result["status"] == "success"
    send.assert_awaited_once_with("DELETE", "/apps/match_live")


async def test_remove_of_missing_app_is_success():
    with _patch_send(404):
        result = await awtrix_tool("remove", name="gone")

    assert result["status"] == "success"


@pytest.mark.parametrize("name", [None, "", "bad name", "a" * 33, "Time", "Battery"])
async def test_invalid_app_name_is_rejected_before_request(name):
    with _patch_send() as send:
        result = await awtrix_tool("push", name=name, text="x")

    assert result["status"] == "error"
    send.assert_not_awaited()


async def test_missing_text_and_bad_color_are_rejected():
    with _patch_send() as send:
        no_text = await awtrix_tool("notify")
        bad_color = await awtrix_tool("notify", text="x", color="green")

    assert no_text["status"] == "error"
    assert bad_color["status"] == "error"
    send.assert_not_awaited()


async def test_drawn_icon_is_sent_inline_as_gif():
    with _patch_send() as send:
        result = await awtrix_tool("notify", text="Love", icon_pixels=HEART)

    assert result["status"] == "success"
    icon = send.await_args.args[2]["icon"]
    image = _decode_data_url(icon)
    assert image.size == (8, 8)
    assert image.getpixel((0, 1)) == (0, 0, 0)
    assert image.getpixel((1, 1)) == (255, 40, 40)


@pytest.mark.parametrize(
    "pixels",
    [
        "R" * 8,
        "/".join(["RRRRRRR"] * 8),
        "/".join(["RRRRRRRR"] * 7),
        "/".join(["RRRRRRRX"] * 8),
    ],
)
async def test_bad_icon_pixels_are_rejected_before_request(pixels):
    with _patch_send() as send:
        result = await awtrix_tool("notify", text="x", icon_pixels=pixels)

    assert result["status"] == "error"
    send.assert_not_awaited()


async def test_icon_and_icon_pixels_together_are_rejected():
    with _patch_send() as send:
        result = await awtrix_tool("notify", text="x", icon="ball", icon_pixels=HEART)

    assert result["status"] == "error"
    send.assert_not_awaited()


async def test_list_icons_returns_ids_without_extensions():
    listing = {
        "files": [
            {"name": "ball.gif", "size": 60},
            {"name": "sun.jpg", "size": 90},
            {"name": "x.txt", "size": 1},
        ]
    }
    with _patch_send(body=listing) as send:
        result = await awtrix_tool("list_icons")

    assert result == {
        "status": "success",
        "action": "list_icons",
        "icons": ["ball", "sun"],
    }
    send.assert_awaited_once_with("GET", "/files?dir=/ICONS")


async def test_save_icon_uploads_gif_named_after_icon():
    with _patch_send() as send:
        result = await awtrix_tool("save_icon", name="heart", icon_pixels=HEART)

    assert result == {"status": "success", "action": "save_icon", "icon": "heart"}
    assert send.await_args.args[:2] == ("POST", "/files?dir=/ICONS")
    filename, content, mime = send.await_args.kwargs["files"]["file"]
    assert (filename, mime) == ("heart.gif", "image/gif")
    assert Image.open(io.BytesIO(content)).size == (8, 8)


async def test_save_icon_needs_name_and_pixels():
    with _patch_send() as send:
        no_pixels = await awtrix_tool("save_icon", name="heart")
        no_name = await awtrix_tool("save_icon", icon_pixels=HEART)

    assert no_pixels["status"] == "error"
    assert no_name["status"] == "error"
    send.assert_not_awaited()


async def test_unreachable_display_returns_error():
    with patch(
        "app.tools.awtrix_tool._send",
        new=AsyncMock(side_effect=httpx.ConnectTimeout("t")),
    ):
        result = await awtrix_tool("notify", text="x")

    assert result["status"] == "error"
    assert "unreachable" in result["message"]


async def test_display_error_status_is_reported():
    with _patch_send(500):
        result = await awtrix_tool("notify", text="x")

    assert result == {"status": "error", "message": "Display answered 500"}


async def test_missing_host_returns_error(awtrix_host):
    awtrix_host.awtrix_host = None
    with _patch_send() as send:
        result = await awtrix_tool("notify", text="x")

    assert result["status"] == "error"
    send.assert_not_awaited()


async def test_demo_mode_skips_network():
    with _patch_send() as send:
        result = await awtrix_tool("notify", text="x", demo_mode=True)

    assert result["status"] == "demo"
    send.assert_not_awaited()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Доброе утро ☀️", "Доброе утро"),
        ("🚀 Лови движ! 🚀", "Лови движ!"),
        ("Гол ⚽ 2:1 🏆", "Гол 2:1"),
        ("Bro 👍🏽 ok 👨‍👩‍👧", "Bro ok"),
        ("21,5° 5 € Привет", "21,5° 5 € Привет"),
    ],
)
async def test_emoji_are_removed_from_text(text, expected):
    with _patch_send() as send:
        result = await awtrix_tool("notify", text=text)

    assert result["status"] == "success"
    assert send.await_args.args[2]["text"] == expected


async def test_text_of_only_emoji_is_rejected():
    with _patch_send() as send:
        result = await awtrix_tool("notify", text="☀️🔥")

    assert result["status"] == "error"
    send.assert_not_awaited()


async def test_send_retries_once_after_a_timeout():
    ok = httpx.Response(200, json={})
    request = AsyncMock(side_effect=[httpx.ReadTimeout("t"), ok])
    with (
        patch("app.tools.awtrix_tool._RETRY_DELAY_SECONDS", 0),
        patch("httpx.AsyncClient.request", new=request),
    ):
        result = await awtrix_tool("notify", text="x")

    assert result["status"] == "success"
    assert request.await_count == 2


async def test_send_gives_up_after_two_attempts():
    request = AsyncMock(side_effect=httpx.ConnectTimeout("t"))
    with (
        patch("app.tools.awtrix_tool._RETRY_DELAY_SECONDS", 0),
        patch("httpx.AsyncClient.request", new=request),
    ):
        result = await awtrix_tool("notify", text="x")

    assert result["status"] == "error"
    assert "unreachable" in result["message"]
    assert request.await_count == 2
