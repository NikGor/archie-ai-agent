"""Tests for the AWTRIX LED display tool."""

from unittest.mock import AsyncMock, patch
import httpx
import pytest
from app.tools.awtrix_tool import awtrix_tool


@pytest.fixture(autouse=True)
def awtrix_host():
    with patch("app.tools.awtrix_tool.settings") as mock_settings:
        mock_settings.awtrix_host = "clock.local"
        mock_settings.awtrix_timeout = 1.0
        yield mock_settings


def _patch_send(code: int = 200) -> AsyncMock:
    return patch("app.tools.awtrix_tool._send", new=AsyncMock(return_value=code))


async def test_notify_posts_payload_with_color_and_duration():
    with _patch_send() as send:
        result = await awtrix_tool(
            "notify", text="Bayern 20:30", color="#00FF00", duration_seconds=6
        )

    assert result == {"status": "success", "action": "notify"}
    send.assert_awaited_once_with(
        "POST",
        "/notifications",
        {"text": "Bayern 20:30", "textColor": "#00FF00", "durationMs": 6000},
    )


async def test_push_puts_named_app_with_lifetime():
    with _patch_send() as send:
        result = await awtrix_tool(
            "push", name="match_live", text="1:0", icon="ball", duration_seconds=60
        )

    assert result["status"] == "success"
    send.assert_awaited_once_with(
        "PUT",
        "/apps/pushed/match_live",
        {"text": "1:0", "icon": "ball", "lifetimeMs": 60000},
    )


async def test_remove_deletes_app():
    with _patch_send() as send:
        result = await awtrix_tool("remove", name="match_live")

    assert result["status"] == "success"
    send.assert_awaited_once_with("DELETE", "/apps/match_live", None)


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
