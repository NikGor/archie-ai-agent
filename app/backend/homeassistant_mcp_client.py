"""JSON-RPC client for the Home Assistant MCP server (smarthome control)."""

import itertools
import json
import logging
from typing import Any
import httpx
from app.config import settings


logger = logging.getLogger(__name__)

_request_id = itertools.count(1)
_tool_names: set[str] = set()


class HomeAssistantMCPError(Exception):
    """Raised when the Home Assistant MCP server is unreachable or returns an error."""


async def _send(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Send one JSON-RPC request to the MCP server and return its `result` object."""
    if not settings.homeassistant_mcp_token:
        raise HomeAssistantMCPError("HOMEASSISTANT_MCP_TOKEN not configured")

    payload = {
        "jsonrpc": "2.0",
        "id": next(_request_id),
        "method": method,
        "params": params,
    }
    headers = {
        "Authorization": f"Bearer {settings.homeassistant_mcp_token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                settings.homeassistant_mcp_url,
                json=payload,
                headers=headers,
                timeout=15.0,
            )
    except httpx.TimeoutException as e:
        logger.error("ha_mcp_error_001: \033[31mRequest timeout\033[0m")
        raise HomeAssistantMCPError("Home Assistant MCP request timed out") from e
    except httpx.HTTPError as e:
        logger.error(f"ha_mcp_error_002: \033[31m{e}\033[0m")
        raise HomeAssistantMCPError(f"Home Assistant MCP request failed: {e}") from e

    if response.status_code != 200:
        logger.error(
            f"ha_mcp_error_003: HTTP \033[31m{response.status_code}\033[0m: {response.text}"
        )
        raise HomeAssistantMCPError(
            f"Home Assistant MCP returned HTTP {response.status_code}"
        )

    body = response.json()
    if "error" in body:
        logger.error(f"ha_mcp_error_004: \033[31m{body['error']}\033[0m")
        raise HomeAssistantMCPError(str(body["error"]))
    return body.get("result", {})


async def _resolve_tool_name(name: str) -> str:
    """
    Map a bare tool name to the name the server exposes.

    Newer Home Assistant versions prefix tools with their domain
    (`homeassistant__GetLiveContext`, `intent__HassTurnOn`); older ones don't. # ARCHIE-214
    """
    for refresh in (False, True):
        if refresh or not _tool_names:
            result = await _send("tools/list", {})
            _tool_names.clear()
            _tool_names.update(tool["name"] for tool in result.get("tools", []))
        if name in _tool_names:
            return name
        matches = [tool for tool in _tool_names if tool.endswith(f"__{name}")]
        if matches:
            return matches[0]
    return name


async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Call a tool exposed by the Home Assistant MCP server (`tools/call`).

    Args:
        name: MCP tool name with or without the domain prefix,
            e.g. "HassTurnOn", "GetLiveContext", "media_player__search_media".
        arguments: Tool arguments per its inputSchema (see /api/mcp tools/list).

    Returns:
        The parsed `{"success": ..., "result": ...}` payload returned by the tool.

    Raises:
        HomeAssistantMCPError: on missing token, network failure, or a tool-level error.
    """
    name = await _resolve_tool_name(name)
    logger.info(
        f"ha_mcp_001: Calling tool \033[36m{name}\033[0m with args \033[33m{arguments}\033[0m"
    )
    result = await _send("tools/call", {"name": name, "arguments": arguments})

    content = result.get("content", [])
    text = content[0].get("text") if content else None
    if text is None:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"success": not result.get("isError", False), "result": text}

    if result.get("isError") or not parsed.get("success", True):
        logger.error(f"ha_mcp_error_005: \033[31m{parsed}\033[0m")
        raise HomeAssistantMCPError(
            str(
                parsed.get("result")
                or parsed.get("message")
                or "Unknown MCP tool error"
            )
        )

    logger.info(f"ha_mcp_002: Tool \033[36m{name}\033[0m succeeded")
    return parsed
