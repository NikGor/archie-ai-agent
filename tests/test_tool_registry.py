"""Tool registry and chat tool access (ARCHIE-292; replaces the ToolFactory
characterization tests of ARCHIE-164)."""

from unittest.mock import patch
import app.agents.swarm.tools  # noqa: F401  registers the swarm tool group
from app.agents.chat.tool_access import chat_tool_names
from app.platform.tools.models import ToolContext
import pytest
from pydantic import ValidationError
from app.platform.tools.registry import READ, ToolGroup, ToolRegistry
from app.tools.catalog import tool_registry


def _registry(**functions) -> ToolRegistry:
    registry = ToolRegistry(
        [
            ToolGroup(
                name="t",
                tools=dict.fromkeys(functions, "x"),
                capabilities=dict.fromkeys(functions, READ),
            )
        ]
    )
    registry._functions = dict(functions)
    return registry


def test_dashboard_and_widget_see_only_smarthome():
    smarthome = list(tool_registry.groups["smarthome"].tools)
    assert chat_tool_names("dashboard") == smarthome
    assert chat_tool_names("widget") == smarthome


def test_plain_sees_every_core_group_but_not_the_swarm_group():
    names = chat_tool_names("plain")
    core = tool_registry.names(include_exclusive=False)
    assert names == core
    assert not set(tool_registry.groups["swarm"].tools) & set(names)


def test_swarm_group_is_exclusive_and_registered_by_the_swarm():
    group = tool_registry.groups["swarm"]
    assert group.exclusive
    assert group.tools["alert_chat_tool"] == "app.agents.swarm.tools.alert_chat_tool"


def test_schemas_follow_registry_order_and_skip_unwanted():
    names = list(reversed(tool_registry.groups["smarthome"].tools))
    schemas = tool_registry.schemas("gpt-4.1", names)
    assert [s["name"] for s in schemas] == list(tool_registry.groups["smarthome"].tools)


def test_schemas_use_the_provider_format():
    openai = tool_registry.schemas("gpt-4.1", ["smarthome_status_tool"])[0]
    assert "required" in openai["parameters"]
    with patch(
        "app.platform.tools.registry.get_provider_for_model", return_value="openrouter"
    ):
        oss = tool_registry.schemas("gpt-4.1", ["smarthome_status_tool"])[0]
    assert "required" not in oss["parameters"]


async def test_unknown_tool_returns_error():
    assert await tool_registry.execute("nonexistent_tool", {}) == {
        "error": "Unknown tool: nonexistent_tool"
    }


async def test_demo_mode_comes_from_the_call_context():
    """ARCHIE-171: demo mode is per call, not per factory."""

    async def fake_tool(demo_mode: bool = False) -> dict:
        return {"demo_mode_seen": demo_mode}

    registry = _registry(fake_tool=fake_tool)
    demo = ToolContext(user_timezone="UTC", demo_mode=True)
    assert await registry.execute("fake_tool", {}, demo) == {"demo_mode_seen": True}
    assert await registry.execute("fake_tool", {}) == {"demo_mode_seen": False}


async def test_demo_mode_is_not_injected_when_absent_from_signature():
    async def fake_tool() -> dict:
        return {"ok": True}

    registry = _registry(fake_tool=fake_tool)
    demo = ToolContext(user_timezone="UTC", demo_mode=True)
    assert await registry.execute("fake_tool", {}, demo) == {"ok": True}


async def test_tool_exception_becomes_error_dict():
    async def failing_tool() -> dict:
        raise RuntimeError("boom")

    result = await _registry(failing_tool=failing_tool).execute("failing_tool", {})
    assert "boom" in result["error"]


def test_every_registered_tool_has_a_capability():
    """ARCHIE-320: policies read effects from the registry, never guess them."""
    for name in tool_registry.names():
        assert tool_registry.capability(name).effect in ("read", "write")


def test_group_without_capability_metadata_is_rejected():
    with pytest.raises(ValidationError):
        ToolGroup(name="t", tools={"a": "x"}, capabilities={})


def test_read_only_follows_capabilities_and_read_actions():
    assert tool_registry.is_read_only("smarthome_status_tool", {})
    assert not tool_registry.is_read_only("smarthome_control_tool", {})
    assert tool_registry.is_read_only("events_tool", {"action": "list"})
    assert not tool_registry.is_read_only("events_tool", {"action": "create"})
    assert tool_registry.is_read_only("cron_tool", {"action": "get"})
    assert not tool_registry.is_read_only("unknown_tool", {})
