"""Chat delegation to dynamic agents (ARCHIE-324)."""

import pytest
import app.platform.runtime.delegation as delegation
from app.agents.chat.tools.delegate_agent_tool import delegate_agent_tool
from app.agents.delegation import chat_policy, step_text
from app.platform.runtime.delegation import DelegationCaller, delegation_scope
from app.platform.runtime.runs import RunEvent
from app.platform.runtime.spawn import SpawnContext
from app.platform.runtime.spec import SpecValidationError, validate_spec
from app.tools.catalog import tool_registry
from app.agents.chat.tool_access import chat_tool_names


def caller(tools=None) -> DelegationCaller:
    return DelegationCaller(
        caller=SpawnContext(source="chat", parent_run_id="chat-1"),
        tools=tuple(tools if tools is not None else chat_tool_names("plain")),
    )


def spec(tools) -> dict:
    return {
        "identity": {"name": "a", "role": "r", "purpose": "p"},
        "objective": {"task": "t", "completion_criteria": ["c"]},
        "model": {"profile": "FAST"},
        "tools": tools,
        "output": {"schema_name": "findings_v1"},
    }


async def test_tool_outside_a_trusted_caller_is_refused(monkeypatch):
    monkeypatch.setattr(delegation, "_handler", None)
    result = await delegate_agent_tool("Compare cafes")
    assert result["status"] == "error" and "trusted caller" in result["message"]


async def test_handler_gets_the_trusted_caller(monkeypatch):
    seen = []

    async def handler(task, context, who):
        seen.append((task, who.caller.source, who.tools))
        return {"state": "completed"}

    monkeypatch.setattr(delegation, "_handler", handler)
    with delegation_scope(caller(["google_search_tool"])):
        assert (await delegate_agent_tool("Compare cafes"))["state"] == "completed"
    assert seen == [("Compare cafes", "chat", ("google_search_tool",))]


def test_chat_policy_has_no_recursion_and_no_foreign_tools():
    policy = chat_policy(caller())
    assert (
        not {"delegate_agent_tool", "agent_task_tool", "spawn_agent_tool"}
        & policy.tools
    )
    for tool in ("delegate_agent_tool", "agent_task_tool"):
        with pytest.raises(SpecValidationError):
            validate_spec(spec([tool]), policy, tool_registry)
    with pytest.raises(SpecValidationError, match="not allowed"):
        validate_spec(
            spec(["smarthome_status_tool"]),
            chat_policy(caller(["google_search_tool"])),
            tool_registry,
        )
    narrow = chat_policy(caller(chat_tool_names("dashboard")))
    assert "google_search_tool" not in narrow.tools


def test_delegated_write_tools_are_only_proposals():
    policy = chat_policy(caller())
    assert policy.may_request_approval
    validated = validate_spec(
        {
            **spec(["notes_tool"]),
            "permissions": {
                "mode": "propose_actions",
                "approval_tools": ["notes_tool"],
            },
        },
        policy,
        tool_registry,
    )
    assert validated.spec.permissions.mode == "propose_actions"


def test_delegate_tool_is_registered_for_text_formats_only():
    assert "delegate_agent_tool" in chat_tool_names("plain")
    assert "delegate_agent_tool" not in chat_tool_names("dashboard")


def test_step_text_reports_tools_to_the_status_stream():
    assert (
        step_text(
            RunEvent(kind="tool_invoked", at=0, detail={"tool": "google_search_tool"})
        )
        == "Agent: google_search_tool"
    )
    assert step_text(RunEvent(kind="task_created", at=0)) is None
