"""Tests for the tool that runs the main agent from a schedule."""

from unittest.mock import AsyncMock, MagicMock, patch
from app.config import settings
from app.agents.chat.tools import agent_task_tool as module
from app.agents.chat.tools.agent_task_tool import agent_task_tool


def _message(text: str | None) -> MagicMock:
    message = MagicMock()
    message.content.text = text
    return message


async def test_prompt_runs_a_plain_chat_turn_with_more_iterations():
    handle = AsyncMock(return_value=_message("Planned 2 matches"))
    with patch.object(module, "handle_chat", handle):
        result = await agent_task_tool("Plan today football", user_name="Niko")

    assert result == {"status": "success", "reply": "Planned 2 matches"}
    request = handle.await_args.args[0]
    assert (request.input, request.user_name, request.response_format) == (
        "Plan today football",
        "Niko",
        "plain",
    )
    assert request.conversation_id.startswith("scheduled-")
    assert handle.await_args.kwargs == {
        "max_command_iterations": settings.agent_task_max_iterations
    }


async def test_each_run_gets_its_own_conversation():
    handle = AsyncMock(return_value=_message("ok"))
    with patch.object(module, "handle_chat", handle):
        await agent_task_tool("a")
        await agent_task_tool("a")

    first, second = (call.args[0].conversation_id for call in handle.await_args_list)
    assert first != second


async def test_empty_prompt_is_rejected():
    handle = AsyncMock()
    with patch.object(module, "handle_chat", handle):
        result = await agent_task_tool("  ")

    assert result["status"] == "error"
    handle.assert_not_awaited()


async def test_task_cannot_start_another_task():
    inner_results = []

    async def outer_handle(*_args, **_kwargs):
        inner_results.append(await agent_task_tool("inner"))
        return _message("done")

    with patch.object(module, "handle_chat", AsyncMock(side_effect=outer_handle)):
        result = await agent_task_tool("outer")

    assert result["status"] == "success"
    assert inner_results[0]["status"] == "error"
    assert "itself" in inner_results[0]["message"]


async def test_failed_run_returns_error_and_allows_next_run():
    handle = AsyncMock(side_effect=[RuntimeError("boom"), _message("ok")])
    with patch.object(module, "handle_chat", handle):
        failed = await agent_task_tool("a")
        next_run = await agent_task_tool("a")

    assert failed == {"status": "error", "message": "RuntimeError: boom"}
    assert next_run["status"] == "success"


async def test_reply_is_clipped_and_empty_text_is_allowed():
    long_handle = AsyncMock(return_value=_message("x" * 900))
    none_handle = AsyncMock(return_value=_message(None))
    with patch.object(module, "handle_chat", long_handle):
        long_result = await agent_task_tool("a")
    with patch.object(module, "handle_chat", none_handle):
        none_result = await agent_task_tool("a")

    assert len(long_result["reply"]) == 500
    assert none_result == {"status": "success", "reply": ""}
