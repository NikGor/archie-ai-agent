"""
Unit tests for tool_executor.execute_tool_call / execute_tool_calls.

ToolFactory is replaced with a lightweight mock — tests verify the
executor's contract: always returns ToolResult, never re-raises exceptions.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.orchestration_sgr import Parameter, ToolCallRequest
from app.models.tool_models import ToolResult
from app.utils.tool_executor import execute_tool_call, execute_tool_calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call(
    tool_name: str = "google_search_tool", query: str = "test"
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_name=tool_name,
        arguments=[Parameter(name="query", value=query)],
        missing_parameters=[],
        is_confirmed=True,
        reason="unit test",
    )


def _mock_factory(return_value=None, side_effect=None) -> MagicMock:
    factory = MagicMock()
    if side_effect is not None:
        factory.execute_tool = AsyncMock(side_effect=side_effect)
    else:
        factory.execute_tool = AsyncMock(return_value=return_value or {"data": "ok"})
    return factory


# ---------------------------------------------------------------------------
# execute_tool_call — single tool
# ---------------------------------------------------------------------------


async def test_successful_tool_returns_tool_result():
    factory = _mock_factory(return_value={"result": "some data"})
    result = await execute_tool_call(_make_tool_call(), tool_factory=factory)
    assert isinstance(result, ToolResult)


async def test_successful_tool_has_success_true():
    factory = _mock_factory(return_value={"result": "ok"})
    result = await execute_tool_call(_make_tool_call(), tool_factory=factory)
    assert result.success is True


async def test_successful_tool_preserves_tool_name():
    factory = _mock_factory()
    result = await execute_tool_call(
        _make_tool_call("google_search_tool"), tool_factory=factory
    )
    assert result.tool_name == "google_search_tool"


async def test_failed_tool_returns_tool_result_not_exception():
    """Executor must catch tool errors and wrap them — never re-raise."""
    factory = _mock_factory(side_effect=RuntimeError("something went wrong"))
    result = await execute_tool_call(_make_tool_call(), tool_factory=factory)
    assert isinstance(result, ToolResult)


async def test_failed_tool_has_success_false():
    factory = _mock_factory(side_effect=RuntimeError("boom"))
    result = await execute_tool_call(_make_tool_call(), tool_factory=factory)
    assert result.success is False


async def test_failed_tool_error_contains_message():
    factory = _mock_factory(side_effect=RuntimeError("boom"))
    result = await execute_tool_call(_make_tool_call(), tool_factory=factory)
    assert result.error is not None
    assert "boom" in result.error


# ---------------------------------------------------------------------------
# execute_tool_calls — batch / parallel
# ---------------------------------------------------------------------------


async def test_multiple_tools_all_results_returned():
    factory = _mock_factory(return_value={"ok": True})
    calls = [
        _make_tool_call("google_search_tool"),
        _make_tool_call("google_search_tool"),
    ]
    results = await execute_tool_calls(calls, tool_factory=factory)
    assert len(results) == 2


async def test_multiple_tools_all_are_tool_results():
    factory = _mock_factory(return_value={"ok": True})
    calls = [_make_tool_call(), _make_tool_call()]
    results = await execute_tool_calls(calls, tool_factory=factory)
    assert all(isinstance(r, ToolResult) for r in results)


async def test_partial_failure_does_not_block_other_results():
    """Even if one tool fails, the other should still succeed."""
    call_count = 0

    async def alternating(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("first fails")
        return {"ok": True}

    factory = MagicMock()
    factory.execute_tool = alternating

    calls = [_make_tool_call(), _make_tool_call()]
    results = await execute_tool_calls(calls, tool_factory=factory)

    assert len(results) == 2
    assert any(r.success is False for r in results)
    assert any(r.success is True for r in results)
