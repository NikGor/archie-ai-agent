"""The user's timezone reaches the web search (ARCHIE-257).

ToolFactory gives ToolContext fields to the tools that declare them,
tool_executor and CommandLoop pass the context, and google_search_tool tells
Gemini to give times in the user's timezone. The LLM never sees the parameter.
"""

import inspect
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, patch
import app.tools.google_search_tool as google_search_tool_module
from app.agent.command_loop import CommandLoop
from app.agent.context import ConversationContext
from app.agent.pipeline_metrics import PipelineMetrics
from app.models.orchestration_sgr import (
    ActionType,
    DecisionResponse,
    Parameter,
    SGROrchestration,
    ToolCallRequest,
)
from app.models.tool_models import ToolContext, ToolResult
from app.models.ws_models import StatusNotifier
from app.tools.google_search_tool import build_search_instruction, google_search_tool
from app.tools.tool_factory import ToolFactory
from app.utils.tool_executor import execute_tool_calls
from app.utils.tools_utils import openai_parse
from tests.conftest import make_user_state


BERLIN = ToolContext(user_timezone="Europe/Berlin")


async def _tool_with_timezone(query: str, user_timezone: str = "UTC") -> dict:
    return {"query": query, "user_timezone": user_timezone}


async def _tool_without_timezone(query: str) -> dict:
    return {"query": query}


def _factory() -> ToolFactory:
    factory = ToolFactory()
    factory.tools = {
        "google_search_tool": _tool_with_timezone,
        "notes_tool": _tool_without_timezone,
    }
    return factory


def _search_call(query: str) -> ToolCallRequest:
    return ToolCallRequest(
        tool_name="google_search_tool",
        arguments=[Parameter(name="query", value=query)],
        missing_parameters=[],
        is_confirmed=True,
        reason="test",
    )


def _decision(
    action_type: Literal["function_call", "final_response"],
    tool_calls: list[ToolCallRequest],
) -> DecisionResponse:
    return DecisionResponse(
        sgr=SGROrchestration(
            action=ActionType(type=action_type, reasoning="because"),
            tool_calls=tool_calls,
            reasoning="thinking",
            intents=[],
        ),
        handover_context="",
    )


# --- ToolFactory / tool_executor ---


async def test_execute_tool_gives_context_only_to_tools_that_declare_it():
    factory = _factory()
    assert await factory.execute_tool("google_search_tool", {"query": "q"}, BERLIN) == {
        "query": "q",
        "user_timezone": "Europe/Berlin",
    }
    assert await factory.execute_tool("notes_tool", {"query": "q"}, BERLIN) == {
        "query": "q"
    }


async def test_execute_tool_without_context_keeps_the_tool_default():
    result = await _factory().execute_tool("google_search_tool", {"query": "q"})
    assert result["user_timezone"] == "UTC"


async def test_context_wins_over_an_argument_from_the_llm():
    result = await _factory().execute_tool(
        "google_search_tool", {"query": "q", "user_timezone": "Asia/Tokyo"}, BERLIN
    )
    assert result["user_timezone"] == "Europe/Berlin"


async def test_execute_tool_calls_passes_context_to_every_call():
    results = await execute_tool_calls(
        [_search_call("a"), _search_call("b")], _factory(), context=BERLIN
    )
    assert [r.output["user_timezone"] for r in results] == ["Europe/Berlin"] * 2


async def test_command_loop_passes_the_user_timezone_to_tools():
    executed = AsyncMock(
        return_value=[ToolResult(tool_name="google_search_tool", output={})]
    )
    make_command_call = AsyncMock(
        side_effect=[
            (_decision("function_call", [_search_call("fixtures")]), None),
            (_decision("final_response", []), None),
        ]
    )
    with patch("app.agent.command_loop.execute_tool_calls", executed):
        await CommandLoop(_factory(), PipelineMetrics(), max_iterations=3).run(
            make_command_call=make_command_call,
            user_input="Футбол завтра",
            command_model="gpt-4.1",
            cmd_provider="openai",
            user_state=make_user_state(user_timezone="Europe/Berlin"),
            response_format="ui_answer",
            ctx=ConversationContext(previous_response_id=None, chat_history=None),
            seed_results=[],
            notifier=StatusNotifier(None),
            on_status=None,
        )
    assert executed.await_args.kwargs["context"] == BERLIN


# --- google_search_tool ---


def test_search_schema_does_not_show_user_timezone():
    assert set(openai_parse(google_search_tool)["parameters"]["properties"]) == {
        "query"
    }


def test_search_instruction_names_the_zone_and_local_time():
    instruction = build_search_instruction("Europe/Berlin")
    assert "Give every time in Europe/Berlin" in instruction
    assert "convert it to Europe/Berlin" in instruction
    assert " CET." in instruction or " CEST." in instruction  # zone of the local time


def test_search_instruction_falls_back_to_utc_for_an_unknown_zone():
    assert "Give every time in UTC" in build_search_instruction("Nowhere/Unknown")


async def test_search_sends_the_instruction_to_gemini(monkeypatch):
    configs = []

    def generate_content(model, contents, config):
        configs.append(config)
        return SimpleNamespace(
            text="answer",
            candidates=[SimpleNamespace(grounding_metadata=None)],
            usage_metadata=SimpleNamespace(
                prompt_token_count=1,
                candidates_token_count=1,
                thoughts_token_count=None,
            ),
        )

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(
        google_search_tool_module.settings, "gemini_api_key", "fake-key"
    )
    monkeypatch.setattr(
        google_search_tool_module.genai, "Client", lambda api_key: client
    )

    await google_search_tool("fixtures tomorrow", user_timezone="Europe/Berlin")
    await google_search_tool("fixtures tomorrow")

    assert "Give every time in Europe/Berlin" in configs[0].system_instruction
    default = inspect.signature(google_search_tool).parameters["user_timezone"].default
    assert f"Give every time in {default}" in configs[1].system_instruction
