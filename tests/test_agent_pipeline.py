"""
Smoke tests for the full AgentFactory pipeline with real LLM calls.

Marked with @pytest.mark.llm — skipped automatically when OPENAI_API_KEY is not set
(see conftest.py).  Tests only verify the shape and completeness of AgentResponse,
not the semantic content of the LLM output.
"""

import pytest

from app.models.output_models import AgentResponse
from app.models.ws_models import StatusUpdate


pytestmark = pytest.mark.llm


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------


async def test_response_is_agent_response(agent_factory, simple_messages):
    response = await agent_factory.arun(
        messages=simple_messages, response_format="plain"
    )
    assert isinstance(response, AgentResponse)


async def test_response_has_content(agent_factory, simple_messages):
    response = await agent_factory.arun(
        messages=simple_messages, response_format="plain"
    )
    assert response.content is not None


async def test_response_has_sgr(agent_factory, simple_messages):
    response = await agent_factory.arun(
        messages=simple_messages, response_format="plain"
    )
    assert response.sgr is not None


async def test_response_has_llm_trace(agent_factory, simple_messages):
    response = await agent_factory.arun(
        messages=simple_messages, response_format="plain"
    )
    assert response.llm_trace is not None


# ---------------------------------------------------------------------------
# Plain format contract
# ---------------------------------------------------------------------------


async def test_plain_format_content_format(agent_factory, simple_messages):
    response = await agent_factory.arun(
        messages=simple_messages, response_format="plain"
    )
    assert response.content.content_format == "plain"


async def test_plain_format_text_is_nonempty_string(agent_factory, simple_messages):
    response = await agent_factory.arun(
        messages=simple_messages, response_format="plain"
    )
    assert isinstance(response.content.text, str)
    assert len(response.content.text) > 0


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------


async def test_llm_trace_has_nonzero_tokens(agent_factory, simple_messages):
    response = await agent_factory.arun(
        messages=simple_messages, response_format="plain"
    )
    assert response.llm_trace.total_tokens > 0
    assert response.llm_trace.input_tokens > 0


# ---------------------------------------------------------------------------
# Status callback
# ---------------------------------------------------------------------------


async def test_status_callback_fires(agent_factory, simple_messages):
    events: list[StatusUpdate] = []

    async def capture(update: StatusUpdate) -> None:
        events.append(update)

    await agent_factory.arun(
        messages=simple_messages,
        response_format="plain",
        on_status=capture,
    )

    assert len(events) >= 2
    assert all(isinstance(e, StatusUpdate) for e in events)


# ---------------------------------------------------------------------------
# Dashboard format — bypasses Stage 1 (separate code path)
# ---------------------------------------------------------------------------


async def test_dashboard_format_returns_content(agent_factory, simple_messages):
    response = await agent_factory.arun(
        messages=simple_messages, response_format="dashboard"
    )
    assert response.content is not None


async def test_dashboard_format_content_format(agent_factory, simple_messages):
    response = await agent_factory.arun(
        messages=simple_messages, response_format="dashboard"
    )
    assert response.content.content_format == "dashboard"
