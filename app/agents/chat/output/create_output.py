"""Stage 3 of the agent flow: formatting the final response based on command
decisions and tool results.
"""

import logging
from typing import Any
from app import config
from app.agents.chat.formats import resolve_format
from app.agents.chat.messages import ChatPromptBuilder
from app.agents.chat.output.postprocess import build_agent_response
from app.agents.chat.output.runners import stream_and_collect
from app.agents.chat.requests import OutputRequest
from app.models.output_models import (
    AgentResponse,
    Level2Response,
    PlainResponse,
)
from app.platform.llm.catalog import get_provider_for_model
from app.platform.llm.registry import get_client, parse_llm_response
from app.platform.prompts.context import datetime_note
from app.platform.tools.models import ToolResult
from app.utils.llm_parser import parse_assembled_stream
from app.utils.schema_filter import build_filtered_ui_response
from app.utils.stream_utils import JsonPathExtractor


# Langfuse generation name for Stage 3 on every path (ARCHIE-211)
_OBSERVATION_NAME = "generate-response"

logger = logging.getLogger(__name__)


def _build_static_prompt(
    prompt_builder: ChatPromptBuilder,
    response_format: str,
    intents: list[str],
    state: dict,
) -> str:
    """Stable, cacheable part of the Stage 3 system prompt (ARCHIE-180)."""
    format_instructions = prompt_builder.build_format_instructions(
        response_format, intents=intents
    )
    assistant_context = prompt_builder.build_assistant_prompt(state, response_format)
    return f"""You are creating the final response for the user.

# Format Instructions
{format_instructions}

# Assistant Context
{assistant_context}"""


def _build_volatile_prompt(
    prompt_builder: ChatPromptBuilder,
    command_summary: str,
    state: dict,
    tool_results: list[ToolResult] | None,
) -> str:
    """Per-turn part of the Stage 3 system prompt, kept out of the cache prefix."""
    tools_context = ""
    if tool_results:
        tools_context = "\n\nTool Results:\n"
        for tool_result in tool_results:
            tools_context += f"- {tool_result.tool_name}: {tool_result.output}\n"
        logger.info(
            f"create_output_003: Added \033[33m{len(tool_results)}\033[0m tool results to context"
        )
    datetime_line = datetime_note(state)
    return f"""# Command Summary
{command_summary}

{datetime_line}{tools_context}

Create a complete, well-formatted response in the specified format."""


async def create_output(request: OutputRequest) -> AgentResponse:
    """
    Create final formatted output response.

    This is Stage 3 of the agent flow - formatting the final response
    based on command decisions and tool results.

    Args:
        user_input: Original user request
        command_summary: Summary of command decisions made
        tool_results: Results from executed tools (if any)
        response_format: Target format (plain, ui_answer, dashboard, formatted_text)
        model: LLM model to use
        state: User state context
        previous_response_id: Previous response ID for OpenAI conversation threading
        chat_history: Chat history text for non-OpenAI providers
        on_stream: Callback for streaming text tokens (plain/voice + OpenRouter only)
        on_stream_event: Callback for stream_placeholder/stream_reasoning events (UI + OpenRouter)
        reasoning_effort: Reasoning effort of the model; None keeps the provider default

    Returns:
        AgentResponse: Final formatted response with SGROutput trace
    """
    user_input = request.user_input
    command_summary = request.command_summary
    tool_results = request.tool_results
    response_format = request.response_format
    model = request.model
    state = request.state
    previous_response_id = request.previous_response_id
    chat_history = request.chat_history
    intents = request.intents
    no_image = request.no_image
    on_stream = request.on_stream
    on_stream_event = request.on_stream_event
    reasoning_effort = request.reasoning_effort
    logger.info("create_output_001: Creating final output response")
    logger.info(f"create_output_002: Format: \033[36m{response_format}\033[0m")

    prompt_builder = ChatPromptBuilder()
    provider = get_provider_for_model(model)
    client = get_client(provider)
    logger.info(
        f"create_output_002b: Using provider: \033[34m{provider}\033[0m for model: \033[36m{model}\033[0m"
    )

    state = state or {}
    intents = intents or []

    static_content = _build_static_prompt(
        prompt_builder, response_format, intents, state
    )
    volatile_content = _build_volatile_prompt(
        prompt_builder, command_summary, state, tool_results
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": static_content, "cacheable": True},
    ]
    if chat_history:
        messages.append(
            {
                "role": "system",
                "content": f"Chat History:\n{chat_history}",
                "cacheable": True,
            }
        )
        logger.info(
            f"create_output_003b: Added chat_history to context (len: \033[33m{len(chat_history)}\033[0m)"
        )
    messages.append({"role": "system", "content": volatile_content})
    messages.append({"role": "user", "content": user_input})

    logger.info(
        f"create_output_004: Calling LLM with \033[33m{len(messages)}\033[0m messages"
    )

    if response_format == "ui_answer":
        # intents=[] → base models only (Card, TextAnswer, Table, Image). No fallback to full schema.
        response_model = build_filtered_ui_response(
            tuple(sorted(intents)), no_image=no_image
        )
        logger.info(
            f"create_output_004b: Using filtered UIResponse for intents: \033[35m{intents}\033[0m, no_image: \033[35m{no_image}\033[0m"
        )
    else:
        response_model = resolve_format(response_format).output_model
        logger.info(
            f"create_output_004b: Using response model: \033[36m{response_model.__name__}\033[0m"
        )

    # ── Streaming path: plain/voice + on_stream callback ──────────────────────
    if (
        hasattr(client, "create_completion_stream")
        and resolve_format(response_format).stream_mode == "text"
        and on_stream
    ):
        response_id_out: list[str] = []
        usage_out: dict[str, int] = {}
        full_json, ttft_ms = await stream_and_collect(
            client=client,
            messages=messages,
            model=model,
            response_model=response_model,
            previous_response_id=previous_response_id,
            observation_name=_OBSERVATION_NAME,
            reasoning_effort=reasoning_effort,
            extractor=JsonPathExtractor(["text"]),
            on_chunk=on_stream,
            response_id_out=response_id_out,
            usage_out=usage_out,
        )
        parsed_stream = parse_assembled_stream(
            full_json, model, PlainResponse, usage_out
        )
        return await build_agent_response(
            parsed_content=parsed_stream.parsed_content,
            response_format=response_format,
            llm_trace=parsed_stream.llm_trace,
            response_id=response_id_out[0] if response_id_out else None,
            ttft_ms=ttft_ms,
            no_image=no_image,
        )

    # ── Streaming path: level2_answer text tokens ──────────────────────────────
    if (
        hasattr(client, "create_completion_stream")
        and resolve_format(response_format).stream_mode == "level2"
        and on_stream_event
    ):

        async def _on_chunk_l2(chunk: str) -> None:
            await on_stream_event("stream_delta", chunk)

        response_id_out_l2: list[str] = []
        usage_out_l2: dict[str, int] = {}
        full_json_l2, ttft_ms_l2 = await stream_and_collect(
            client=client,
            messages=messages,
            model=model,
            response_model=response_model,
            previous_response_id=previous_response_id,
            observation_name=_OBSERVATION_NAME,
            reasoning_effort=reasoning_effort,
            extractor=JsonPathExtractor(["level2_answer", "text", "text"]),
            on_chunk=_on_chunk_l2,
            response_id_out=response_id_out_l2,
            usage_out=usage_out_l2,
        )
        parsed_l2 = parse_assembled_stream(
            full_json_l2, model, Level2Response, usage_out_l2
        )
        return await build_agent_response(
            parsed_content=parsed_l2.parsed_content,
            response_format=response_format,
            llm_trace=parsed_l2.llm_trace,
            # ARCHIE-210: without the resp_ id the next OpenAI turn loses the thread
            response_id=response_id_out_l2[0] if response_id_out_l2 else None,
            ttft_ms=ttft_ms_l2,
            no_image=no_image,
        )

    # ── Streaming path: UI formats + on_stream_event callback ─────────────────
    if (
        hasattr(client, "create_completion_stream")
        and resolve_format(response_format).stream_mode == "ui"
        and on_stream_event
    ):
        await on_stream_event("stream_placeholder", None)

        async def _on_chunk_ui(chunk: str) -> None:
            await on_stream_event("stream_reasoning", chunk)

        extra_extractors_ui = None
        if response_format == "ui_answer":
            _intro_extractor = JsonPathExtractor(["ui_answer", "intro_text", "text"])

            async def _on_chunk_intro(chunk: str) -> None:
                await on_stream_event("stream_delta", chunk)

            extra_extractors_ui = [(_intro_extractor, _on_chunk_intro)]

        response_id_out_ui: list[str] = []
        usage_out_ui: dict[str, int] = {}
        full_json_ui, ttft_ms_ui = await stream_and_collect(
            client=client,
            messages=messages,
            model=model,
            response_model=response_model,
            previous_response_id=previous_response_id,
            observation_name=_OBSERVATION_NAME,
            reasoning_effort=reasoning_effort,
            extractor=JsonPathExtractor(["sgr", "rationale"]),
            on_chunk=_on_chunk_ui,
            response_id_out=response_id_out_ui,
            max_output_tokens=config.UI_STREAM_MAX_OUTPUT_TOKENS,
            usage_out=usage_out_ui,
            extra_extractors=extra_extractors_ui,
        )
        parsed_stream_ui = parse_assembled_stream(
            full_json_ui, model, response_model, usage_out_ui
        )
        return await build_agent_response(
            parsed_content=parsed_stream_ui.parsed_content,
            response_format=response_format,
            llm_trace=parsed_stream_ui.llm_trace,
            response_id=response_id_out_ui[0] if response_id_out_ui else None,
            ttft_ms=ttft_ms_ui,
            no_image=no_image,
        )

    # ── Blocking path ──────────────────────────────────────────────────────────
    # previous_response_id/chat_history are expected to already be gated for
    # `provider` by the caller (see ConversationContext.for_provider).
    raw_response = await client.create_completion(
        messages=messages,
        model=model,
        response_format=response_model,
        previous_response_id=previous_response_id,
        observation_name=_OBSERVATION_NAME,
        reasoning_effort=reasoning_effort,
    )
    parsed = parse_llm_response(
        raw_response=raw_response,
        provider=provider,
        expected_type=response_model,
    )
    return await build_agent_response(
        parsed_content=parsed.parsed_content,
        response_format=response_format,
        llm_trace=parsed.llm_trace,
        response_id=parsed.response_id,
        no_image=no_image,
    )
