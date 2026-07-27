"""Stage 3 of the agent flow: formatting the final response based on command
decisions and tool results.
"""

import logging
from ... import config
from ...agent.prompt_builder import PromptBuilder
from ...backend.llm.registry import get_client
from ...models.output_models import (
    AgentResponse,
    Level2Response,
    PlainResponse,
    get_response_model_for_format,
)
from ...models.tool_models import ToolResult
from ...models.ws_models import StreamCallback, StreamEventCallback
from ...utils.llm_parser import parse_assembled_stream, parse_llm_response
from ...utils.provider_utils import get_provider_for_model
from ...utils.schema_filter import build_filtered_ui_response
from ...utils.stream_utils import JsonPathExtractor
from .postprocess import build_agent_response
from .runners import stream_and_collect


_STREAMABLE_FORMATS = frozenset({"plain", "voice", "formatted_text"})
_LEVEL2_STREAMABLE_FORMATS = frozenset({"level2_answer"})
_UI_STREAMABLE_FORMATS = frozenset(
    {"ui_answer", "dashboard", "widget", "level3_answer"}
)

logger = logging.getLogger(__name__)


def _build_system_prompt(
    prompt_builder: PromptBuilder,
    response_format: str,
    intents: list[str],
    command_summary: str,
    state: dict,
    tool_results: list[ToolResult] | None,
) -> str:
    format_instructions = prompt_builder.build_format_instructions(
        response_format, intents=intents
    )
    assistant_context = prompt_builder.build_assistant_prompt(state, response_format)

    tools_context = ""
    if tool_results:
        tools_context = "\n\nTool Results:\n"
        for tool_result in tool_results:
            tools_context += f"- {tool_result.tool_name}: {tool_result.output}\n"
        logger.info(
            f"create_output_003: Added \033[33m{len(tool_results)}\033[0m tool results to context"
        )

    return f"""You are creating the final response for the user.

# Command Summary
{command_summary}

# Format Instructions
{format_instructions}

# Assistant Context
{assistant_context}
{tools_context}

Create a complete, well-formatted response in the specified format."""


async def create_output(
    user_input: str,
    command_summary: str,
    tool_results: list[ToolResult] | None = None,
    response_format: str = "plain",
    model: str = config.DEFAULT_MODEL,
    state: dict | None = None,
    previous_response_id: str | None = None,
    chat_history: str | None = None,
    intents: list[str] | None = None,
    no_image: bool = False,
    on_stream: StreamCallback = None,
    on_stream_event: StreamEventCallback = None,
) -> AgentResponse:
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

    Returns:
        AgentResponse: Final formatted response with SGROutput trace
    """
    logger.info("create_output_001: Creating final output response")
    logger.info(f"create_output_002: Format: \033[36m{response_format}\033[0m")

    prompt_builder = PromptBuilder()
    provider = get_provider_for_model(model)
    client = get_client(provider)
    logger.info(
        f"create_output_002b: Using provider: \033[34m{provider}\033[0m for model: \033[36m{model}\033[0m"
    )

    state = state or {}
    intents = intents or []

    system_prompt_content = _build_system_prompt(
        prompt_builder, response_format, intents, command_summary, state, tool_results
    )
    messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": user_input},
    ]
    if chat_history:
        messages.insert(
            1, {"role": "system", "content": f"Chat History:\n{chat_history}"}
        )
        logger.info(
            f"create_output_003b: Added chat_history to context (len: \033[33m{len(chat_history)}\033[0m)"
        )

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
        response_model = get_response_model_for_format(response_format)
        logger.info(
            f"create_output_004b: Using response model: \033[36m{response_model.__name__}\033[0m"
        )

    # ── Streaming path: plain/voice + on_stream callback ──────────────────────
    if (
        hasattr(client, "create_completion_stream")
        and response_format in _STREAMABLE_FORMATS
        and on_stream
    ):
        response_id_out: list[str] = []
        full_json, ttft_ms = await stream_and_collect(
            client=client,
            messages=messages,
            model=model,
            response_model=response_model,
            previous_response_id=previous_response_id,
            extractor=JsonPathExtractor(["text"]),
            on_chunk=on_stream,
            response_id_out=response_id_out,
        )
        parsed_stream = parse_assembled_stream(full_json, model, PlainResponse)
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
        and response_format in _LEVEL2_STREAMABLE_FORMATS
        and on_stream_event
    ):

        async def _on_chunk_l2(chunk: str) -> None:
            await on_stream_event("stream_delta", chunk)  # type: ignore[misc]

        full_json_l2, ttft_ms_l2 = await stream_and_collect(
            client=client,
            messages=messages,
            model=model,
            response_model=response_model,
            previous_response_id=previous_response_id,
            extractor=JsonPathExtractor(["level2_answer", "text", "text"]),
            on_chunk=_on_chunk_l2,
        )
        parsed_l2 = parse_assembled_stream(full_json_l2, model, Level2Response)
        return await build_agent_response(
            parsed_content=parsed_l2.parsed_content,
            response_format=response_format,
            llm_trace=parsed_l2.llm_trace,
            response_id=None,
            ttft_ms=ttft_ms_l2,
            no_image=no_image,
        )

    # ── Streaming path: UI formats + on_stream_event callback ─────────────────
    if (
        hasattr(client, "create_completion_stream")
        and response_format in _UI_STREAMABLE_FORMATS
        and on_stream_event
    ):
        await on_stream_event("stream_placeholder", None)

        async def _on_chunk_ui(chunk: str) -> None:
            await on_stream_event("stream_reasoning", chunk)  # type: ignore[misc]

        extra_extractors_ui = None
        if response_format == "ui_answer":
            _intro_extractor = JsonPathExtractor(["ui_answer", "intro_text", "text"])

            async def _on_chunk_intro(chunk: str) -> None:
                await on_stream_event("stream_delta", chunk)  # type: ignore[misc]

            extra_extractors_ui = [(_intro_extractor, _on_chunk_intro)]

        response_id_out_ui: list[str] = []
        full_json_ui, ttft_ms_ui = await stream_and_collect(
            client=client,
            messages=messages,
            model=model,
            response_model=response_model,
            previous_response_id=previous_response_id,
            extractor=JsonPathExtractor(["sgr", "reasoning"]),
            on_chunk=_on_chunk_ui,
            response_id_out=response_id_out_ui,
            max_output_tokens=config.UI_STREAM_MAX_OUTPUT_TOKENS,
            extra_extractors=extra_extractors_ui,
        )
        parsed_stream_ui = parse_assembled_stream(full_json_ui, model, response_model)
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
