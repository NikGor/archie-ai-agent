import logging
from pydantic import ValidationError
from ..agent.prompt_builder import PromptBuilder
from ..backend.gemini_client import GeminiClient
from ..backend.openai_client import OpenAIClient
from ..backend.openrouter_client import OpenRouterClient
from ..models.output_models import (
    AgentResponse,
    PlainResponse,
    UIResponse,
    get_response_model_for_format,
)
from ..models.tool_models import ToolResult
from ..models.ws_models import StreamCallback
from ..utils.llm_parser import build_content_from_parsed, parse_llm_response
from ..utils.provider_utils import get_provider_for_model
from ..utils.schema_filter import build_filtered_ui_response
from ..utils.stream_utils import JsonTextExtractor

_STREAMABLE_FORMATS = frozenset({"plain", "voice"})

logger = logging.getLogger(__name__)

# Initialize clients once at module level
_openai_client = OpenAIClient()
_openrouter_client = OpenRouterClient()
_gemini_client = GeminiClient()

_clients: dict[str, OpenAIClient | OpenRouterClient | GeminiClient] = {
    "openai": _openai_client,
    "openrouter": _openrouter_client,
    "gemini": _gemini_client,
}


def _clear_card_image_prompts(ui_response: UIResponse) -> None:
    """Clear image_prompt on all Card objects inside CardGrid items."""
    for item in ui_response.ui_answer.items:
        if item.type == "card_grid":
            for card in item.content.cards:  # type: ignore[union-attr]
                if hasattr(card, "image_prompt"):
                    setattr(card, "image_prompt", None)
    logger.info("create_output_008: no_image=True — cleared image_prompt on all cards")


async def create_output(
    user_input: str,
    command_summary: str,
    tool_results: list[ToolResult] | None = None,
    response_format: str = "plain",
    model: str = "gpt-4.1",
    state: dict | None = None,
    previous_response_id: str | None = None,
    chat_history: str | None = None,
    intents: list[str] | None = None,
    no_image: bool = False,
    on_stream: StreamCallback = None,
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
        on_stream: Optional callback for streaming text tokens (plain/voice + OpenRouter only)

    Returns:
        AgentResponse: Final formatted response with SGROutput trace
    """
    logger.info("create_output_001: Creating final output response")
    logger.info(f"create_output_002: Format: \033[36m{response_format}\033[0m")

    prompt_builder = PromptBuilder()

    provider = get_provider_for_model(model)
    client = _clients[provider]
    logger.info(
        f"create_output_002b: Using provider: \033[34m{provider}\033[0m for model: \033[36m{model}\033[0m"
    )

    if state is None:
        state = {}
    if intents is None:
        intents = []

    format_instructions = prompt_builder.build_format_instructions(
        response_format, intents=intents
    )
    assistant_context = prompt_builder.build_assistant_prompt(state, response_format)

    tools_context = ""
    if tool_results:
        tools_context = "\n\nTool Results:\n"
        for tool_result in tool_results:
            tool_name = tool_result.tool_name
            tool_output = tool_result.output
            tools_context += f"- {tool_name}: {tool_output}\n"
        logger.info(
            f"create_output_003: Added \033[33m{len(tool_results)}\033[0m tool results to context"
        )

    system_prompt_content = f"""You are creating the final response for the user.

# Command Summary
{command_summary}

# Format Instructions
{format_instructions}

# Assistant Context
{assistant_context}
{tools_context}

Create a complete, well-formatted response in the specified format."""

    messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": user_input},
    ]

    if chat_history:
        messages.insert(
            1,
            {"role": "system", "content": f"Chat History:\n{chat_history}"},
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

    # ── Streaming path: OpenRouter + plain/voice + on_stream callback ──────────
    if (
        provider == "openrouter"
        and response_format in _STREAMABLE_FORMATS
        and on_stream
    ):
        assert isinstance(client, OpenRouterClient)
        extractor = JsonTextExtractor()
        json_parts: list[str] = []
        async for token in client.create_completion_stream(
            messages=messages,
            model=model,
            response_format=response_model,
        ):
            json_parts.append(token)
            text_chunk = extractor.feed(token)
            if text_chunk:
                await on_stream(text_chunk)

        full_json = "".join(json_parts)
        parsed_obj = PlainResponse.model_validate_json(full_json)
        content = build_content_from_parsed(
            parsed_content=parsed_obj,
            response_format=response_format,
        )
        result = AgentResponse(
            content=content,
            sgr=parsed_obj.sgr,
            llm_trace=None,
            response_id=None,
        )
        content_text = str(result.content) if result.content else ""
        logger.info(
            f"create_output_005: Streamed response length: \033[33m{len(content_text)}\033[0m"
        )
        logger.info(
            f"create_output_006: UI reasoning: \033[35m{result.sgr.ui_reasoning}\033[0m"
        )
        return result

    # ── Blocking path ──────────────────────────────────────────────────────────
    raw_response = await client.create_completion(
        messages=messages,
        model=model,
        response_format=response_model,
        previous_response_id=previous_response_id if provider == "openai" else None,
    )

    parsed = parse_llm_response(
        raw_response=raw_response,
        provider=provider,
        expected_type=response_model,
    )

    # Coerce dynamic filtered model back to standard UIResponse
    # so build_content_from_parsed works without changes.
    parsed_content = parsed.parsed_content
    if response_format == "ui_answer":
        try:
            parsed_content = UIResponse.model_validate(parsed_content.model_dump())
        except ValidationError as e:
            logger.error(f"create_output_007: UIResponse coercion failed: {e}")
            raise
        if no_image:
            _clear_card_image_prompts(parsed_content)

    content = build_content_from_parsed(
        parsed_content=parsed_content,
        response_format=response_format,
    )

    result = AgentResponse(
        content=content,
        sgr=parsed_content.sgr,
        llm_trace=parsed.llm_trace,
        response_id=parsed.response_id,
    )

    content_text = str(result.content) if result.content else ""
    logger.info(
        f"create_output_005: Created response with length: \033[33m{len(content_text)}\033[0m"
    )
    logger.info(
        f"create_output_006: UI reasoning: \033[35m{result.sgr.ui_reasoning}\033[0m"
    )

    return result
