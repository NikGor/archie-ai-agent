import logging
from typing import Any

from archie_shared.chat.models import Content

from ..agent.prompt_builder import PromptBuilder
from ..backend.gemini_client import GeminiClient
from ..backend.openai_client import OpenAIClient
from ..backend.openrouter_client import OpenRouterClient
from ..models.output_models import AgentResponse, get_response_model_for_format
from ..models.tool_models import ToolResult
from ..utils.llm_parser import parse_llm_response, build_content_from_parsed
from ..utils.provider_utils import get_provider_for_model


logger = logging.getLogger(__name__)

# Initialize clients once at module level
_openai_client = OpenAIClient()
_openrouter_client = OpenRouterClient()
_gemini_client = GeminiClient()

_clients = {
    "openai": _openai_client,
    "openrouter": _openrouter_client,
    "gemini": _gemini_client,
}


async def create_output(
    user_input: str,
    command_summary: str,
    tool_results: list[ToolResult] | None = None,
    response_format: str = "plain",
    model: str = "gpt-4.1",
    state: dict | None = None,
    previous_response_id: str | None = None,
    chat_history: str | None = None,
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

    format_instructions = prompt_builder.build_format_instructions(response_format)
    assistant_context = prompt_builder.build_assistant_prompt(state, response_format)

    tools_context = ""
    if tool_results:
        tools_context = "\n\nTool Results:\n"
        for result in tool_results:
            tool_name = result.tool_name
            tool_output = result.output
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

    response_model = get_response_model_for_format(response_format)
    logger.info(
        f"create_output_004b: Using response model: \033[36m{response_model.__name__}\033[0m"
    )

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

    content = build_content_from_parsed(
        parsed_content=parsed.parsed_content,
        response_format=response_format,
    )

    result = AgentResponse(
        content=content,
        sgr=parsed.parsed_content.sgr,
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
