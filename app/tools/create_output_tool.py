import logging
from typing import Any
from ..backend.openai_client import OpenAIClient
from ..backend.gemini_client import GeminiClient
from ..agent.prompt_builder import PromptBuilder
from ..config import MODEL_PROVIDERS
from ..models.output_models import AgentResponse
from ..utils.llm_parser import parse_llm_response


logger = logging.getLogger(__name__)


def _get_provider_for_model(model: str) -> str:
    """Returns provider name for a given model."""
    for provider, models in MODEL_PROVIDERS.items():
        if model in models:
            return provider
    return "openai"


async def create_output(
    user_input: str,
    command_summary: str,
    tool_results: list[dict[str, Any]] | None = None,
    response_format: str = "plain",
    model: str = "gpt-4.1",
    state: dict | None = None,
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

    Returns:
        AgentResponse: Final formatted response with SGROutput trace
    """
    logger.info("create_output_001: Creating final output response")
    logger.info(f"create_output_002: Format: \033[36m{response_format}\033[0m")

    prompt_builder = PromptBuilder()
    openai_client = OpenAIClient()
    gemini_client = GeminiClient()

    provider = _get_provider_for_model(model)
    client = openai_client if provider == "openai" else gemini_client
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
            tool_name = result.get("tool_name", "unknown")
            tool_output = result.get("output", {})
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

    logger.info(
        f"create_output_004: Calling LLM with \033[33m{len(messages)}\033[0m messages"
    )

    raw_response = await client.create_completion(
        messages=messages,
        model=model,
        response_format=AgentResponse,
    )

    parsed = parse_llm_response(
        raw_response=raw_response,
        provider=provider,
        expected_type=AgentResponse,
    )

    result = AgentResponse(
        content=parsed.parsed_content.content,
        sgr=parsed.parsed_content.sgr,
        llm_trace=parsed.llm_trace,
        response_id=parsed.response_id,
    )

    content_text = str(result.content) if result.content else ""
    logger.info(
        f"create_output_005: Created response with length: \033[33m{len(content_text)}\033[0m"
    )

    return result
