"""OpenAI client module for direct API integration using structured outputs."""

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, pydantic_function_tool
from pydantic import BaseModel

from .models.response_models import (
    AgentResponse,
)
from .tools import get_weather
from .utils.openai_utils import create_llm_trace_from_openai_response

logger = logging.getLogger(__name__)
load_dotenv()


class GetWeather(BaseModel):
    location: str


tools = [
    {"type": "web_search"},
    pydantic_function_tool(GetWeather, name="get_weather"),
]


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def call_tool(
    tool_name: str,
    tool_arguments: dict[str, Any],
) -> Any:
    """Call a specific tool with the given arguments."""
    logger.info(f"Calling tool: {tool_name} with arguments: {tool_arguments}")
    try:
        if tool_name == "get_weather":
            location = tool_arguments.get("location")
            result = await get_weather(city_name=location)
            return result
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return {"error": f"Tool execution failed: {e!s}"}


async def handle_function_call(
    response: Any,
    messages: list[dict[str, Any]],
    model: str,
) -> Any:
    """Handle function call response from OpenAI."""
    tool_arguments = json.loads(response.output[0].arguments)
    tool_name = response.output[0].name
    tool_result = await call_tool(
        tool_name=tool_name,
        tool_arguments=tool_arguments,
    )
    logger.info(f"openai_tool_result: Tool {tool_name} result: {tool_result}")
    messages.append(
        {
            "role": "assistant",
            "content": f"Function Result: {tool_name}: {tool_result}",
        }
    )
    return client.responses.parse(
        model=model,
        input=messages,
        text_format=AgentResponse,
    )


async def handle_web_search_call(
    response: Any,
    messages: list[dict[str, Any]],
    model: str,
) -> Any:
    """Handle web search call response from OpenAI."""
    logger.info("openai_websearch: Processing web search call")
    for output_item in response.output:
        if output_item.type == "message":
            messages.append(
                {
                    "role": "assistant",
                    "content": output_item.content[0].text,
                }
            )
            logger.debug("openai_websearch: Added message content to conversation")
            break
    return client.responses.parse(
        model=model,
        input=messages,
        text_format=AgentResponse,
    )


def log_response(result: AgentResponse) -> None:
    """Log the agent response in a structured format."""
    logger.info(f"openai_002: Response len: \033[33m{len(result.response)}\033[0m")
    response_dict = {
        "response": result.response,
        "metadata": result.metadata.dict() if result.metadata else None,
    }
    logger.info(
        f"openai_003: Full response:\n\033[32m{json.dumps(response_dict, indent=2, ensure_ascii=False)}\033[0m"
    )


async def create_agent_response(
    messages: list[dict[str, Any]],
    model: str = "gpt-4.1",
) -> AgentResponse:
    """Create an agent response using OpenAI structured outputs."""
    logger.info(
        f"openai_001: Calling \033[36m{model}\033[0m with \033[33m{len(messages)}\033[0m msgs"
    )
    try:
        response = client.responses.parse(
            model=model,
            input=messages,
            text_format=AgentResponse,
            tools=tools,
        )

        # Handle different response types using raw response first
        if response.output[0].type == "function_call":
            response = await handle_function_call(response, messages, model)
        elif response.output[0].type == "web_search_call":
            response = await handle_web_search_call(response, messages, model)

        # Log metrics directly from response
        usage = response.usage
        logger.info(
            f"openai_004: Usage - Input: \033[33m{usage.input_tokens}\033[0m | "  # type: ignore
            f"Output: \033[33m{usage.output_tokens}\033[0m | "  # type: ignore
            f"Total: \033[33m{usage.total_tokens}\033[0m | "  # type: ignore
            f"Cached: \033[33m{usage.input_tokens_details.cached_tokens}\033[0m"  # type: ignore
        )
        logger.info(
            f"openai_005: Status: {response.status} | Model: \033[36m{response.model}\033[0m"
        )

        # Extract the AgentResponse and create new one with LLM trace
        parsed_result = response.output[0].content[0].parsed  # type: ignore
        llm_trace = create_llm_trace_from_openai_response(response)

        result = AgentResponse(
            response=parsed_result.response,  # type: ignore
            metadata=parsed_result.metadata,  # type: ignore
            sgr=parsed_result.sgr,  # type: ignore
            llm_trace=llm_trace,
        )

        logger.info(
            f"openai_debug: Created AgentResponse with llm_trace: {hasattr(result, 'llm_trace')}"
        )
        logger.info(f"openai_debug: AgentResponse type: {type(result)}")
        log_response(result)
        return result

    except Exception as e:
        logger.error(f"openai_error_001: \033[31m{e!s}\033[0m")
        raise
