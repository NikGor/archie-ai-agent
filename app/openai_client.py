"""
OpenAI client module for direct API integration using structured outputs.
"""
import json
import logging
import os
from typing import Any
from openai import OpenAI, pydantic_function_tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from .models import Metadata
from .tools import get_weather

logger = logging.getLogger(__name__)
load_dotenv()


class WeatherParams(BaseModel):
    """Parameters for weather tool"""
    location: str = Field(description="City and country e.g. Bogotá, Colombia")
    
class WebSearch(BaseModel):
    """Parameters for web search tool"""
    query: str = Field(description="Search query string")



class AgentResponse(BaseModel):
    """Response model for AI agent output"""

    response: str = Field(
        description="""
        Main text response from the AI agent in the specified response format.
        Don't duplicate metadata information in the main response text.
        """
    )
    metadata: Metadata = Field(
        description="Additional metadata for enriching the response"
    )


# Global client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def call_tool(
    tool_name: str,
    tool_arguments: dict[str, Any]
) -> Any:
    """
    Call a specific tool with the given arguments.

    Args:
        tool_name: The name of the tool to call.
        tool_arguments: The arguments to pass to the tool.

    Returns:
        The result of the tool call.
    """
    logger.info(f"Calling tool: {tool_name} with arguments: {tool_arguments}")
    try:
        if tool_name == "get_weather":
            # Extract location from arguments
            location = tool_arguments.get("location")
            result = get_weather(city_name=location)
            logger.info(f"Tool {tool_name} executed successfully")
            return result
        else:
            logger.error(f"Unknown tool: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return {"error": f"Tool execution failed: {str(e)}"}

async def create_agent_response(
    messages: list[dict[str, Any]],
    model: str = "gpt-4.1",
) -> AgentResponse:
    """
    Create an agent response using OpenAI structured outputs.
    
    Args:
        messages: Complete conversation history with system prompt already included
        model: OpenAI model to use
        
    Returns:
        Structured AgentResponse with response text and metadata
    """
    logger.debug(f"Sending {len(messages)} messages to OpenAI")

    tools = [
        pydantic_function_tool(
            WeatherParams,
            name="get_weather",  
            description="Get current temperature for a given location."
        ),
        pydantic_function_tool(
            WebSearch,
            name="web_search",
            description="Search the web for a given query."
        )
    ]

    try:
        response = client.responses.parse(
            model=model,
            input=messages,
            text_format=AgentResponse,
            # tools=tools,
        )
        logger.info(f"Received response: {response}")
        if getattr(response.output, "type", None) == "function_call":
            logger.info(
                f"RML 908: Agent called tool: {response.tool_name} with arguments: {response.tool_arguments}"
            )

            tool_result = await call_tool(
                tool_name=response.tool_name,
                tool_arguments=response.tool_arguments or {},
            )
            logger.info(f"RML 911: Tool result: {tool_result}")

            # Add tool call to chat history
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_call": {
                    "name": response.tool_name,
                    "arguments": response.tool_arguments or {},
                }
            })
            messages.append({
                "role": "function",
                "name": response.tool_name,
                "content": json.dumps(tool_result),
            })
            response = client.responses.parse(
                model=model,
                input=messages,
                text_format=AgentResponse,
            )
        
        logger.info(f"Received structured response from OpenAI: {response}")
        return response.output_parsed

    except Exception as e:
        logger.error(f"Error in OpenAI API call: {e}")
        raise
