"""
OpenAI client module for direct API integration using structured outputs.
"""
import json
import logging
import os
from typing import Any, Optional, List, Literal
from openai import OpenAI, pydantic_function_tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from .models import Metadata
from .tools import get_weather

logger = logging.getLogger(__name__)
load_dotenv()


class GetWeather(BaseModel):
    location: str

class CalculateTip(BaseModel):
    bill_amount: float
    tip_percentage: float
    split_between: Optional[int] = 1

class SendEmail(BaseModel):
    to: List[str]
    subject: str
    body: str
    cc: Optional[List[str]] = None
    priority: Optional[Literal["low", "normal", "high"]] = "normal"

class ConvertCurrency(BaseModel):
    amount: float
    from_currency: str
    to_currency: str
    date: Optional[str] = None

tools = [
    pydantic_function_tool(GetWeather, name="get_weather"),
    pydantic_function_tool(CalculateTip, name="calculate_tip"),
    pydantic_function_tool(SendEmail, name="send_email"),
    pydantic_function_tool(ConvertCurrency, name="convert_currency"),
]


class AgentResponse(BaseModel):
    """Response model for AI agent output"""

    response: str = Field(
        description="""
        Main text response from the AI agent in the specified response format.
        Don't duplicate metadata information in the main response text.
        """
    )
    thinking: str = Field(
        description="Internal reasoning or thought process and output designing"
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

    try:
        response = client.responses.parse(
            model=model,
            input=messages,
            text_format=AgentResponse,
            tools=tools,
        )
        if response.output[0].type == "function_call":
            # Parse arguments from JSON string to dict
            tool_arguments = json.loads(response.output[0].arguments)
            tool_result = await call_tool(
                tool_name=response.output[0].name,
                tool_arguments=tool_arguments,
            )
            logger.info(f"RML 911: Tool result: {tool_result}")
            
            # Add function result to messages and call model again
            messages.append({
                "role": "assistant",
                "content": f"Called Function: {response.output[0].name}"
            })
            messages.append({
                "role": "user", 
                "content": f"Function Result: {response.output[0].name}: {json.dumps(tool_result, ensure_ascii=False)}"
            })
            
            # Call model again with function result
            response = client.responses.parse(
                model=model,
                input=messages,
                text_format=AgentResponse,
            )
        
        logger.info(f"Received structured response from OpenAI: {response}")
        return response.output[0].content[0].parsed

    except Exception as e:
        logger.error(f"Error in OpenAI API call: {e}")
        raise
