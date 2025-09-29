"""
OpenAI client module for direct API integration using structured outputs.
"""
import json
import logging
import os
from typing import Any, Optional, List, Literal, Dict
from openai import OpenAI, pydantic_function_tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from .models import Metadata
from .tools import get_weather

logger = logging.getLogger(__name__)
load_dotenv()


class GetWeather(BaseModel):
    location: str


tools = [
    {"type": "web_search"},
    pydantic_function_tool(GetWeather, name="get_weather"),
]

class SourceRef(BaseModel):
    id: int = Field(description="Local incremental id for this session (1..N)")
    url: str = Field(description="Source URL")
    title: Optional[str] = Field(default=None, description="Page/article title")
    snippet: Optional[str] = Field(default=None, description="Short relevant excerpt")

class EvidenceItem(BaseModel):
    claim: str = Field(description="Concrete factual claim used in the response/metadata")
    support: Literal["supported", "contradicted", "uncertain"] = Field(
        description="Does the cited evidence support the claim?"
    )
    source_ids: List[int] = Field(
        description="IDs from sources[] backing this claim (empty if uncertain)"
    )

class RoutingDecision(BaseModel):
    intent: Literal[
        "answer_general", "weather", "sports_score", "web_search", "clarify", "out_of_scope"
    ] = Field(description="Chosen path for this turn")
    rationale: str = Field(description="One-sentence reason for choosing this intent")

class SlotsStatus(BaseModel):
    required: List[str] = Field(description="Required fields for this intent")
    collected: Dict[str, str] = Field(description="Slot name -> value already available")
    missing: List[str] = Field(description="Still missing; ask one-by-one")

class PreActionChecklist(BaseModel):
    summary: str = Field(description="Short summary of the planned action/assumption")
    decision: Literal["confirm", "clarify", "reject", "none"] = Field(
        description="If an action is pending, ask for confirmation; 'none' if N/A"
    )

class VerificationStatus(BaseModel):
    level: Literal["verified", "partially_verified", "unverified"] = Field(
        description="Overall verification level for factual content"
    )
    confidence_pct: int = Field(ge=0, le=100, description="Calibrated confidence 0..100")

class SGRTrace(BaseModel):
    """Schema-Guided Reasoning trace (not user-facing UI)"""
    routing: RoutingDecision
    slots: SlotsStatus
    evidence: List[EvidenceItem] = Field(
        default_factory=list, description="Claims and how they are supported"
    )
    sources: List[SourceRef] = Field(
        default_factory=list, description="Deduplicated list of sources used"
    )
    verification: VerificationStatus
    pre_action: PreActionChecklist

# --- Integrate with your AgentResponse ---

class AgentResponse(BaseModel):
    """Response model for AI agent output"""

    response: str = Field(
        description=(
            "Main text response from the AI agent in the specified response format. "
            "Don't duplicate metadata information in the main response text."
        )
    )
    metadata: Metadata = Field(
        description="Additional metadata for enriching the response"
    )
    sgr: SGRTrace = Field(
        description="Mandatory SGR reasoning trace for this turn (internal; not to be shown to user as-is)"
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
            location = tool_arguments.get("location")
            result = get_weather(city_name=location)
            return result
        else:
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
            tool_name = response.output[0].name
            tool_result = await call_tool(
                tool_name=tool_name,
                tool_arguments=tool_arguments,
            )
            logger.info(f"RML 911: Tool result: {tool_result}")
            
            # Add function result to messages and call model again
            messages.append({
                "role": "assistant", 
                "content": f"Function Result: {tool_name}: {tool_result}"
            })
            
            # Call model again with function result
            response = client.responses.parse(
                model=model,
                input=messages,
                text_format=AgentResponse,
            )
        elif response.output[0].type == "web_search_call":
            # Web search was performed, find the message with parsed response
            for output_item in response.output:
                if output_item.type == "message":
                    messages.append({
                        "role": "assistant",
                        "content": output_item.content[0].text
                    })
                    break
            response = client.responses.parse(
                model=model,
                input=messages,
                text_format=AgentResponse,
            )

        return response.output[0].content[0].parsed

    except Exception as e:
        logger.error(f"Error in OpenAI API call: {e}")
        raise
