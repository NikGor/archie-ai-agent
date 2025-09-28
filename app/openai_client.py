"""
OpenAI client module for direct API integration using structured outputs.
"""

import logging
import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .models import Metadata

logger = logging.getLogger(__name__)
load_dotenv()


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
        )
        logger.debug("Received structured response from OpenAI")
        return response.output_parsed

    except Exception as e:
        logger.error(f"Error in OpenAI API call: {e}")
        raise
