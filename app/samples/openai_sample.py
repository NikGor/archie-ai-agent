from datetime import datetime
import json
import logging
import os
from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from app.models import orchestration_sgr
from app.tools.tool_factory import ToolFactory
from app.utils.llm_parser import parse_llm_response

load_dotenv()
logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

current_date = datetime.now().strftime("%B %d, %Y")


class SimpleResponse(BaseModel):
    """Simple test response model"""

    answer: str = Field(description="Answer text")
    reasoning: str = Field(description="Brief reasoning")


SIMPLE_PROMPT = """You are a helpful AI assistant.
Current date: {current_date}
User location: {location}

Always strictly follow the provided response format.

Available tools:
{tools_list}

Answer the user's question clearly and concisely."""


def chat():
    tool_factory = ToolFactory()
    model = "gpt-4.1"
    response_format = "plain"

    tools = tool_factory.get_tool_schemas(model, response_format)
    tools_formatted = json.dumps(tools, indent=2, ensure_ascii=False)

    system_prompt = {
        "role": "system",
        "content": SIMPLE_PROMPT.format(
            current_date=current_date,
            location="Berlin, Germany",
            tools_list=tools_formatted,
        ),
    }
    user_input = {
        "role": "user",
        "content": "Create a dinner appointment for tomorrow at 7 PM with John",
    }
    messages = [system_prompt, user_input]

    response = client.responses.parse(
        model="gpt-4.1",
        input=messages,
        text_format=orchestration_sgr.DecisionResponse,
    )

    # Test parser
    parsed = parse_llm_response(
        raw_response=response,
        provider="openai",
        expected_type=orchestration_sgr.DecisionResponse,
    )

    print("=== Parsed Response ===")
    print(f"Response ID: {parsed.response_id}")
    print(f"Has Function Call: {parsed.has_function_call}")
    if parsed.has_function_call:
        print(f"Function Name: {parsed.function_name}")
        print(f"Function Arguments: {parsed.function_arguments}")
    else:
        print(f"Parsed Content Type: {type(parsed.parsed_content).__name__}")
        print(
            f"Content: {json.dumps(parsed.parsed_content.model_dump(), indent=2, ensure_ascii=False)}"
        )

    print("\n=== LLM Trace ===")
    print(json.dumps(parsed.llm_trace.model_dump(), indent=2, ensure_ascii=False))

    # print("\n=== Full Raw Response ===")
    # print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))


def main():
    """Main entry point for the application"""
    chat()


if __name__ == "__main__":
    main()

# Run with: PYTHONPATH=. poetry run python app/samples/openai_sample.py
