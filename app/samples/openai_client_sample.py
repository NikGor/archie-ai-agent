"""OpenAI client sample for testing updated create_completion method."""

import asyncio
import logging
from pydantic import BaseModel, Field
from app.backend.openai_client import OpenAIClient
from app.utils.llm_parser import parse_openai_response


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleResponse(BaseModel):
    """Simple structured response."""
    answer: str = Field(description="Answer text")
    confidence: float = Field(description="Confidence score 0-1")


def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get current weather for a location.

    Args:
        location: City name or address
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        Weather description string
    """
    return f"Weather in {location}: 22°{unit[0].upper()}, sunny"


def create_event(title: str, date: str, time: str) -> str:
    """
    Create a calendar event.

    Args:
        title: Event title
        date: Event date in YYYY-MM-DD format
        time: Event time in HH:MM format

    Returns:
        Confirmation message
    """
    return f"Created event: {title} on {date} at {time}"


async def test_simple_text(client: OpenAIClient, model: str):
    """Test 1: Simple text output without response_format or tools."""
    print(f"\n{'='*60}")
    print(f"TEST 1: Simple text output - {model}")
    print("="*60)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    response = await client.create_completion(
        messages=messages,
        model=model,
    )
    print(f"Response type: {type(response)}")
    print(f"Output text: {response.output_text}")
    print(f"Status: {response.status}")
    return response


async def test_structured_output(client: OpenAIClient, model: str):
    """Test 2: Structured output with response_format (как было)."""
    print(f"\n{'='*60}")
    print(f"TEST 2: Structured output - {model}")
    print("="*60)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    response = await client.create_completion(
        messages=messages,
        model=model,
        response_format=SimpleResponse,
    )
    print(f"Response type: {type(response)}")
    print(f"Parsed: {response.output_parsed}")
    print(f"Status: {response.status}")
    # Test parser
    parsed = parse_openai_response(response, SimpleResponse)
    print(f"Parser result: {parsed.parsed_content}")
    print(f"LLM trace: model={parsed.llm_trace.model}, in={parsed.llm_trace.input_tokens}, out={parsed.llm_trace.output_tokens}")
    return response


async def test_function_calling(client: OpenAIClient, model: str):
    """Test 3: Function calling without structured output."""
    print(f"\n{'='*60}")
    print(f"TEST 3: Function calling - {model}")
    print("="*60)
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "What's the weather in Berlin?"},
    ]
    response = await client.create_completion(
        messages=messages,
        model=model,
        tools=[get_weather, create_event],
    )
    print(f"Response type: {type(response)}")
    print(f"Output: {response.output}")
    print(f"Status: {response.status}")
    # Check for tool calls
    for item in response.output:
        if hasattr(item, "type") and item.type == "function_call":
            print(f"Tool call: {item.name}({item.arguments})")
    return response


async def test_function_with_structured(client: OpenAIClient, model: str):
    """Test 4: Function calling WITH structured output."""
    print(f"\n{'='*60}")
    print(f"TEST 4: Function + Structured - {model}")
    print("="*60)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Create an event for the user."},
        {"role": "user", "content": "Schedule a meeting tomorrow at 3pm called 'Team Sync'"},
    ]
    response = await client.create_completion(
        messages=messages,
        model=model,
        response_format=SimpleResponse,
        tools=[get_weather, create_event],
    )
    print(f"Response type: {type(response)}")
    print(f"Output: {response.output}")
    print(f"Parsed: {response.output_parsed}")
    print(f"Status: {response.status}")
    return response


async def main():
    client = OpenAIClient()
    # Test models
    standard_model = "gpt-4.1-mini"
    reasoning_model = "o3-mini"
    print("\n" + "="*60)
    print("TESTING OpenAI Client - Standard Model")
    print("="*60)
    await test_simple_text(client, standard_model)
    await test_structured_output(client, standard_model)
    await test_function_calling(client, standard_model)
    await test_function_with_structured(client, standard_model)
    print("\n" + "="*60)
    print("TESTING OpenAI Client - Reasoning Model")
    print("="*60)
    await test_simple_text(client, reasoning_model)
    await test_structured_output(client, reasoning_model)
    await test_function_calling(client, reasoning_model)
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())

# Run with: PYTHONPATH=. poetry run python app/samples/openai_client_sample.py
