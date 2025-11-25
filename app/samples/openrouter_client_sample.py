"""OpenRouter client sample for testing updated create_completion method."""

import asyncio
import logging
from pydantic import BaseModel, Field
from app.backend.openrouter_client import OpenRouterClient
from app.utils.tools_utils import openai_parse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_MODELS = [
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4.1-mini",
]


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


async def test_simple_text(client: OpenRouterClient, model: str) -> bool:
    """Test 1: Simple text output - should work on ALL models."""
    print(f"\n--- Simple text: {model} ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    try:
        response = await client.create_completion(
            messages=messages,
            model=model,
        )
        content = response.choices[0].message.content
        print(f"✅ Response: {content[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_structured_output(client: OpenRouterClient, model: str) -> bool:
    """Test 2: Structured output with response_format."""
    print(f"\n--- Structured output: {model} ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    try:
        response = await client.create_completion(
            messages=messages,
            model=model,
            response_format=SimpleResponse,
        )
        content = response.choices[0].message.content
        parsed = SimpleResponse.model_validate_json(content)
        print(f"✅ Parsed: answer={parsed.answer}, confidence={parsed.confidence}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_function_calling(client: OpenRouterClient, model: str) -> bool:
    """Test 3: Function calling without structured output."""
    print(f"\n--- Function calling: {model} ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when needed."},
        {"role": "user", "content": "What's the weather in Berlin?"},
    ]
    # Parse tools to OpenRouter format (dict schemas)
    tools = [openai_parse(get_weather), openai_parse(create_event)]
    try:
        response = await client.create_completion(
            messages=messages,
            model=model,
            tools=tools,
        )
        message = response.choices[0].message
        if message.tool_calls:
            for tc in message.tool_calls:
                print(f"✅ Tool call: {tc.function.name}({tc.function.arguments})")
            return True
        else:
            print(f"⚠️ No tool call, response: {message.content[:100] if message.content else 'None'}...")
            return True  # Not a failure, model chose not to use tools
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def run_all_tests():
    """Run all tests on all models."""
    client = OpenRouterClient()
    results = {}
    for model in TEST_MODELS:
        print(f"\n{'='*60}")
        print(f"TESTING: {model}")
        print("="*60)
        results[model] = {
            "simple": await test_simple_text(client, model),
            "structured": await test_structured_output(client, model),
            "functions": await test_function_calling(client, model),
        }
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    for model, tests in results.items():
        status = "✅" if all(tests.values()) else "⚠️"
        details = " | ".join(f"{k}:{'✅' if v else '❌'}" for k, v in tests.items())
        print(f"{status} {model}: {details}")


async def test_single(model: str):
    """Test a single model."""
    client = OpenRouterClient()
    print(f"\n{'='*60}")
    print(f"TESTING: {model}")
    print("="*60)
    await test_simple_text(client, model)
    await test_structured_output(client, model)
    await test_function_calling(client, model)


def main():
    import sys
    if len(sys.argv) > 1:
        model = sys.argv[1]
        asyncio.run(test_single(model))
    else:
        asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()

# Run with:
#   PYTHONPATH=. poetry run python app/samples/openrouter_client_sample.py                    # Test all
#   PYTHONPATH=. poetry run python app/samples/openrouter_client_sample.py google/gemini-2.5-flash  # Test one
