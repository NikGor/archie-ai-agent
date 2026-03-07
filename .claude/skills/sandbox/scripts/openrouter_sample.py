"""OpenRouter sample for testing structured outputs and parser compatibility."""

from datetime import datetime
import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from app.models import orchestration_sgr
from app.tools.tool_factory import ToolFactory
from app.utils.llm_parser import parse_openrouter_response

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=OPENROUTER_BASE_URL,
)

current_date = datetime.now().strftime("%B %d, %Y")

# Models to test - comment/uncomment as needed
TEST_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    # "google/gemini-3-pro-preview",  # Known issues with empty responses
    "anthropic/claude-sonnet-4",
    # "anthropic/claude-haiku-4.5",
    # "x-ai/grok-4-fast",
    # "deepseek/deepseek-chat-v3-0324:free",
]


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


def test_single_model(
    model: str,
    messages: list[dict],
    response_model: type[BaseModel],
    schema: dict,
) -> dict:
    """Test a single model and return results."""
    result = {
        "model": model,
        "success": False,
        "error": None,
        "raw_response": None,
        "parsed": None,
        "parser_result": None,
    }
    try:
        print(f"\n{'='*60}")
        print(f"Testing: {model}")
        print("=" * 60)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        # Full response structure for debugging
        print(f"\n--- Full Response Structure ---")
        print(f"id: {response.id}")
        print(f"model: {response.model}")
        print(f"object: {getattr(response, 'object', None)}")
        print(f"created: {getattr(response, 'created', None)}")
        print(
            f"choices: {type(response.choices).__name__} len={len(response.choices) if response.choices else 0}"
        )
        if response.choices:
            c = response.choices[0]
            print(f"  [0].index: {c.index}")
            print(f"  [0].finish_reason: {c.finish_reason}")
            print(f"  [0].message.role: {c.message.role}")
            print(
                f"  [0].message.content: {c.message.content[:100] if c.message.content else None}..."
            )
        print(f"usage: {response.usage}")
        if response.usage:
            print(f"  prompt_tokens: {getattr(response.usage, 'prompt_tokens', None)}")
            print(
                f"  completion_tokens: {getattr(response.usage, 'completion_tokens', None)}"
            )
            print(f"  total_tokens: {getattr(response.usage, 'total_tokens', None)}")
            print(
                f"  prompt_tokens_details: {getattr(response.usage, 'prompt_tokens_details', None)}"
            )
            print(
                f"  completion_tokens_details: {getattr(response.usage, 'completion_tokens_details', None)}"
            )
        result["raw_response"] = {
            "id": response.id,
            "model": response.model,
            "choices_count": len(response.choices) if response.choices else 0,
            "has_usage": response.usage is not None,
        }
        if response.choices:
            content = response.choices[0].message.content
            if content:
                result["parsed"] = response_model.model_validate_json(content)
                print(f"\n--- Manual Parse OK ---")
        print(f"\n--- Testing llm_parser.parse_openrouter_response ---")
        parser_result = parse_openrouter_response(response, response_model)
        result["parser_result"] = {
            "parsed_type": type(parser_result.parsed_content).__name__,
            "response_id": parser_result.response_id,
            "model": parser_result.llm_trace.model,
            "input_tokens": parser_result.llm_trace.input_tokens,
            "output_tokens": parser_result.llm_trace.output_tokens,
        }
        print(f"Parser OK: {result['parser_result']}")
        result["success"] = True
        print(f"\n✅ {model} - SUCCESS")
    except Exception as e:
        result["error"] = str(e)
        print(f"\n❌ {model} - FAILED: {e}")
    return result


def test_all_models():
    """Test all models in TEST_MODELS list."""
    tool_factory = ToolFactory()
    response_format_type = "plain"
    tools = tool_factory.get_tool_schemas(TEST_MODELS[0], response_format_type)
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
    response_model = orchestration_sgr.DecisionResponse
    schema = response_model.model_json_schema()
    if "properties" in schema:
        schema["properties"].pop("llm_trace", None)
        schema["properties"].pop("response_id", None)
        if "required" in schema:
            schema["required"] = [
                field
                for field in schema["required"]
                if field not in ["llm_trace", "response_id"]
            ]
    results = []
    for model in TEST_MODELS:
        result = test_single_model(model, messages, response_model, schema)
        results.append(result)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r["success"])
    print(f"Passed: {success_count}/{len(results)}")
    print()
    for r in results:
        status = "✅" if r["success"] else "❌"
        error_info = f" - {r['error'][:50]}..." if r["error"] else ""
        print(f"{status} {r['model']}{error_info}")
    return results


def test_single(model: str = "google/gemini-2.5-flash"):
    """Test a single specific model."""
    tool_factory = ToolFactory()
    response_format_type = "plain"
    tools = tool_factory.get_tool_schemas(model, response_format_type)
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
    response_model = orchestration_sgr.DecisionResponse
    schema = response_model.model_json_schema()
    if "properties" in schema:
        schema["properties"].pop("llm_trace", None)
        schema["properties"].pop("response_id", None)
        if "required" in schema:
            schema["required"] = [
                field
                for field in schema["required"]
                if field not in ["llm_trace", "response_id"]
            ]
    return test_single_model(model, messages, response_model, schema)


def main():
    """Main entry point - run all model tests."""
    import sys

    if len(sys.argv) > 1:
        model = sys.argv[1]
        print(f"Testing single model: {model}")
        test_single(model)
    else:
        test_all_models()


if __name__ == "__main__":
    main()

# Run with:
#   PYTHONPATH=. poetry run python app/samples/openrouter_sample.py                    # Test all
#   PYTHONPATH=. poetry run python app/samples/openrouter_sample.py google/gemini-2.5-flash  # Test one
