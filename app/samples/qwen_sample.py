from datetime import datetime
import json
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

current_date = datetime.now().strftime("%B %d, %Y")


class SimpleResponse(BaseModel):
    """Simple test response model"""
    answer: str = Field(description="Answer text")
    reasoning: str = Field(description="Brief reasoning")


SIMPLE_PROMPT = """You are a helpful AI assistant.
Current date: {current_date}
User location: {location}

Answer the user's question clearly and concisely in JSON format.
Follow this JSON schema:
{json_schema}
"""


def chat():
    system_prompt = {
        "role": "system",
        "content": SIMPLE_PROMPT.format(
            current_date=current_date,
            location="Berlin, Germany",
            json_schema=json.dumps(SimpleResponse.model_json_schema(), indent=2)
        ),
    }
    user_input = {
        "role": "user",
        "content": "What is the capital of France?",
    }
    messages = [system_prompt, user_input]
    
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=messages,
        response_format={
            "type": "json_object"
        },
    )
    
    content = response.choices[0].message.content
    try:
        result = SimpleResponse.model_validate_json(content)
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw content: {content}")

    if hasattr(response, 'usage'):
        print("\nToken Usage:")
        print(response.usage)


def main():
    """Main entry point for the application"""
    chat()


if __name__ == "__main__":
    main()

# Run with: PYTHONPATH=. poetry run python app/samples/qwen_sample.py