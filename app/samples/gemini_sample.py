from datetime import datetime
import json
import logging
import os
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

current_date = datetime.now().strftime("%B %d, %Y")


class SimpleResponse(BaseModel):
    """Simple test response model"""
    answer: str = Field(description="Answer text")
    reasoning: str = Field(description="Brief reasoning")


SIMPLE_PROMPT = """You are a helpful AI assistant.
Current date: {current_date}
User location: {location}

Answer the user's question clearly and concisely."""


def chat():
    prompt = SIMPLE_PROMPT.format(
        current_date=current_date,
        location="Berlin, Germany"
    ) + "\n\nUser question: What is the capital of France?"
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": SimpleResponse.model_json_schema(),
        },
    )
    
    result = SimpleResponse.model_validate_json(response.text)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

    print("\nToken Usage:")
    print(response.usage_metadata)


def main():
    """Main entry point for the application"""
    chat()


if __name__ == "__main__":
    main()

# Run with: PYTHONPATH=. poetry run python app/samples/gemini_sample.py
