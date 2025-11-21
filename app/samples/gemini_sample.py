from datetime import datetime
import json
import logging
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
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


schedule_meeting_function = {
    "name": "schedule_meeting",
    "description": "Schedules a meeting with specified attendees at a given time and date.",
    "parameters": {
        "type": "object",
        "properties": {
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of people attending the meeting.",
            },
            "date": {
                "type": "string",
                "description": "Date of the meeting (e.g., '2024-07-29')",
            },
            "time": {
                "type": "string",
                "description": "Time of the meeting (e.g., '15:00')",
            },
            "topic": {
                "type": "string",
                "description": "The subject or topic of the meeting.",
            },
        },
        "required": ["attendees", "date", "time", "topic"],
    },
}


def chat():
    prompt = SIMPLE_PROMPT.format(
        current_date=current_date,
        location="Berlin, Germany"
    ) + "\n\nUser question: Schedule a meeting about the new project with Alice and Bob 29.11.2025 at 10 AM."
    
    tools = types.Tool(function_declarations=[schedule_meeting_function])

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            # "response_mime_type": "application/json",
            # "response_json_schema": SimpleResponse.model_json_schema(),
            "tools": [tools],
        },
    )
    
    # result = SimpleResponse.model_validate_json(response.text)
    result = response
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

    print("\nToken Usage:")
    print(response.usage_metadata)


def main():
    """Main entry point for the application"""
    chat()


if __name__ == "__main__":
    main()

# Run with: PYTHONPATH=. poetry run python app/samples/gemini_sample.py
