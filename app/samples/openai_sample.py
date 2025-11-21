from datetime import datetime
import json
import logging
import os
from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

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

Answer the user's question clearly and concisely."""


def chat():
    system_prompt = {
        "role": "system",
        "content": SIMPLE_PROMPT.format(
            current_date=current_date,
            location="Berlin, Germany"
        ),
    }
    user_input = {
        "role": "user",
        "content": "What is the capital of France?",
    }
    messages = [system_prompt, user_input]
    
    response = client.responses.parse(
        model="gpt-4.1",
        input=messages,
        text_format=SimpleResponse,
    )
    
    # result = response.output[0].content[0].parsed
    result = response
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


def main():
    """Main entry point for the application"""
    chat()


if __name__ == "__main__":
    main()

# Run with: PYTHONPATH=. poetry run python app/response_understand.py
