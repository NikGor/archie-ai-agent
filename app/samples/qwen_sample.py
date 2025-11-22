import os
from pyexpat.errors import messages
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

client = instructor.from_openai(
    OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN"),
    ),
    mode=instructor.Mode.JSON,
)


class SimpleResponse(BaseModel):
    answer: str = Field(description="Answer text")
    reasoning: str = Field(description="Brief reasoning")


def chat():
    try:
        resp, raw_resp = client.chat.completions.create_with_completion(
            model="Qwen/Qwen3-4B-Instruct-2507",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_model=SimpleResponse,
        )
        print(raw_resp.model_dump_json(indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Validation failed completely: {e}")


if __name__ == "__main__":
    chat()
