import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def list_gemini_models():
    try:
        print("Available Gemini Models:")
        # The new SDK might have a different way to list models,
        # but let's try the standard way or check documentation if this fails.
        # Based on google-genai package usage:
        for m in client.models.list():
            print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing Gemini models: {e}")


if __name__ == "__main__":
    list_gemini_models()
