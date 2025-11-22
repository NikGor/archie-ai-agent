import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def list_models():
    try:
        models = client.models.list()
        print("Available Models:")
        for model in sorted(models.data, key=lambda x: x.id):
            if model.id.startswith(("o1", "o3", "gpt-5", "gpt-4")):
                print(f"- {model.id}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()
