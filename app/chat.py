import os
import asyncio
import logging
from dotenv import load_dotenv
from agents import Runner
from .agent_builder import build_main_agent
from .state import get_state


logger = logging.getLogger(__name__)
load_dotenv()

PERSONA = 'business'
USER_NAME = 'Николай'

# Get application state
app_state = get_state(user_name=USER_NAME, persona=PERSONA)
main_agent = build_main_agent(app_state)

async def chat():
    print("Welcome to the AI assistant. Type 'exit' to quit.")
    print(f"Active persona: {PERSONA}")
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        history.append({"role": "user", "content": user_input})
        result = await Runner.run(
            main_agent,
            history,
        )
        assistant_reply = result.final_output if hasattr(result, "final_output") else str(result)
        print(f"Assistant: {assistant_reply}")
        history.append({"role": "assistant", "content": assistant_reply})

def main():
    """Main entry point for the application"""
    asyncio.run(chat())
