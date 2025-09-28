import asyncio
import logging
import os

from agents import Runner
from dotenv import load_dotenv

from .agent_builder import build_main_agent
from .state import get_state

logger = logging.getLogger(__name__)
load_dotenv()

# Get configuration from environment variables
PERSONA = os.getenv("DEFAULT_PERSONA", "business")
USER_NAME = os.getenv("DEFAULT_USER_NAME", "Николай")

# Get application state
logger.info(f"Initializing chat interface with persona: {PERSONA}, user: {USER_NAME}")
app_state = get_state(user_name=USER_NAME, persona=PERSONA)
main_agent = build_main_agent(app_state)
logger.info("Chat interface initialized successfully")


async def chat():
    print("Welcome to the AI assistant. Type 'exit' to quit.")
    print(f"Active persona: {PERSONA}")
    logger.info("Starting chat session")
    history = []
    while True:
        user_input = input("**You**: ")
        if user_input.lower() == "exit":
            print("**Assistant**: Goodbye!")
            logger.info("Chat session ended by user")
            break

        logger.info(f"Processing user input: {user_input}")
        history.append({"role": "user", "content": user_input})

        try:
            result = await Runner.run(
                main_agent,
                history,
            )
            logger.info(
                f"Agent processing successful. Response length: {len(str(result))}"
            )
            logger.debug(f"Full agent result: {result}")
            print(f"**Assistant**: {result}")
            history.append({"role": "assistant", "content": str(result)})
        except Exception as e:
            logger.error(f"Error processing user input: {e}", exc_info=True)
            print(f"**Assistant**: Sorry, I encountered an error: {e}")


def main():
    """Main entry point for the application"""
    asyncio.run(chat())
