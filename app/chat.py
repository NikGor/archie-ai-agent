import asyncio
import logging
import os
from dotenv import load_dotenv
from .agent_builder import create_main_agent_response
from .state import get_state

logger = logging.getLogger(__name__)
load_dotenv()

# Get configuration from environment variables
PERSONA = os.getenv("DEFAULT_PERSONA", "business")
USER_NAME = os.getenv("DEFAULT_USER_NAME", "Николай")

# Get application state
logger.info(f"Initializing chat interface with persona: {PERSONA}, user: {USER_NAME}")
app_state = get_state(user_name=USER_NAME, persona=PERSONA)
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
            agent_response = await create_main_agent_response(history)
            logger.info(
                f"Agent processing successful. Response length: {len(agent_response.response)}"
            )
            logger.debug(f"Full agent result: {agent_response}")
            print(f"**Assistant**: {agent_response.response}")
            history.append({"role": "assistant", "content": agent_response.response})
        except Exception as e:
            logger.error(f"Error processing user input: {e}", exc_info=True)
            print(f"**Assistant**: Sorry, I encountered an error: {e}")


def main():
    """Main entry point for the application"""
    asyncio.run(chat())


if __name__ == "__main__":
    main()
