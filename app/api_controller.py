import logging
import os
import httpx
from dotenv import load_dotenv
from .agent_builder import create_main_agent_response
from .models import ChatMessage
from .utils import generate_conversation_id, generate_message_id

logger = logging.getLogger(__name__)
load_dotenv()

# Get configuration from environment variables
PERSONA = os.getenv("DEFAULT_PERSONA", "business")
USER_NAME = os.getenv("DEFAULT_USER_NAME", "Николай")
DEFAULT_CONVERSATION_ID = os.getenv("DEFAULT_CONVERSATION_ID", "default")
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8002")

logger.info(f"Initializing with persona: {PERSONA}, user: {USER_NAME}")
logger.info(f"Backend API URL: {BACKEND_API_URL}")


async def handle_chat(user_message: ChatMessage) -> ChatMessage:
    """Handle chat message with external backend API."""

    logger.info(f"Handling chat message for conversation: {user_message.conversation_id or 'new'}")
    logger.debug(f"User message: {user_message.text}")

    conversation_id = user_message.conversation_id
    conversation_history = []

    # Case 1: No conversation_id - create new conversation
    if not conversation_id:
        logger.info("Creating new conversation")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BACKEND_API_URL}/conversations",
                    json={}
                )
                if response.status_code == 200:
                    conversation_data = response.json()
                    conversation_id = conversation_data["conversation_id"]
                    logger.info(f"Created new conversation: {conversation_id}")
                else:
                    logger.error(f"Failed to create conversation: {response.status_code}")
                    conversation_id = generate_conversation_id()
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            conversation_id = generate_conversation_id()

    # Case 2: conversation_id exists - load chat history
    else:
        logger.info(f"Loading history for conversation: {conversation_id}")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{BACKEND_API_URL}/chat_history",
                    params={"conversation_id": conversation_id}
                )
                if response.status_code == 200:
                    history_data = response.json()
                    # Convert to OpenAI format
                    conversation_history = [
                        {"role": msg["role"], "content": msg["text"]}
                        for msg in history_data["messages"]
                    ]
                    logger.debug(f"Loaded {len(conversation_history)} messages from history")
                else:
                    logger.warning(f"Failed to load conversation history: {response.status_code}")
        except Exception as e:
            logger.warning(f"Error loading conversation history: {e}")

    # Update user message with conversation_id and generate message_id
    user_message.conversation_id = conversation_id
    user_message.message_id = generate_message_id()

    # Add user message to history for agent processing
    conversation_history.append({"role": "user", "content": user_message.text})

    # Process with agent
    logger.info("Processing message with AI agent...")
    agent_response = await create_main_agent_response(conversation_history)
    logger.debug(f"Agent processing completed")

    # Extract response text and metadata from the structured output
    response_text = agent_response.response
    metadata = agent_response.metadata

    # Create assistant message
    assistant_message = ChatMessage(
        message_id=generate_message_id(),
        role="assistant",
        text=response_text,
        text_format=user_message.text_format,
        conversation_id=conversation_id,
        metadata=metadata,
    )

    # Save both user and assistant messages to backend
    try:
        async with httpx.AsyncClient() as client:
            # Save user message
            await client.post(
                f"{BACKEND_API_URL}/messages",
                json={
                    "role": user_message.role,
                    "text": user_message.text,
                    "text_format": user_message.text_format,
                    "conversation_id": conversation_id,
                }
            )
            logger.debug("User message saved to backend")

            # Save assistant message
            await client.post(
                f"{BACKEND_API_URL}/messages",
                json={
                    "role": assistant_message.role,
                    "text": assistant_message.text,
                    "text_format": assistant_message.text_format,
                    "conversation_id": conversation_id,
                    "metadata": assistant_message.metadata.dict() if assistant_message.metadata else None,
                }
            )
            logger.debug("Assistant message saved to backend")
    except Exception as e:
        logger.warning(f"Error saving messages to backend: {e}")

    logger.info("Chat message handling completed successfully")
    return assistant_message
