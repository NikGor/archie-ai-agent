import json
import logging
import os

import httpx
import yaml
from dotenv import load_dotenv

from .agent_builder import create_main_agent_response
from .models import ChatMessage
from .utils import generate_conversation_id, generate_message_id

logger = logging.getLogger(__name__)
load_dotenv()
PERSONA = os.getenv("DEFAULT_PERSONA", "business")
USER_NAME = os.getenv("DEFAULT_USER_NAME", "Николай")
DEFAULT_CONVERSATION_ID = os.getenv("DEFAULT_CONVERSATION_ID", "default")
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8002")


async def create_new_conversation() -> str:
    """Create a new conversation and return its ID."""
    try:
        logger.info("api_003: Creating new conversation")
        create_payload = {}
        logger.info(
            f"api_004: POST /conversations\n\033[36m{json.dumps(create_payload, indent=2)}\033[0m"
        )
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_API_URL}/conversations",
                json=create_payload,
            )
            if response.status_code == 200:
                conversation_data = response.json()
                conversation_id = conversation_data["conversation_id"]
                logger.info(f"api_005: New conv: \033[32m{conversation_id}\033[0m")
                return conversation_id
            else:
                return generate_conversation_id()
    except Exception:
        return generate_conversation_id()


async def load_conversation_history(conversation_id: str) -> list[dict[str, str]]:
    """Load conversation history from backend."""
    try:
        logger.info(f"api_006: Loading history for: \033[36m{conversation_id}\033[0m")
        logger.info(f"api_007: GET /chat_history?conversation_id={conversation_id}")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BACKEND_API_URL}/chat_history",
                params={"conversation_id": conversation_id},
            )
            if response.status_code == 200:
                history_data = yaml.safe_load(response.text)
                conversation_history = [
                    {"role": msg["role"], "content": msg["text"]}
                    for msg in history_data["messages"]
                ]
                logger.info(
                    f"api_008: Loaded chat history with: \033[33m{len(conversation_history)}\033[0m msgs"
                )
                return conversation_history
            return []
    except Exception:
        return []


async def save_messages_to_backend(
    user_message: ChatMessage,
    assistant_message: ChatMessage,
    conversation_id: str,
) -> None:
    """Save both user and assistant messages to backend."""
    try:
        async with httpx.AsyncClient() as client:
            user_payload = {
                "role": user_message.role,
                "text": user_message.text,
                "text_format": user_message.text_format,
                "conversation_id": conversation_id,
            }
            logger.info("api_010: Saving user message")
            logger.info(
                f"api_011: POST /messages\n\033[36m{json.dumps(user_payload, indent=2, ensure_ascii=False)}\033[0m"
            )
            await client.post(
                f"{BACKEND_API_URL}/messages",
                json=user_payload,
            )
            assistant_payload = {
                "role": assistant_message.role,
                "text": assistant_message.text,
                "text_format": assistant_message.text_format,
                "conversation_id": conversation_id,
                "metadata": (
                    assistant_message.metadata.dict()
                    if assistant_message.metadata
                    else None
                ),
            }
            logger.info("api_012: Saving assistant message")
            logger.info(
                f"api_013: POST /messages\n\033[36m{json.dumps(assistant_payload, indent=2, ensure_ascii=False)}\033[0m"
            )
            await client.post(
                f"{BACKEND_API_URL}/messages",
                json=assistant_payload,
            )
            logger.info("api_014: Messages saved successfully")
    except Exception as e:
        logger.warning(f"api_015: Save error: \033[31m{e!s}\033[0m")


async def handle_chat(user_message: ChatMessage) -> ChatMessage:
    """Handle chat message with external backend API."""
    logger.info("=== STEP 3: Message Processing ===")
    logger.info(
        f"api_002: processing conversation: \033[36m{user_message.conversation_id or 'NEW'}\033[0m"
    )
    conversation_id = user_message.conversation_id
    if not conversation_id:
        conversation_id = await create_new_conversation()
    conversation_history = await load_conversation_history(conversation_id)
    user_message.conversation_id = conversation_id
    user_message.message_id = generate_message_id()
    conversation_history.append({"role": "user", "content": user_message.text})
    logger.info("=== STEP 4: AI Processing ===")
    agent_response = await create_main_agent_response(conversation_history)
    response_text = agent_response.response
    metadata = agent_response.metadata
    logger.info("=== STEP 5: Saving to Backend ===")
    logger.info(f"api_009: Response len: \033[33m{len(response_text)}\033[0m")
    assistant_message = ChatMessage(
        message_id=generate_message_id(),
        role="assistant",
        text=response_text,
        text_format=user_message.text_format,
        conversation_id=conversation_id,
        metadata=metadata,
    )
    await save_messages_to_backend(user_message, assistant_message, conversation_id)
    return assistant_message
