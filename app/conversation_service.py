"""Conversation service for managing conversations and message history."""

import json
import logging

import httpx
import yaml

from .utils import generate_conversation_id

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for managing conversations with external backend."""

    def __init__(self, backend_url: str):
        self.backend_url = backend_url

    async def create_new_conversation(self) -> str:
        """Create a new conversation and return its ID."""
        try:
            logger.info("conversation_001: Creating new conversation")
            create_payload = {}
            logger.info(
                f"conversation_002: POST /conversations\n\033[36m{json.dumps(create_payload, indent=2)}\033[0m"
            )
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/conversations",
                    json=create_payload,
                )
                if response.status_code == 200:
                    conversation_data = response.json()
                    conversation_id = conversation_data["conversation_id"]
                    logger.info(
                        f"conversation_003: New conv: \033[32m{conversation_id}\033[0m"
                    )
                    return conversation_id
                else:
                    return generate_conversation_id()
        except Exception:
            return generate_conversation_id()

    async def load_conversation_history(
        self, conversation_id: str
    ) -> list[dict[str, str]]:
        """Load conversation history from backend."""
        try:
            logger.info(
                f"conversation_004: Loading history for: \033[36m{conversation_id}\033[0m"
            )
            logger.info(
                f"conversation_005: GET /chat_history?conversation_id={conversation_id}"
            )
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.backend_url}/chat_history",
                    params={"conversation_id": conversation_id},
                )
                if response.status_code == 200:
                    history_data = yaml.safe_load(response.text)
                    conversation_history = [
                        {"role": msg["role"], "content": msg["text"]}
                        for msg in history_data["messages"]
                    ]
                    logger.info(
                        f"conversation_006: Loaded chat history with: \033[33m{len(conversation_history)}\033[0m msgs"
                    )
                    return conversation_history
                return []
        except Exception:
            return []
