"""Message service for handling message persistence."""

import json
import logging

import httpx

from ..models import ChatMessage

logger = logging.getLogger(__name__)


class MessageService:
    """Service for managing messages with external backend."""

    def __init__(self, backend_url: str):
        self.backend_url = backend_url

    async def save_messages_to_backend(
        self,
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
                logger.info("message_001: Saving user message")
                logger.info(
                    f"message_002: POST /messages\n\033[36m{json.dumps(user_payload, indent=2, ensure_ascii=False)}\033[0m"
                )
                await client.post(
                    f"{self.backend_url}/messages",
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
                    "llm_trace": (
                        assistant_message.llm_trace.dict()
                        if assistant_message.llm_trace
                        else None
                    ),
                }
                logger.info("message_003: Saving assistant message")
                logger.info(
                    f"message_004: POST /messages\n\033[36m{json.dumps(assistant_payload, indent=2, ensure_ascii=False)}\033[0m"
                )
                await client.post(
                    f"{self.backend_url}/messages",
                    json=assistant_payload,
                )
                logger.info("message_005: Messages saved successfully")
        except Exception as e:
            logger.warning(f"message_error_001: Save error: \033[31m{e!s}\033[0m")
