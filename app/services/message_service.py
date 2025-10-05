"""Message service for handling message persistence via backend API."""

import logging
from archie_shared.chat.models import ChatMessage
from ..utils.backend_client import BackendClient

logger = logging.getLogger(__name__)


class MessageService:
    """Service for managing messages via backend API."""

    def __init__(self):
        self.backend_client = BackendClient()

    async def save_messages_to_database(
        self,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        conversation_id: str,
    ) -> None:
        """Save both user and assistant messages via backend API."""
        try:
            logger.info("message_001: Saving user message via API")
            logger.info(f"message_002: User message ID: \033[36m{user_message.message_id}\033[0m")
            
            # Set conversation_id for both messages
            user_message.conversation_id = conversation_id
            assistant_message.conversation_id = conversation_id
            
            # Save user message
            await self.backend_client.create_message(user_message)
            
            # Save assistant message  
            logger.info("message_003: Saving assistant message via API")
            logger.info(f"message_004: Assistant message ID: \033[36m{assistant_message.message_id}\033[0m")
            await self.backend_client.create_message(assistant_message)
            
            logger.info("message_005: Messages saved successfully via API")
        except Exception as e:
            logger.warning(f"message_error_001: Save error: \033[31m{e!s}\033[0m")
