"""Conversation service for managing conversations via backend API."""

import logging
from ..utils.backend_client import BackendClient

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for managing conversations via backend API."""

    def __init__(self):
        self.backend_client = BackendClient()

    async def create_new_conversation(self) -> str:
        """Create a new conversation via backend API and return its ID."""
        try:
            logger.info("conversation_001: Creating new conversation via API")
            conversation_id = await self.backend_client.create_conversation()
            logger.info(f"conversation_002: Created conversation: \033[32m{conversation_id}\033[0m")
            return conversation_id
        except Exception as e:
            logger.error(f"conversation_error_001: Creation error: \033[31m{e!s}\033[0m")
            raise


