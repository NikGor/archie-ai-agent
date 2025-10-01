"""API Controller - Facade connecting endpoints to services."""

import logging

from .models import ChatMessage
from .services.chat_service import ChatService

logger = logging.getLogger(__name__)
chat_service = ChatService()


async def handle_chat(user_message: ChatMessage) -> ChatMessage:
    """Handle chat message - facade to chat service."""
    return await chat_service.process_chat_message(user_message)
