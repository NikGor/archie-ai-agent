"""Chat service for coordinating chat workflow."""

import logging
import os
from dotenv import load_dotenv
from archie_shared.chat.models import ChatMessage
from ..agent_builder import create_main_agent_response
from ..utils.general_utils import generate_message_id
from .conversation_service import ConversationService
from .message_service import MessageService

logger = logging.getLogger(__name__)
load_dotenv()
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8002")


class ChatService:
    """Service for coordinating chat workflow."""

    def __init__(self):
        self.conversation_service = ConversationService(BACKEND_API_URL)
        self.message_service = MessageService(BACKEND_API_URL)

    async def process_chat_message(self, user_message: ChatMessage) -> ChatMessage:
        """Process a chat message through the complete workflow."""
        logger.info("=== STEP 3: Message Processing ===")
        logger.info(
            f"chat_001: processing conversation: \033[36m{user_message.conversation_id or 'NEW'}\033[0m"
        )

        # Get or create conversation
        conversation_id = user_message.conversation_id
        if not conversation_id:
            conversation_id = await self.conversation_service.create_new_conversation()

        # Load conversation history
        conversation_history = (
            await self.conversation_service.load_conversation_history(conversation_id)
        )

        # Prepare user message
        user_message.conversation_id = conversation_id
        user_message.message_id = generate_message_id()
        conversation_history.append({"role": "user", "content": user_message.text})

        # Generate AI response
        logger.info("=== STEP 4: AI Processing ===")
        agent_response = await create_main_agent_response(conversation_history)
        response_text = agent_response.response
        metadata = agent_response.metadata

        # Create assistant message
        logger.info("=== STEP 5: Saving to Backend ===")
        logger.info(f"chat_002: Response len: \033[33m{len(response_text)}\033[0m")
        assistant_message = ChatMessage(
            message_id=generate_message_id(),
            role="assistant",
            text=response_text,
            text_format=user_message.text_format,
            conversation_id=conversation_id,
            metadata=metadata,
            llm_trace=agent_response.llm_trace,
        )

        # Save messages
        await self.message_service.save_messages_to_backend(
            user_message, assistant_message, conversation_id
        )

        return assistant_message
