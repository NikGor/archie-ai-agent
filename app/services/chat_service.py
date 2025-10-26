"""Chat service for coordinating chat workflow."""

import logging
from archie_shared.chat.models import ChatMessage
from ..agent.builder import create_main_agent_response
from ..utils.general_utils import generate_message_id
from ..utils.backend_client import BackendClient
from .conversation_service import ConversationService
from .message_service import MessageService

logger = logging.getLogger(__name__)


class ChatService:
    """Service for coordinating chat workflow."""

    def __init__(self):
        self.conversation_service = ConversationService()
        self.message_service = MessageService()
        self.backend_client = BackendClient()

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

        # Prepare user message  
        user_message.conversation_id = conversation_id
        
        # Set user message ID
        if not user_message.message_id:
            user_message.message_id = generate_message_id()
        
        # Create simple message for OpenAI
        current_messages = [{"role": "user", "content": user_message.text}]

        # Generate AI response  
        logger.info("=== STEP 4: AI Processing ===")
        # Use model from user request if provided, otherwise default
        model = user_message.model if user_message.model else "gpt-4.1"
        # Pass previous_message_id from user request to OpenAI for context
        agent_response = await create_main_agent_response(current_messages, user_message.previous_message_id, model)
        response_text = agent_response.response
        metadata = agent_response.metadata

        # Create assistant message
        logger.info("=== STEP 5: Saving to Database ===")
        logger.info(f"chat_002: Response len: \033[33m{len(response_text)}\033[0m")
        
        assistant_message = ChatMessage(
            message_id=agent_response.response_id or generate_message_id(),
            role="assistant",
            text=response_text,
            text_format=user_message.text_format,
            conversation_id=conversation_id,
            metadata=metadata,
            llm_trace=agent_response.llm_trace,
        )

        # Save messages
        await self.message_service.save_messages_to_database(
            user_message, assistant_message, conversation_id
        )

        return assistant_message
