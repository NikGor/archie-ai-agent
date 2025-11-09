"""Chat service for coordinating chat workflow."""

import logging
from archie_shared.chat.models import ChatMessage, ChatRequest, Content
from ..agent.builder import create_main_agent_response
from ..utils.backend_client import BackendClient
from ..utils.general_utils import generate_message_id
from .conversation_service import ConversationService
from .message_service import MessageService


logger = logging.getLogger(__name__)


class ChatService:
    """Service for coordinating chat workflow."""

    def __init__(self):
        self.conversation_service = ConversationService()
        self.message_service = MessageService()
        self.backend_client = BackendClient()

    async def process_chat_message(self, user_request: ChatRequest) -> ChatMessage:
        """Process a chat request through the complete workflow."""
        logger.info("=== STEP 3: Message Processing ===")
        logger.info(
            f"chat_001: processing conversation: \033[36m{user_request.conversation_id or 'NEW'}\033[0m"
        )

        # Get or create conversation
        conversation_id = user_request.conversation_id
        if not conversation_id:
            conversation_id = await self.conversation_service.create_new_conversation()

        # Create user message from request — wrap raw input into Content model
        user_content = Content(
            content_format="plain",
            text=user_request.input,
        )

        user_message = ChatMessage(
            message_id=generate_message_id(),
            role="user",
            content=user_content,
            conversation_id=conversation_id,
            previous_message_id=user_request.previous_message_id,
            model=user_request.model,
        )

        # Create simple message for OpenAI
        current_messages = [{"role": "user", "content": user_request.input}]

        # Generate AI response
        logger.info("=== STEP 4: AI Processing ===")
        # Use model from user request if provided, otherwise default
        model = user_request.model if user_request.model else "gpt-4.1"
        # Pass previous_message_id from user request to OpenAI for context
        agent_response = await create_main_agent_response(
            current_messages,
            user_request.previous_message_id,
            model,
            user_request.response_format,
        )

        # Create assistant message - AgentResponse.content уже Content object
        logger.info("=== STEP 5: Saving to Database ===")

        assistant_message = ChatMessage(
            message_id=agent_response.response_id or generate_message_id(),
            role="assistant",
            content=agent_response.content,  # Используем Content object напрямую
            conversation_id=conversation_id,
            previous_message_id=user_message.message_id,
            model=model,
            llm_trace=agent_response.llm_trace,
        )

        # Log response length
        content_text = (
            str(assistant_message.content) if assistant_message.content else ""
        )
        logger.info(f"chat_002: Response len: \033[33m{len(content_text)}\033[0m")

        # Save messages
        await self.message_service.save_messages_to_database(
            user_message, assistant_message, conversation_id
        )

        return assistant_message
