"""API controller for handling chat requests."""

import logging
from archie_shared.chat.models import ChatMessage, ChatRequest, Content
from .agent.agent_factory import AgentFactory
from .utils.backend_client import BackendClient
from .utils.general_utils import generate_message_id


logger = logging.getLogger(__name__)


async def handle_chat(user_request: ChatRequest) -> ChatMessage:
    """Handle chat request by orchestrating agent and backend interactions."""
    logger.info("=== STEP 3: Message Processing ===")
    logger.info(
        f"api_controller_001: Processing conversation: \033[36m{user_request.conversation_id or 'NEW'}\033[0m"
    )
    backend_client = BackendClient()
    conversation_id = user_request.conversation_id
    if not conversation_id:
        logger.info("api_controller_002: Creating new conversation")
        conversation_id = await backend_client.create_conversation()
        logger.info(
            f"api_controller_003: Created conversation: \033[32m{conversation_id}\033[0m"
        )
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
    logger.info(
        f"api_controller_004: User message ID: \033[36m{user_message.message_id}\033[0m"
    )
    current_messages = [{"role": "user", "content": user_request.input}]
    logger.info("=== STEP 4: AI Processing ===")
    model = user_request.model if user_request.model else "gpt-4.1"
    agent_factory = AgentFactory()
    agent_response = await agent_factory.create_agent_response(
        messages=current_messages,
        model=model,
        response_format=user_request.response_format,
        previous_response_id=user_request.previous_message_id,
    )
    logger.info("=== STEP 5: Saving to Database ===")
    assistant_message = ChatMessage(
        message_id=agent_response.response_id or generate_message_id(),
        role="assistant",
        content=agent_response.content,
        conversation_id=conversation_id,
        previous_message_id=user_message.message_id,
        model=model,
        llm_trace=agent_response.llm_trace,
    )
    content_text = str(assistant_message.content) if assistant_message.content else ""
    logger.info(
        f"api_controller_005: Assistant response length: \033[33m{len(content_text)}\033[0m"
    )
    logger.info("api_controller_006: Saving user message")
    await backend_client.create_message(user_message)
    logger.info("api_controller_007: Saving assistant message")
    await backend_client.create_message(assistant_message)
    logger.info("api_controller_008: Messages saved successfully")
    return assistant_message
