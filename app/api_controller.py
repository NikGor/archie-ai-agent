"""API controller for handling chat requests."""

import logging
from archie_shared.chat.models import ChatMessage, ChatRequest
from .agent.agent_factory import AgentFactory
from .utils.general_utils import generate_message_id


logger = logging.getLogger(__name__)


async def handle_chat(user_request: ChatRequest) -> ChatMessage:
    """Handle chat request by processing through AI agent only."""
    logger.info("=== STEP 3: AI Processing ===")
    logger.info(
        f"api_controller_001: Processing request with conversation: \033[36m{user_request.conversation_id or 'NONE'}\033[0m, "
        f"Input len: \033[33m{len(user_request.input)}\033[0m"
    )
    
    # Prepare messages for AI agent
    current_messages = [{"role": "user", "content": user_request.input}]
    model = user_request.model if user_request.model else "gpt-4.1"
    
    # Generate AI response
    agent_factory = AgentFactory()
    agent_response = await agent_factory.create_agent_response(
        messages=current_messages,
        model=model,
        response_format=user_request.response_format,
        previous_response_id=user_request.previous_message_id,
    )
    
    # Create response message
    assistant_message = ChatMessage(
        message_id=agent_response.response_id or generate_message_id(),
        role="assistant",
        content=agent_response.content,
        conversation_id=user_request.conversation_id,
        previous_message_id=user_request.previous_message_id,
        model=model,
        llm_trace=agent_response.llm_trace,
    )
    
    content_text = str(assistant_message.content) if assistant_message.content else ""
    logger.info(
        f"api_controller_002: Generated response length: \033[33m{len(content_text)}\033[0m"
    )
    logger.info("=== STEP 4: Response Ready ===")
    
    return assistant_message
