import logging

from fastapi import APIRouter, HTTPException

from .api_controller import handle_chat
from .models import ChatMessage, ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatMessage:
    """Chat endpoint for handling user messages."""
    try:
        logger.info(
            f"Received chat request for conversation: {request.conversation_id or 'new conversation'}"
        )

        # Convert ChatRequest to ChatMessage
        chat_message = ChatMessage(
            role=request.role,
            text=request.text,
            text_format=request.text_format,
            conversation_id=request.conversation_id,
        )

        result = await handle_chat(chat_message)
        logger.info("Chat request processed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e!s}")
