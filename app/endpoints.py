import logging
from fastapi import APIRouter, HTTPException
from .models import ChatMessage
from .api_controller import handle_chat

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/chat")
async def chat_endpoint(request: ChatMessage) -> ChatMessage:
    """Chat endpoint for handling user messages."""
    try:
        logger.info(f"Received chat request for conversation: {request.conversation_id or 'default'}")
        result = await handle_chat(request)
        logger.info("Chat request processed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
