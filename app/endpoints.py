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
        logger.info("=== STEP 2: Chat Request ===")
        logger.info(
            f"endpoints_001: processing conversation: \033[36m{request.conversation_id or 'NEW'}\033[0m, "
            f"Len: \033[33m{len(request.text)}\033[0m"
        )
        chat_message = ChatMessage(
            role=request.role,
            text=request.text,
            text_format=request.text_format,
            conversation_id=request.conversation_id,
        )
        result = await handle_chat(chat_message)
        logger.info("=== STEP 6: Response Ready ===")
        logger.info(f"endpoints_002: Response len: \033[33m{len(result.text)}\033[0m")
        return result
    except Exception as e:
        logger.error(f"endpoints_error_001: \033[31m{e!s}\033[0m", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {e!s}",
        )
