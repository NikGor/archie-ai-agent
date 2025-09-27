import logging
from typing import List
from fastapi import APIRouter, HTTPException, Query
from .models import ChatMessage, Conversation, ChatRequest
from .api_controller import handle_chat
from .services.backend import get_database

logger = logging.getLogger(__name__)
router = APIRouter()
db = get_database()

@router.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatMessage:
    """Chat endpoint for handling user messages."""
    try:
        logger.info(f"Received chat request for conversation: {request.conversation_id or 'new conversation'}")
        
        # Convert ChatRequest to ChatMessage
        chat_message = ChatMessage(
            role=request.role,
            text=request.text,
            text_format=request.text_format,
            conversation_id=request.conversation_id
        )
        
        result = await handle_chat(chat_message)
        logger.info("Chat request processed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/conversations")
async def get_conversations(limit: int = Query(50, description="Maximum number of conversations to return")) -> List[Conversation]:
    """Get all conversations."""
    try:
        logger.info(f"Fetching conversations with limit: {limit}")
        conversations = db.get_all_conversations(limit=limit)
        logger.info(f"Successfully retrieved {len(conversations)} conversations")
        return conversations
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> Conversation:
    """Get a complete conversation with all its messages."""
    try:
        logger.info(f"Fetching complete conversation: {conversation_id}")
        conversation = db.get_conversation_with_messages(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
        logger.info(f"Successfully retrieved conversation {conversation_id} with {len(conversation.messages)} messages")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
