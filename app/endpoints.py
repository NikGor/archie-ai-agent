from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel
from .api_controller import handle_chat

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return await handle_chat(request.message, request.conversation_id or "default")
