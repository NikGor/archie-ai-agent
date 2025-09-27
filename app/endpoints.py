from fastapi import APIRouter
from pydantic import BaseModel
from .api_controller import handle_chat

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return await handle_chat(request.message)
