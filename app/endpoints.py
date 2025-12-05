import logging
from archie_shared.chat.models import ChatMessage, ChatRequest, Content
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .api_controller import handle_chat
from .models.ws_models import StatusUpdate
from .utils.general_utils import generate_message_id


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatMessage:
    """Chat endpoint for handling user messages."""
    try:
        logger.info("=== STEP 2: Chat Request ===")
        logger.info(
            f"endpoints_001: processing conversation: \033[36m{request.conversation_id or 'NEW'}\033[0m, "
            f"Len: \033[33m{len(request.input)}\033[0m"
        )
        if request.previous_message_id:
            logger.info(
                f"endpoints_003: Using previous_message_id: \033[36m{request.previous_message_id}\033[0m"
            )
        result = await handle_chat(request)
        logger.info("=== STEP 5: Response Ready ===")
        content_text = str(result.content) if result.content else ""
        logger.info(f"endpoints_002: Response len: \033[33m{len(content_text)}\033[0m")
        return result
    except Exception as e:
        logger.error(f"endpoints_error_001: \033[31m{e!s}\033[0m", exc_info=True)
        return ChatMessage(
            message_id=generate_message_id(),
            role="assistant",
            content=Content(content_format="plain", text=f"Error: {e!s}"),
            conversation_id=request.conversation_id,
        )


@router.websocket("/ws_chat")
async def ws_chat(websocket: WebSocket):
    """WebSocket endpoint for chat with real-time status updates."""
    await websocket.accept()
    logger.info("ws_chat_001: WebSocket connection accepted")
    try:
        data = await websocket.receive_json()
        logger.info(
            f"ws_chat_002: Received request, input len: \033[33m{len(data.get('input', ''))}\033[0m"
        )
        request = ChatRequest(**data)

        async def send_status(status: StatusUpdate):
            await websocket.send_json({"type": "status", **status.model_dump()})

        result = await handle_chat(request, on_status=send_status)
        await websocket.send_json(
            {
                "type": "final",
                "data": result.model_dump(mode="json"),
            }
        )
        logger.info("ws_chat_003: Final response sent")
    except WebSocketDisconnect:
        logger.info("ws_chat_004: Client disconnected")
    except Exception as e:
        logger.error(f"ws_chat_error_001: \033[31m{e!s}\033[0m", exc_info=True)
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()
