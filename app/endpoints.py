import logging
from archie_shared.chat.models import ChatMessage, ChatRequest, Content
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
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
    except HTTPException:
        raise
    except ValidationError as e:
        logger.exception("endpoints_error_002: \033[31mValidation error\033[0m")
        return ChatMessage(
            message_id=generate_message_id(),
            role="assistant",
            content=Content(content_format="plain", text=f"Validation error: {e!s}"),
            conversation_id=request.conversation_id,
        )
    except Exception as e:
        logger.exception(f"endpoints_error_001: \033[31m{e!s}\033[0m")
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

        async def send_stream(text: str) -> None:
            await websocket.send_json({"type": "stream_text", "text": text})

        async def send_stream_event(event_type: str, text: str | None = None) -> None:
            payload: dict = {"type": event_type}
            if text is not None:
                payload["text"] = text
            await websocket.send_json(payload)

        result = await handle_chat(
            request,
            on_status=send_status,
            on_stream=send_stream,
            on_stream_event=send_stream_event,
        )
        await websocket.send_json({"type": "stream_complete"})
        await websocket.send_json(
            {
                "type": "final",
                "data": result.model_dump(mode="json"),
            }
        )
        logger.info("ws_chat_003: Final response sent")
    except WebSocketDisconnect:
        logger.info("ws_chat_004: Client disconnected")
    except ValidationError as e:
        logger.exception("ws_chat_error_002: \033[31mValidation error\033[0m")
        await websocket.send_json(
            {"type": "error", "message": f"Validation error: {e!s}"}
        )
    except Exception as e:
        logger.exception(f"ws_chat_error_001: \033[31m{e!s}\033[0m")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()
