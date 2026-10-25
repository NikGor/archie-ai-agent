"""API controller for handling chat requests."""

import functools
import logging
from collections.abc import Awaitable, Callable
from archie_shared.chat.models import ChatMessage, ChatRequest
from app.agents.chat.requests import ChatRunRequest
from .agents.chat.pipeline import ChatPipeline
from .agents.chat.quality.rules import spec_versions
from .config import DEFAULT_MODEL
from .models.ws_models import StatusUpdate, StreamCallback, StreamEventCallback
from .platform.llm.catalog import supports_vision
from .utils.file_utils import build_input_with_files
from .utils.general_utils import generate_message_id
from .utils.tracing import trace_chat_turn


logger = logging.getLogger(__name__)

StatusCallback = Callable[[StatusUpdate], Awaitable[None]] | None


@functools.cache
def get_chat_pipeline() -> ChatPipeline:
    """The chat pipeline of this process, built on first use (ARCHIE-171)."""
    return ChatPipeline()


async def handle_chat(
    user_request: ChatRequest,
    on_status: StatusCallback = None,
    on_stream: StreamCallback = None,
    on_stream_event: StreamEventCallback = None,
    max_command_iterations: int | None = None,
) -> ChatMessage:
    """Handle chat request by processing through AI agent only."""
    logger.info("=== STEP 3: AI Processing ===")
    no_image = bool(user_request.no_image)
    logger.info(
        f"api_controller_001: Processing request with conversation: \033[36m{user_request.conversation_id or 'NONE'}\033[0m, "
        f"Input len: \033[33m{len(user_request.input)}\033[0m, "
        f"demo_mode: \033[35m{user_request.demo_mode}\033[0m, "
        f"no_image: \033[35m{no_image}\033[0m"
    )
    agent_input = await build_input_with_files(user_request.input, user_request.files)
    current_messages = [{"role": "user", "content": agent_input}]
    command_model = user_request.command_model or DEFAULT_MODEL
    final_output_model = user_request.final_output_model or DEFAULT_MODEL
    images = user_request.images or None
    if images and not supports_vision(command_model):
        logger.warning(
            f"api_controller_003: Command model \033[36m{command_model}\033[0m has no vision support, "
            f"dropping \033[33m{len(images)}\033[0m image(s)"
        )
        images = None
    agent_response = await get_chat_pipeline().arun(
        ChatRunRequest(
            messages=current_messages,
            command_model=command_model,
            final_output_model=final_output_model,
            command_effort=user_request.command_effort,
            final_output_effort=user_request.final_output_effort,
            response_format=user_request.response_format,
            previous_response_id=user_request.previous_message_id,
            chat_history=user_request.chat_history,
            user_name=user_request.user_name,
            conversation_id=user_request.conversation_id,
            no_image=no_image,
            images=images,
            max_command_iterations=max_command_iterations,
            on_status=on_status,
            on_stream=on_stream,
            on_stream_event=on_stream_event,
            demo_mode=bool(user_request.demo_mode),
        )
    )

    # Create response message
    assistant_message = ChatMessage(
        message_id=agent_response.response_id or generate_message_id(),
        role="assistant",
        content=agent_response.content,
        # Stored with the message so monitoring judges it by the right format spec
        response_format=agent_response.response_format or user_request.response_format,
        conversation_id=user_request.conversation_id,
        previous_message_id=user_request.previous_message_id,
        model=f"{command_model}/{final_output_model}",
        llm_trace=agent_response.llm_trace,
        pipeline_trace=agent_response.pipeline_trace,
    )

    content_text = str(assistant_message.content) if assistant_message.content else ""
    logger.info(
        f"api_controller_002: Generated response length: \033[33m{len(content_text)}\033[0m"
    )
    logger.info("=== STEP 4: Response Ready ===")

    return assistant_message


def _chat_spec_versions(response_format: str) -> dict[str, str]:
    return spec_versions(response_format, trace_found=True)


# One Langfuse trace per chat turn, with the spec versions that produced it
handle_chat = trace_chat_turn(handle_chat, spec_versions=_chat_spec_versions)
