"""Internal chat requests; callbacks and per-run facts stay out of the pipeline singleton."""

from archie_shared.chat.models import ImageAttachment, ReasoningEffort
from pydantic import BaseModel, ConfigDict, Field
from app import config
from app.config import DEFAULT_MODEL
from app.models.ws_models import StatusCallback, StreamCallback, StreamEventCallback
from app.platform.tools.models import ToolResult


class ChatRunRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[dict[str, str]] = Field(..., description="messages")
    command_model: str = Field(DEFAULT_MODEL, description="command model")
    final_output_model: str = Field(DEFAULT_MODEL, description="final output model")
    command_effort: ReasoningEffort | None = Field(None, description="command effort")
    final_output_effort: ReasoningEffort | None = Field(
        None, description="final output effort"
    )
    response_format: str = Field("plain", description="response format")
    previous_response_id: str | None = Field(None, description="previous response id")
    chat_history: str | None = Field(None, description="chat history")
    user_name: str | None = Field(None, description="user name")
    conversation_id: str | None = Field(None, description="conversation id")
    no_image: bool = Field(False, description="no image")
    images: list[ImageAttachment] | None = Field(None, description="images")
    max_command_iterations: int | None = Field(
        None, description="max command iterations"
    )
    on_status: StatusCallback = Field(None, description="on status")
    on_stream: StreamCallback = Field(None, description="on stream")
    on_stream_event: StreamEventCallback = Field(None, description="on stream event")
    demo_mode: bool = Field(False, description="demo mode")


class OutputRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_input: str = Field(..., description="user input")
    command_summary: str = Field(..., description="command summary")
    tool_results: list[ToolResult] | None = Field(None, description="tool results")
    response_format: str = Field("plain", description="response format")
    model: str = Field(config.DEFAULT_MODEL, description="model")
    state: dict | None = Field(None, description="state")
    previous_response_id: str | None = Field(None, description="previous response id")
    chat_history: str | None = Field(None, description="chat history")
    intents: list[str] | None = Field(None, description="intents")
    no_image: bool = Field(False, description="no image")
    on_stream: StreamCallback = Field(None, description="on stream")
    on_stream_event: StreamEventCallback = Field(None, description="on stream event")
    reasoning_effort: ReasoningEffort | None = Field(
        None, description="reasoning effort"
    )
