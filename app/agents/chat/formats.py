"""Chat format policy, preserving the legacy fallbacks (ARCHIE-170)."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from app.models.output_models import (
    DashboardResponse,
    Level2Response,
    Level3Response,
    PlainResponse,
    UIResponse,
    WidgetResponse,
)


class FormatSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Requested format name")
    prompt_spec: str = Field(description="Compiled format prompt")
    output_model: type[BaseModel] = Field(description="Stage 3 response model")
    stream_mode: Literal["none", "text", "level2", "ui"] = Field(
        "none", description="Streaming path"
    )
    skips_command_loop: bool = Field(False, description="Direct Stage 3 output")
    tool_groups: tuple[str, ...] | None = Field(
        None, description="Allowed nonexclusive tool groups; None means all"
    )
    content_format: str = Field("plain", description="Content discriminator")
    content_field: str = Field("text", description="Parsed payload field")


FORMATS = {
    "plain": FormatSpec(
        name="plain",
        prompt_spec="format_plain",
        output_model=PlainResponse,
        stream_mode="text",
    ),
    "voice": FormatSpec(
        name="voice",
        prompt_spec="format_plain",
        output_model=PlainResponse,
        stream_mode="text",
    ),
    "gemini_tts": FormatSpec(
        name="gemini_tts", prompt_spec="format_gemini_tts", output_model=PlainResponse
    ),
    "formatted_text": FormatSpec(
        name="formatted_text",
        prompt_spec="format_formatted_text",
        output_model=PlainResponse,
        stream_mode="text",
    ),
    "level2_answer": FormatSpec(
        name="level2_answer",
        prompt_spec="format_level2_answer",
        output_model=Level2Response,
        stream_mode="level2",
        content_format="level2_answer",
        content_field="level2_answer",
    ),
    "level3_answer": FormatSpec(
        name="level3_answer",
        prompt_spec="format_level3_answer",
        output_model=Level3Response,
        stream_mode="ui",
        content_format="level3_answer",
        content_field="level3_answer",
    ),
    "ui_answer": FormatSpec(
        name="ui_answer",
        prompt_spec="format_ui_answer",
        output_model=UIResponse,
        stream_mode="ui",
        content_format="ui_answer",
        content_field="ui_answer",
    ),
    "dashboard": FormatSpec(
        name="dashboard",
        prompt_spec="format_dashboard",
        output_model=DashboardResponse,
        stream_mode="ui",
        skips_command_loop=True,
        tool_groups=("smarthome",),
        content_format="dashboard",
        content_field="dashboard",
    ),
    "widget": FormatSpec(
        name="widget",
        prompt_spec="format_widget",
        output_model=WidgetResponse,
        stream_mode="ui",
        skips_command_loop=True,
        tool_groups=("smarthome",),
        content_format="widget",
        content_field="widget",
    ),
}


def resolve_format(name: str) -> FormatSpec:
    """Unknown formats use formatted instructions, plain payload and no stream."""
    return FORMATS.get(name) or FormatSpec(
        name=name, prompt_spec="format_formatted_text", output_model=PlainResponse
    )
