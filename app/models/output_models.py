"""Models for final response generation phase (Stage 2)."""

from typing import TypeAlias
from archie_shared.chat.models import Content, LllmTrace, PipelineTrace
from archie_shared.ui.models import (
    ClimateWidget,
    Dashboard,
    DocumentsWidget,
    FootballWidget,
    Level2Answer,
    Level3Answer,
    LightWidget,
    MusicWidget,
    UIAnswer,
    Widget,
)
from pydantic import BaseModel, Field


class FactCheck(BaseModel):
    statement: str = Field(description="The factual statement in your answer to verify")
    is_correct: bool = Field(
        description="Whether the statement is confirmed or hallucinated"
    )
    evidence: str | None = Field(
        default=None,
        description="Evidence supporting the correctness of the statement (if available)",
    )


class SGROutput(BaseModel):
    """SGR trace for final response generation phase"""

    # reasoning is first so the LLM generates it before the main content,
    # enabling streaming of the reasoning to the frontend.
    reasoning: str = Field(
        description="Step-by-step reasoning for constructing this final response"
    )
    fact_checks: list[FactCheck] = Field(
        description="List of factual statements verified during response generation"
    )
    ui_reasoning: str = Field(
        description="Reasoning behind chosen UI format and components (cards, buttons, tables, etc.)"
    )
    orchestration_summary: str | None = Field(
        default=None,
        description="Brief summary of tools called and orchestration decisions made",
    )


class AgentResponse(BaseModel):
    """Response model for AI agent output"""

    content: Content | None = Field(
        default=None,
        description="Main content response from the AI agent in the specified response format.",
    )
    sgr: SGROutput = Field(description="Output reasoning trace for this final response")
    llm_trace: LllmTrace | None = Field(
        default=None,
        description="LLM usage tracking information (filled by parser, not LLM)",
    )
    response_id: str | None = Field(
        default=None,
        description="OpenAI response ID for conversation continuity (filled by parser)",
    )
    ttft_ms: int | None = Field(
        default=None,
        description="Time to first token in milliseconds (streaming mode only, filled by create_output)",
    )
    pipeline_trace: PipelineTrace | None = Field(
        default=None,
        description="Per-stage timing and LLM cost trace for the full arun() call",
    )


# Format-specific response models for LLM calls
# These are lightweight models sent to LLM instead of full Content schema
#
# IMPORTANT: sgr is always the FIRST field so the LLM generates reasoning
# before the main content payload — enables reasoning streaming to the frontend.


class PlainResponse(BaseModel):
    """LLM response model for plain text format"""

    sgr: SGROutput = Field(description="Output reasoning trace")
    text: str = Field(description="Plain text response without formatting")


class Level2Response(BaseModel):
    """LLM response model for level2_answer format"""

    sgr: SGROutput = Field(description="Output reasoning trace")
    level2_answer: Level2Answer = Field(description="Text with quick action buttons")


class Level3Response(BaseModel):
    """LLM response model for level3_answer format"""

    sgr: SGROutput = Field(description="Output reasoning trace")
    level3_answer: Level3Answer = Field(
        description="Text with widgets and quick actions"
    )


class UIResponse(BaseModel):
    """LLM response model for ui_answer format"""

    sgr: SGROutput = Field(description="Output reasoning trace")
    ui_answer: UIAnswer = Field(description="Full UI elements content")


class DashboardResponse(BaseModel):
    """LLM response model for dashboard format"""

    sgr: SGROutput = Field(description="Output reasoning trace")
    dashboard: Dashboard = Field(description="Dashboard with tiles and quick actions")


WidgetType: TypeAlias = (
    Widget
    | LightWidget
    | ClimateWidget
    | FootballWidget
    | MusicWidget
    | DocumentsWidget
)


class WidgetResponse(BaseModel):
    """LLM response model for widget format"""

    sgr: SGROutput = Field(description="Output reasoning trace")
    widget: WidgetType = Field(description="Standalone widget content")


# Mapping from response_format to LLM response model
FORMAT_TO_MODEL: dict[str, type[BaseModel]] = {
    "plain": PlainResponse,
    "voice": PlainResponse,
    "formatted_text": PlainResponse,
    "level2_answer": Level2Response,
    "level3_answer": Level3Response,
    "ui_answer": UIResponse,
    "dashboard": DashboardResponse,
    "widget": WidgetResponse,
}


def get_response_model_for_format(response_format: str) -> type[BaseModel]:
    """Get the appropriate LLM response model for the given format."""
    return FORMAT_TO_MODEL.get(response_format, PlainResponse)
