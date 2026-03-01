"""Models for final response generation phase (Stage 2)."""

from typing import Union
from pydantic import BaseModel, Field
from archie_shared.chat.models import LllmTrace, PipelineTrace, Content
from archie_shared.ui.models import (
    Level2Answer,
    Level3Answer,
    UIAnswer,
    Dashboard,
    Widget,
    LightWidget,
    ClimateWidget,
    FootballWidget,
    MusicWidget,
    DocumentsWidget,
)


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
    reasoning: str = Field(
        description="Step-by-step reasoning for constructing this final response"
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
    pipeline_trace: PipelineTrace | None = Field(
        default=None,
        description="Per-stage timing and LLM cost trace for the full arun() call",
    )


# Format-specific response models for LLM calls
# These are lightweight models sent to LLM instead of full Content schema


class PlainResponse(BaseModel):
    """LLM response model for plain text format"""

    text: str = Field(description="Plain text response without formatting")
    sgr: SGROutput = Field(description="Output reasoning trace")


class Level2Response(BaseModel):
    """LLM response model for level2_answer format"""

    level2_answer: Level2Answer = Field(description="Text with quick action buttons")
    sgr: SGROutput = Field(description="Output reasoning trace")


class Level3Response(BaseModel):
    """LLM response model for level3_answer format"""

    level3_answer: Level3Answer = Field(
        description="Text with widgets and quick actions"
    )
    sgr: SGROutput = Field(description="Output reasoning trace")


class UIResponse(BaseModel):
    """LLM response model for ui_answer format"""

    ui_answer: UIAnswer = Field(description="Full UI elements content")
    sgr: SGROutput = Field(description="Output reasoning trace")


class DashboardResponse(BaseModel):
    """LLM response model for dashboard format"""

    dashboard: Dashboard = Field(description="Dashboard with tiles and quick actions")
    sgr: SGROutput = Field(description="Output reasoning trace")


WidgetType = Union[
    Widget, LightWidget, ClimateWidget, FootballWidget, MusicWidget, DocumentsWidget
]


class WidgetResponse(BaseModel):
    """LLM response model for widget format"""

    widget: WidgetType = Field(description="Standalone widget content")
    sgr: SGROutput = Field(description="Output reasoning trace")


# Mapping from response_format to LLM response model
FORMAT_TO_MODEL: dict[str, type[BaseModel]] = {
    "plain": PlainResponse,
    "voice": PlainResponse,
    "level2_answer": Level2Response,
    "level3_answer": Level3Response,
    "ui_answer": UIResponse,
    "dashboard": DashboardResponse,
    "widget": WidgetResponse,
}


def get_response_model_for_format(response_format: str) -> type[BaseModel]:
    """Get the appropriate LLM response model for the given format."""
    return FORMAT_TO_MODEL.get(response_format, PlainResponse)
