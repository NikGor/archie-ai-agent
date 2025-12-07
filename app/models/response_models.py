from typing import Literal
from pydantic import BaseModel, Field
from archie_shared.chat.models import LllmTrace, Content


class SourceRef(BaseModel):
    id: int = Field(description="Local incremental id for this session (1..N)")
    url: str = Field(description="Source URL")
    title: str | None = Field(
        default=None,
        description="Page/article title",
    )
    snippet: str | None = Field(
        default=None,
        description="Short relevant excerpt",
    )


class EvidenceItem(BaseModel):
    claim: str = Field(
        description="Concrete factual claim used in the response/metadata"
    )
    support: Literal["supported", "contradicted", "uncertain"] = Field(
        description="Does the cited evidence support the claim?"
    )
    source_ids: list[int] = Field(
        description="IDs from sources[] backing this claim (empty if uncertain)"
    )


class RoutingDecision(BaseModel):
    intent: Literal[
        "answer_general",
        "weather",
        "sports_score",
        "web_search",
        "clarify",
        "out_of_scope",
    ] = Field(description="Chosen path for this turn")
    rationale: str = Field(description="One-sentence reason for choosing this intent")


class SlotsStatus(BaseModel):
    needed: list[str] = Field(
        default=[],
        description="Required fields for this intent",
    )
    filled: list[str] = Field(
        default=[],
        description="Fields that have been filled",
    )
    pending: list[str] = Field(
        default=[],
        description="Still missing; ask one-by-one",
    )


class PreActionChecklist(BaseModel):
    summary: str = Field(description="Short summary of the planned action/assumption")
    decision: Literal["confirm", "clarify", "reject", "none"] = Field(
        description="If an action is pending, ask for confirmation; 'none' if N/A"
    )


class VerificationStatus(BaseModel):
    level: Literal["verified", "partially_verified", "unverified"] = Field(
        description="Overall verification level for factual content"
    )
    confidence_pct: int = Field(
        ge=0,
        le=100,
        description="Calibrated confidence 0..100",
    )


class SGRTrace(BaseModel):
    """Schema-Guided Reasoning trace (not user-facing UI)"""

    routing: RoutingDecision
    slots: SlotsStatus = Field(default_factory=SlotsStatus)
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Claims and how they are supported",
    )
    sources: list[SourceRef] = Field(
        default_factory=list,
        description="Deduplicated list of sources used",
    )
    verification: VerificationStatus
    pre_action: PreActionChecklist
    reasoning: str = Field(
        description="Detailed reasoning steps taken to arrive at this response"
    )
    ui_reasoning: str = Field(
        description="Reasoning behind the chosen UI format and components"
    )

class AgentResponse(BaseModel):
    """Response model for AI agent output"""

    content: Content | None = Field(
        default=None,
        description="Main content response from the AI agent in the specified response format.",
    )
    sgr: SGRTrace = Field(
        description="Mandatory SGR reasoning trace for this turn (internal; not to be shown to user as-is)"
    )
    llm_trace: LllmTrace = Field(description="LLM usage tracking information")
    response_id: str | None = Field(
        default=None, description="OpenAI response ID for conversation continuity"
    )
