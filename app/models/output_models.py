"""Models for final response generation phase (Stage 2)."""

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


class VerificationStatus(BaseModel):
    level: Literal["verified", "partially_verified", "unverified"] = Field(
        description="Overall verification level for factual content"
    )
    confidence_pct: int = Field(
        ge=0,
        le=100,
        description="Calibrated confidence 0..100",
    )


class SGROutput(BaseModel):
    """SGR trace for final response generation phase"""

    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Claims and how they are supported by sources",
    )
    sources: list[SourceRef] = Field(
        default_factory=list,
        description="Deduplicated list of sources used in response",
    )
    verification: VerificationStatus
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
    sgr: SGROutput = Field(
        description="Output reasoning trace for this final response"
    )
    llm_trace: LllmTrace | None = Field(
        default=None,
        description="LLM usage tracking information (filled by parser, not LLM)",
    )
    response_id: str | None = Field(
        default=None,
        description="OpenAI response ID for conversation continuity (filled by parser)",
    )
