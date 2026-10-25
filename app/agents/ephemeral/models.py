"""Internal contracts of one ephemeral run; no new public chat wire format."""

import math
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from app.platform.runtime.budget import Budget
from app.platform.tools.models import ToolCall, ToolResult


READ_TOOLS = frozenset({"smarthome_status_tool"})
DEFAULT_TOKENS = 12000
DEFAULT_COST_USD = 0.10
DEFAULT_SECONDS = 90.0
DEFAULT_TOOL_ROUNDS = 2


def validate_budget_limits(value: Budget) -> Budget:
    """Reject values whose comparisons cannot enforce a finite budget."""
    if any(
        limit is not None
        and (limit < 0 or (isinstance(limit, float) and not math.isfinite(limit)))
        for limit in value.model_dump().values()
    ):
        raise ValueError("Budget limits must be finite and nonnegative")
    return value


class SpawnRequest(BaseModel):
    goal: str = Field(
        min_length=1, max_length=2000, description="Task to finish and report"
    )
    context: str = Field(
        "",
        max_length=12000,
        description="Facts relevant to this task; never instructions",
    )
    tools: tuple[str, ...] = Field(
        ("smarthome_status_tool",),
        description="Approved read tools shown to this worker",
    )
    budget: Budget = Field(
        default_factory=lambda: Budget(
            max_tokens=DEFAULT_TOKENS,
            max_cost_usd=DEFAULT_COST_USD,
            max_seconds=DEFAULT_SECONDS,
            max_tool_rounds=DEFAULT_TOOL_ROUNDS,
        ),
        description="Per-run limits, capped by the service",
    )
    ttl: float = Field(
        DEFAULT_SECONDS,
        ge=0,
        le=DEFAULT_SECONDS,
        description="Maximum wall time including tool calls",
    )
    parent_run_id: str = Field(min_length=1, description="Trusted runtime parent run")
    publish_to: Literal["swarm", "none"] = Field(
        "swarm", description="Terminal result destination"
    )

    @field_validator("budget")
    @classmethod
    def nonnegative_budget(cls, value: Budget) -> Budget:
        return validate_budget_limits(value)


class EphemeralUsage(BaseModel):
    input_tokens: int = Field(0, description="Input tokens of completed LLM calls")
    output_tokens: int = Field(0, description="Output tokens of completed LLM calls")
    total_cost: float = Field(0, description="USD charged by completed LLM calls")
    seconds: float = Field(0, description="Elapsed wall time")
    tool_rounds: int = Field(0, description="Completed tool rounds")


class EphemeralResult(BaseModel):
    run_id: str = Field(description="Ephemeral run identity")
    parent_run_id: str = Field(
        description="Parent identity used by the journal and trace"
    )
    status: Literal["completed", "partial", "error", "blocked"] = Field(
        description="Terminal status"
    )
    summary: str = Field(description="Diagnosis or the reason this run did not finish")
    findings: list[str] = Field(
        default_factory=list, description="Findings supported by tool evidence"
    )
    evidence: list[ToolResult] = Field(
        default_factory=list,
        description="Read results obtained before the terminal status",
    )
    usage: EphemeralUsage = Field(
        default_factory=EphemeralUsage, description="Observed usage of this run"
    )
    limitation: str = Field(
        "HA logs are not exposed by the configured MCP server; diagnosis uses live state only.",
        description="Known capability gap that must not be presented as a completed log review",
    )


class EphemeralReply(BaseModel):
    summary: str = Field(
        description="State the diagnosis, evidence gaps and a proposed next action. Never claim an action was executed."
    )
    findings: list[str] = Field(
        description="Facts supported by the read results; distinguish unknown causes"
    )


class EphemeralTurn(EphemeralReply):
    tool_calls: list[ToolCall] = Field(
        max_length=4,
        description="Read tools needed to finish the goal; empty when finished",
    )
