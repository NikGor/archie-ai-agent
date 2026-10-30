"""Registered result schemas of AgentSpec runs (ARCHIE-320).

A spec names its output schema by name and version (`findings_v1`); it never
carries a JSON schema or code. Agent kinds may register their own schemas.
"""

from pydantic import BaseModel, ConfigDict, Field
from app.platform.tools.models import Parameter, ToolCall


class CriterionCheck(BaseModel):
    criterion: str = Field(description="One completion criterion of the task")
    met: bool = Field(description="True only when the evidence shows it is met")
    evidence: str = Field(description="Short fact that supports the answer")


class ProposedAction(BaseModel):
    tool_name: str = Field(description="Tool with a side effect the user must approve")
    arguments: list[Parameter] = Field(description="Arguments with their values")
    reason: str = Field(description="Why this action reaches the goal")


class AgentReply(BaseModel):
    """Common fields of every registered agent reply."""

    summary: str = Field(
        description="Result for the parent: what you found and the evidence gaps. Never claim an action was executed."
    )
    criteria: list[CriterionCheck] = Field(
        description="Each completion criterion with met true or false"
    )
    proposed_actions: list[ProposedAction] = Field(
        description="Side effects for the user to approve; empty when none"
    )


class FindingsReply(AgentReply):
    findings: list[str] = Field(
        description="Facts supported by the tool results; mark unknown causes"
    )


class AnswerReply(AgentReply):
    answer: str = Field(description="Direct answer to the task")
    sources: list[str] = Field(description="Sources or tool results the answer uses")


class FindingsTurn(FindingsReply):
    tool_calls: list[ToolCall] = Field(
        max_length=4, description="Tools needed to finish the task; empty when done"
    )


class AnswerTurn(AnswerReply):
    tool_calls: list[ToolCall] = Field(
        max_length=4, description="Tools needed to finish the task; empty when done"
    )


class ResultSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Name and version, e.g. findings_v1")
    description: str = Field(description="When a configurator picks this schema")
    reply: type[BaseModel] = Field(description="Final reply without tool calls")
    turn: type[BaseModel] = Field(description="Reply with tool calls")
    configurable: bool = Field(
        default=True,
        description="A configurator may pick it; presets register False",
    )


RESULT_SCHEMAS: dict[str, ResultSchema] = {}


def register_result_schema(schema: ResultSchema) -> None:
    RESULT_SCHEMAS[schema.name] = schema


def result_schema(name: str) -> ResultSchema:
    return RESULT_SCHEMAS[name]


register_result_schema(
    ResultSchema(
        name="findings_v1",
        description="Diagnosis or research: summary and a list of supported findings",
        reply=FindingsReply,
        turn=FindingsTurn,
    )
)
register_result_schema(
    ResultSchema(
        name="answer_v1",
        description="One direct answer with its sources",
        reply=AnswerReply,
        turn=AnswerTurn,
    )
)
