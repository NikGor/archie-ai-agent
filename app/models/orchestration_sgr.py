from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from ..platform.tools.models import Parameter, ToolCall
from ..utils.intent_config import IntentType


# Tools the chat may call. Swarm-only tools stay out of the Stage 1 schema, so
# the LLM cannot pick a tool it has no description of (ARCHIE-305). Kept equal
# to the non-exclusive groups of the tool registry by a drift test.
ChatToolName = Literal[
    "smarthome_control_tool",
    "smarthome_status_tool",
    "spotify_tool",
    "tv_tool",
    "google_search_tool",
    "google_places_search_tool",
    "task_tool",
    "notes_tool",
    "events_tool",
    "gmail_tool",
    "cron_tool",
    "announce_tool",
    "awtrix_tool",
    "agent_task_tool",
    "document_search_tool",
    "skill_loader_tool",
]


class ActionType(BaseModel):
    type: Literal["function_call", "parameters_request", "final_response"]
    # Schema name "rationale": Claude 5.5 refuses a "reasoning" field as output
    # extraction (content_filter); Python keeps .reasoning
    model_config = ConfigDict(populate_by_name=True)

    reasoning: str = Field(alias="rationale", description="Why this action type was chosen")


class ToolCallRequest(BaseModel):
    tool_name: ChatToolName = Field(description="Name of the tool to call")
    arguments: list[Parameter] = Field(
        description="Arguments to pass to the tool with their values, which can be found from context",
    )
    missing_parameters: list[str] = Field(
        description="Parameters that still need to be requested from user",
    )
    is_confirmed: bool = Field(
        default=False,
        description="Has the user explicitly confirmed the tool call with current parameters? Even when all parameters are present, always ask the user for additional confirmation before function_call. Exceptions: smarthome_control_tool for light/climate and task_tool need no confirmation — set true directly.",
    )
    reason: str = Field(description="Why this tool is needed")

    def to_call(self) -> ToolCall:
        """The neutral call the executor runs (ARCHIE-292)."""
        return ToolCall(
            tool_name=self.tool_name, arguments=self.arguments, reason=self.reason
        )


class SGROrchestration(BaseModel):
    """Decision of the command stage"""

    model_config = ConfigDict(populate_by_name=True)

    action: ActionType
    tool_calls: list[ToolCallRequest] = Field(
        description="Tools to execute (can be parallel)",
    )
    reasoning: str = Field(alias="rationale", description="Why this decision was made")
    intents: list[IntentType] = Field(
        default_factory=list,
        description=(
            "UI intents for response_format=ui_answer. "
            "Controls which specialized UI components are available in Stage 3. "
            "Leave empty for base components only (TextAnswer, Card, CardGrid, Table, Image)."
        ),
    )


class DecisionResponse(BaseModel):
    """Response from orchestration/decision-making LLM call"""

    sgr: SGROrchestration = Field(description="Orchestration reasoning trace")
    image_description: str = Field(
        default="",
        description=(
            "Only when the user attached images: a concise factual description of what "
            "the images show (text, objects, numbers) relevant to the request. "
            "Passed to the final answer stage, which cannot see the images. Empty otherwise."
        ),
    )
    handover_context: str = Field(
        description="results obtained from tool calls, missing parameters or information from internal knowledge to prepare the final answer"
    )
