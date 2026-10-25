from typing import Literal
from pydantic import BaseModel, Field
from ..platform.tools.models import Parameter, ToolCall, ToolName
from ..utils.intent_config import IntentType


class ActionType(BaseModel):
    type: Literal["function_call", "parameters_request", "final_response"]
    reasoning: str = Field(description="Why this action type was chosen")


class ToolCallRequest(BaseModel):
    tool_name: ToolName = Field(description="Name of the tool to call")
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
    """SGR trace for orchestration and decision-making phase"""

    action: ActionType
    tool_calls: list[ToolCallRequest] = Field(
        description="Tools to execute (can be parallel)",
    )
    reasoning: str = Field(
        description="Step-by-step reasoning for this orchestration decision"
    )
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
