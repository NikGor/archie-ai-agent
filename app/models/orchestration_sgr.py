from typing import List, Literal
from pydantic import BaseModel, Field


ToolName = Literal[
    "light_control_tool",
    "climate_control_tool",
    "spotify_tool",
    "google_search_tool",
    "football_tool",
    "task_tool",
    "notes_tool",
    "events_tool",
    "document_search_tool",
]


class ActionType(BaseModel):
    type: Literal["function_call", "parameters_request", "final_response"]
    reasoning: str = Field(description="Why this action type was chosen")


class Parameter(BaseModel):
    name: str = Field(description="Parameter name")
    value: str = Field(description="Parameter value")


class ToolCallRequest(BaseModel):
    tool_name: ToolName = Field(description="Name of the tool to call")
    arguments: List[Parameter] = Field(
        description="Arguments to pass to the tool with their values, which can be found from context",
    )
    missing_parameters: List[str] = Field(
        description="Parameters that still need to be requested from user",
    )
    is_confirmed: bool = Field(
        default=False,
        description="Has the user explicitly confirmed the tool call with current parameters? Even when all parameters are present, always ask the user for additional confirmation before function_call.",
    )
    reason: str = Field(description="Why this tool is needed")


class SGROrchestration(BaseModel):
    """SGR trace for orchestration and decision-making phase"""

    action: ActionType
    tool_calls: List[ToolCallRequest] = Field(
        description="Tools to execute (can be parallel)",
    )
    reasoning: str = Field(
        description="Step-by-step reasoning for this orchestration decision"
    )


class DecisionResponse(BaseModel):
    """Response from orchestration/decision-making LLM call"""

    sgr: SGROrchestration = Field(description="Orchestration reasoning trace")
    handover_context: str = Field(
        description="results obtained from tool calls, missing parameters or information from internal knowledge to prepare the final answer"
    )
