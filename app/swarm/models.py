"""Internal swarm models: agent profiles, action rules and LLM output schemas.

Wire-level types (SwarmMessage, WS events) live in archie_shared.swarm.models.
"""

from archie_shared.swarm.models import SwarmAgentInfo, SwarmAgentRole
from pydantic import BaseModel, Field
from ..models.orchestration_sgr import Parameter, ToolName


class ActionRule(BaseModel):
    """Lets an action agent call one tool without approval, within argument limits."""

    tool_name: str = Field(description="Tool the agent may call on its own")
    allowed_values: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Argument name -> allowed values; arguments not listed are free",
    )


class SwarmAgentProfile(BaseModel):
    """Declarative description of one swarm agent."""

    agent_id: str = Field(description="ID used in @mentions, lowercase")
    display_name: str = Field(description="Human-readable name")
    role: SwarmAgentRole = Field(
        description="state agents only observe; action agents may act"
    )
    description: str = Field(description="One line: what the agent is responsible for")
    tools: list[str] = Field(
        description="Tools shown to the agent (TOOLS_CONFIG names)"
    )
    auto_actions: list[ActionRule] = Field(
        default_factory=list,
        description="Tools with side effects the agent may call without approval",
    )
    skill: str | None = Field(
        default=None, description="Runtime skill preloaded into the prompt"
    )
    references: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Skill name -> reference files preloaded into the prompt",
    )
    model: str | None = Field(
        default=None, description="LLM override; settings.swarm_model if empty"
    )
    persona: str | None = Field(
        default=None,
        description="Persona prompt from persona_prompts/ (e.g. 'bro') for voice and tone",
    )
    groups: list[str] = Field(
        default_factory=list,
        description="Groups; an '@group' mention wakes the members one after another",
    )
    listens_to: list[str] = Field(
        default_factory=list,
        description="Agent IDs whose messages wake this agent without an @mention",
    )
    icon: str = Field(description="Lucide icon name")
    color: str = Field(description="Accent color as CSS hex")

    def to_info(self, enabled: bool) -> SwarmAgentInfo:
        return SwarmAgentInfo(
            agent_id=self.agent_id,
            display_name=self.display_name,
            role=self.role,
            description=self.description,
            icon=self.icon,
            color=self.color,
            enabled=enabled,
            groups=self.groups,
        )


class SwarmToolCall(BaseModel):
    """A tool call requested by a swarm agent (compact ToolCallRequest)."""

    tool_name: ToolName = Field(description="Name of the tool to call")
    arguments: list[Parameter] = Field(description="Tool arguments")


class SwarmTurn(BaseModel):
    """First LLM call of an agent turn: decide tools and/or reply."""

    reasoning: str = Field(description="One short sentence: why you act this way")
    tool_calls: list[SwarmToolCall] = Field(
        description="Tools to call now; empty if no tool is needed"
    )
    reply: str = Field(
        description=(
            "Chat message, max 280 characters, may @mention other agents. "
            "Empty if you call tools or have nothing useful to say"
        )
    )


class SwarmReply(BaseModel):
    """Second LLM call of an agent turn: reply after tool results."""

    reply: str = Field(
        description="Chat message, max 280 characters, may @mention other agents"
    )


class CouncilTriage(BaseModel):
    """Council chair's first call: answer now or convene the council."""

    reasoning: str = Field(description="One short sentence")
    convene: bool = Field(
        description="True if the council must discuss before the decision"
    )
    reply: str = Field(
        description=(
            "If convene is false: the council decision for the user, max 280 characters, "
            "with an @mention and a short task for a home agent if work is needed. "
            "Empty if convene is true"
        )
    )
