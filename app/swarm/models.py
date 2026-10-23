"""Internal swarm models: agent profiles, action rules and LLM output schemas.

Wire-level types (SwarmMessage, WS events) live in archie_shared.swarm.models.
"""

from archie_shared.swarm.models import SwarmAgentInfo, SwarmAgentRole, SwarmImportance
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
        description="Persona spec name without the persona_ prefix (e.g. 'bro') for voice and tone",
    )
    groups: list[str] = Field(
        default_factory=list,
        description="Groups; an '@group' mention wakes the members one after another",
    )
    listens_to: list[str] = Field(
        default_factory=list,
        description="Agent IDs whose messages wake this agent without an @mention",
    )
    tool_result_max_chars: int | None = Field(
        default=None,
        description="Tool result size shown to the agent; settings value if empty",
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


class WatchOutcome(BaseModel):
    """Extra fields of a spontaneous check (ARCHIE-238)."""

    importance: SwarmImportance = Field(
        description=(
            "low: only for the chat; normal: show on the LED display; "
            "high: needs attention now, also speak aloud"
        )
    )
    today_note: str = Field(
        description=(
            "Short note of what you saw now (values, IDs, times), for the next "
            "check to compare. Write it also if you report nothing"
        )
    )
    quiet_until: str = Field(
        description=(
            "Local time HH:MM before which nothing new can happen in your area, "
            "e.g. the start of the next match; empty if unknown"
        )
    )
    remember_today: list[str] = Field(
        description=(
            "Facts that you, other agents or the user may need later today, one "
            "short sentence each; all agents see them until midnight. Only facts "
            "that are not yet in today's memory; usually empty"
        )
    )


class SwarmWatchTurn(SwarmTurn, WatchOutcome):
    """First LLM call of a spontaneous check."""


class SwarmWatchReply(SwarmReply, WatchOutcome):
    """Last LLM call of a spontaneous check: the reply is the alert, or empty."""

    reply: str = Field(
        description="Alert for the user, max 200 characters; empty if nothing is new"
    )


class CouncilDelegation(BaseModel):
    """A task the council gives to a home agent."""

    agent_id: str = Field(description="ID of the home agent, without @")
    task: str = Field(description="Short task for the agent, in the user's language")


class CouncilDecision(BaseModel):
    """The single council answer to the user."""

    reply: str = Field(
        description="Answer to the user, max 200 characters, without any @mention"
    )
    delegations: list[CouncilDelegation] = Field(
        description=(
            "Tasks for home agents: only commands the user gave, or data the answer needs. "
            "Empty if you only give advice or offer an action"
        )
    )
    remember: list[str] = Field(
        description=(
            "Lasting facts or preferences to keep in long-term memory, one short "
            "sentence each; only when the user asks to remember or states a "
            "lasting preference. Usually empty"
        )
    )
    remember_today: list[str] = Field(
        description=(
            "Facts from the chat that the swarm may need later today only, one "
            "short sentence each (e.g. a plan of the user for today). Usually empty"
        )
    )


class CouncilAlertDecision(CouncilDecision):
    """Chair's decision on an alert of a spontaneous check (ARCHIE-245)."""

    reasoning: str = Field(
        description="One short sentence: does the user need this now, and can a home agent fix it"
    )


class CouncilTriage(CouncilDecision):
    """Council chair's first call: decide now or convene the council."""

    reasoning: str = Field(description="One short sentence")
    convene: bool = Field(
        description="True if the council must discuss before the decision"
    )
