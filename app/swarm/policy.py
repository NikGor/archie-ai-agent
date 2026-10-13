"""Action policy: what a swarm agent may execute on its own."""

import logging
from typing import Literal
from .models import SwarmAgentProfile, SwarmToolCall


logger = logging.getLogger(__name__)

# Tools that only read data. Every other tool has side effects.
READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "smarthome_status_tool",
        "google_search_tool",
        "google_places_search_tool",
        "document_search_tool",
        "skill_loader_tool",
    }
)

PolicyDecision = Literal["allow", "block"]


def check_tool_call(
    profile: SwarmAgentProfile, tool_call: SwarmToolCall
) -> PolicyDecision:
    """Allow a call only if the tool is in the profile and, for side effects,
    matches an auto_actions rule. State agents never get side effects."""
    if tool_call.tool_name not in profile.tools:
        return _block(profile, tool_call, "tool not in profile")
    if tool_call.tool_name in READ_ONLY_TOOLS:
        return "allow"
    if profile.role != "action":
        return _block(profile, tool_call, "state agent cannot act")
    arguments = {param.name: param.value for param in tool_call.arguments}
    for rule in profile.auto_actions:
        if rule.tool_name != tool_call.tool_name:
            continue
        if all(
            arguments.get(name) in values
            for name, values in rule.allowed_values.items()
        ):
            return "allow"
    return _block(profile, tool_call, "arguments outside auto_actions")


def _block(
    profile: SwarmAgentProfile, tool_call: SwarmToolCall, reason: str
) -> PolicyDecision:
    logger.warning(
        f"swarm_policy_warning_001: Blocked \033[33m{tool_call.tool_name}\033[0m "
        f"for \033[36m{profile.agent_id}\033[0m: {reason}"
    )
    return "block"
