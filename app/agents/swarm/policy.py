"""Action policy: what a swarm agent may execute on its own."""

import logging
from typing import Literal
import app.agents.swarm.tools  # noqa: F401  registers the swarm tool group
from app.agents.swarm.models import SwarmAgentProfile, SwarmToolCall
from app.tools.catalog import tool_registry


logger = logging.getLogger(__name__)

PolicyDecision = Literal["allow", "block"]


def is_read_only(tool_call: SwarmToolCall) -> bool:
    """Read-only by the tool capabilities of the registry (ARCHIE-320)."""
    return tool_registry.is_read_only(
        tool_call.tool_name, {p.name: p.value for p in tool_call.arguments}
    )


def check_tool_call(
    profile: SwarmAgentProfile, tool_call: SwarmToolCall
) -> PolicyDecision:
    """Allow a call only if the tool is in the profile and, for side effects,
    matches an auto_actions rule. State agents never get side effects."""
    reason = block_reason(profile, tool_call)
    if reason is None:
        return "allow"
    logger.warning(
        f"swarm_policy_warning_001: Blocked \033[33m{tool_call.tool_name}\033[0m "
        f"for \033[36m{profile.agent_id}\033[0m: {reason}"
    )
    return "block"


def block_reason(profile: SwarmAgentProfile, tool_call: SwarmToolCall) -> str | None:
    """Why the call is blocked, in words the agent can act on; None if allowed."""
    if tool_call.tool_name not in profile.tools:
        return "this tool is not yours"
    if is_read_only(tool_call):
        return None
    if profile.role != "action":
        return "you only read data, you cannot change anything"
    arguments = {param.name: param.value for param in tool_call.arguments}
    rules = [r for r in profile.auto_actions if r.tool_name == tool_call.tool_name]
    for rule in rules:
        if all(
            arguments.get(name) in values
            for name, values in rule.allowed_values.items()
        ):
            return None
    allowed = "; ".join(
        f"{name} must be one of: {', '.join(values)}"
        for rule in rules
        for name, values in rule.allowed_values.items()
    )
    return f"wrong arguments ({allowed})" if allowed else "you may not do this"
