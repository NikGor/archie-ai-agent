"""Hand a multi-step task to a temporary agent configured for it (ARCHIE-324).

The platform configures the agent (AgentConfigurator), checks the spec against
the chat's own tools and runs it read-only. Changes come back as proposed
actions for the user to confirm.
"""

import logging
from typing import Any
from app.platform.runtime.delegation import delegate


logger = logging.getLogger(__name__)


async def delegate_agent_tool(task: str, context: str = "") -> dict[str, Any]:
    """
    Give a multi-step task to a temporary specialist agent and get its result.

    Use it only when the task needs several tool calls, several sources or a
    comparison of evidence, for example: compare three restaurants by reviews
    and prices, or find out why a device stopped working. For a task that one
    tool call can finish, call that tool directly. The agent only reads data;
    changes come back as proposed_actions that the user must confirm.

    Args:
        task: The full task with every detail the agent needs
        context: Facts from the chat the task needs; no personal data it does not need

    Returns:
        dict[str, Any]: state, summary, findings or answer, proposed_actions, run_id
    """
    if not task.strip():
        return {"status": "error", "message": "task is required"}
    try:
        return await delegate(task, context)
    except (PermissionError, RuntimeError) as exc:
        logger.warning(f"delegate_agent_tool_warning_001: {exc}")
        return {"status": "error", "message": str(exc)}
