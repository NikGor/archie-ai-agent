"""Ask the runtime for one bounded read-only worker (ARCHIE-300)."""

from app.platform.runtime.spawn import spawn


async def spawn_agent_tool(
    goal: str, context: str = "", demo_mode: bool = False
) -> dict:
    """Diagnose a specific Home Assistant fault without changing anything.

    Args:
        goal: Fault to investigate and report.
        context: Relevant observed facts, entity identifiers and error messages.
    """
    if demo_mode:
        return {
            "status": "demo",
            "summary": "A bounded read-only diagnostic worker would run.",
        }
    result = await spawn(goal, context)
    return result.model_dump(mode="json")
