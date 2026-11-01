"""Hook from the scheduled tool to the agent systems of this process
(ARCHIE-341): the cron scheduler runs `agent_system_tool`, the tool reaches
the runner that the composition root registered. Tools never import agents."""

from collections.abc import Awaitable, Callable
from pydantic import JsonValue


SystemRunner = Callable[[dict, str, bool], Awaitable[dict[str, JsonValue]]]
_runner: SystemRunner | None = None


def register_system_runner(runner: SystemRunner) -> None:
    global _runner
    _runner = runner


async def run_scheduled_system(
    spec: dict, task: str, demo_mode: bool
) -> dict[str, JsonValue]:
    if _runner is None:
        raise RuntimeError("Agent systems are not configured in this process")
    return await _runner(spec, task, demo_mode)
