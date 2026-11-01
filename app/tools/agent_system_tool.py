"""Scheduled run of a saved agent system (ARCHIE-341). Only the cron scheduler
calls it: the tool group is exclusive, no agent sees it."""

import json
import logging
from typing import Any
from app.platform.runtime.system_runner import run_scheduled_system


logger = logging.getLogger(__name__)


async def agent_system_tool(
    spec_json: str, task: str = "", demo_mode: bool = False
) -> dict[str, Any]:
    """Run an agent system from the agent constructor.

    Args:
        spec_json: AgentSystemSpec as JSON.
        task: Task of the run; empty uses the fixed task of the system.
        demo_mode: Tools return demo data.
    """
    spec = json.loads(spec_json)
    logger.info(f"agent_system_tool_001: scheduled run of {spec.get('name')}")
    return await run_scheduled_system(spec, task, demo_mode)
