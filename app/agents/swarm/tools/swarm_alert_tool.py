"""Swarm alert tool: posts an event to the swarm chat (ARCHIE-248).

Meant for cron jobs: at the planned time the event comes to the swarm as an
alert of @cron. Its importance decides the display and the voice; the council
then decides if the swarm acts or opens an alert chat.
"""

import logging
from typing import Any, Literal
from app.agents.swarm.service import swarm_service


logger = logging.getLogger(__name__)

CRON_AGENT = "cron"


async def swarm_alert_tool(
    text: str,
    importance: Literal["low", "normal", "high"] = "normal",
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Post an event to the swarm chat as an alert. Schedule it with cron_tool to
    remind the user at an exact time; the swarm chooses how to deliver it.

    Args:
        text: The event in the user's language, e.g. "Bayern – Dortmund starts in 5 minutes (18:30)"
        importance: low = swarm chat only, normal = also the LED display,
            high = also the voice (only in the voice hours)

    Returns:
        Dict with the status and the ID of the posted message
    """
    logger.info(f"swarm_alert_001: [{importance}] \033[33m{text}\033[0m")
    if demo_mode:
        return {"status": "demo", "message": "[DEMO] Alert would be posted"}
    if await swarm_service.memory.get_mode() == "stopped":
        return {"status": "skipped", "message": "The swarm is stopped"}
    message = await swarm_service.ticker.post_alert(CRON_AGENT, text, importance)
    return {"status": "success", "message_id": message.message_id}
