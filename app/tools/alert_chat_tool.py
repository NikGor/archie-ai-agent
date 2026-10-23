"""Alert chat tool: opens a new Archie chat about an event (ARCHIE-247).

The swarm writes a user message about the event. Django creates the
conversation, saves the message and runs the normal chat pipeline, so Archie
answers with a UI answer: cards, forms and buttons for the next steps.
In demo mode the chat still opens; Archie's tools in it run in demo mode.
"""

import logging
import uuid
from typing import Any
import httpx
from app.config import settings
from app.swarm.agents import SwarmRoster


logger = logging.getLogger(__name__)

# The full pipeline (tools + UI answer) can take a minute
_TIMEOUT_SECONDS = 180.0
TITLE_PREFIX = "🔔 "


def chat_path(conversation_id: str) -> str:
    """Path of a chat on the Archie web app; the browser keeps its own host."""
    return f"/c/{conversation_id}/"


async def alert_chat_tool(
    title: str,
    message: str,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Open a new Archie chat about an event. The message goes to Archie as if the
    user wrote it; Archie answers with cards, forms and buttons for next steps.

    Args:
        title: Short chat title, e.g. "Bayern – Dortmund in 5 min"
        message: The user's message to Archie: what happened and a request to
            offer useful next steps

    Returns:
        Dict with the chat path ("/c/<id>/") and the chat title
    """
    conversation_id = str(uuid.uuid4())
    title = TITLE_PREFIX + " ".join(title.split())
    logger.info(f"alert_chat_001: \033[36m{title}\033[0m: {message[:120]}")
    user_name = settings.swarm_user_name or await SwarmRoster().get_owner()
    base = settings.web_django_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{base}/ai-assistant/api/messages/",
                json={
                    "conversation_id": conversation_id,
                    "input": message,
                    "user_name": user_name,
                    "response_format": "ui_answer",
                    # Swarm dry run: the chat opens, its tools run in demo mode
                    "demo_mode": demo_mode,
                },
            )
            response.raise_for_status()
            await client.patch(
                f"{base}/ai-assistant/api/conversations/{conversation_id}/",
                json={"title": title},
            )
    except httpx.HTTPError as error:
        logger.error(f"alert_chat_error_001: \033[31m{error}\033[0m")
        return {"success": False, "error": str(error)}
    return {"chat": chat_path(conversation_id), "title": title}
