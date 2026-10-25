"""Tool that runs the main Archie agent on a prompt, for scheduled tasks.

cron_tool calls tools without an LLM. This tool is the bridge for jobs that
need the agent itself: search, read the calendar, create events, schedule more
jobs. The run is a normal chat turn (traced in Langfuse) with more Stage 1
rounds than an interactive turn.
"""

import logging
import uuid
from contextvars import ContextVar
from typing import Any
from archie_shared.chat.models import ChatRequest
from app.api_controller import handle_chat
from app.config import settings


logger = logging.getLogger(__name__)

# A task that calls this tool again would start an endless chain of agent runs.
_inside_task: ContextVar[bool] = ContextVar("agent_task_active", default=False)
_REPLY_LIMIT = 500


async def agent_task_tool(
    prompt: str, user_name: str | None = None, demo_mode: bool = False
) -> dict[str, Any]:
    """
    Run the Archie agent on a prompt later, without the user in the chat.

    Only for jobs created with cron_tool that need real agent work at the time
    they run: for example every morning plan today's football matches, or
    check something and report it. Write the prompt as the user would write it
    to Archie, with all details. Never call it to answer in a chat now.

    Example (scheduled via cron_tool):
        tool_name="agent_task_tool",
        arguments_json='{"prompt":"Plan today football matches","user_name":"Niko"}'

    Args:
        prompt: Full request for the agent, as the user would write it
        user_name: User whose profile and settings the agent uses

    Returns:
        dict[str, Any]: Run status with the agent's short reply
    """
    if not prompt.strip():
        return {"status": "error", "message": "prompt is required"}
    if _inside_task.get():
        logger.warning("agent_task_tool_warning_001: Nested agent task refused")
        return {"status": "error", "message": "agent_task_tool cannot call itself"}
    request = ChatRequest(
        input=prompt,
        user_name=user_name,
        response_format="plain",
        conversation_id=f"scheduled-{uuid.uuid4().hex[:12]}",
        demo_mode=demo_mode,
        no_image=True,
    )
    token = _inside_task.set(True)
    try:
        message = await handle_chat(
            request, max_command_iterations=settings.agent_task_max_iterations
        )
    except Exception as exc:
        logger.exception("agent_task_tool_error_001: Scheduled agent run failed")
        return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
    finally:
        _inside_task.reset(token)
    reply = (message.content.text or "")[:_REPLY_LIMIT]
    logger.info(
        f"agent_task_tool_001: Run finished for \033[35m{user_name}\033[0m, reply length {len(reply)}"
    )
    return {"status": "success", "reply": reply}
