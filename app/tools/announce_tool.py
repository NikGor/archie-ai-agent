"""Tool that speaks a message aloud on the home speaker via archie-voice.

Publishes to a Redis pub/sub channel that the on-device archie-voice service
subscribes to. Meant to be scheduled by cron_tool for proactive spoken
reminders (e.g. a football match starting), but can also be called directly.
"""

import json
import logging
from typing import Any
from redis.exceptions import RedisError
from app.backend.redis_factory import get_async_redis

logger = logging.getLogger(__name__)

# Must match ANNOUNCE_CHANNEL in the archie-voice service config.
_ANNOUNCE_CHANNEL = "archie:voice:announce"


async def announce_tool(
    prompt: str | None = None,
    text: str | None = None,
    persona: str | None = None,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Speak a message aloud on the user's home speaker.

    Use it for proactive voice reminders, usually scheduled through cron_tool —
    for example, reminding the user about a football match tonight. Pass a
    prompt (a topic) and the message is generated fresh when spoken, so it stays
    current; pass text only when the exact wording must be spoken verbatim.

    Example (scheduled via cron_tool):
        tool_name="announce_tool",
        arguments_json='{"prompt":"remind me the match starts in an hour"}'

    Args:
        prompt: Topic to turn into a fresh spoken line at play time
        text: Exact words to speak verbatim; use instead of prompt
        persona: Optional persona override for the voice; defaults to user's

    Returns:
        dict[str, Any]: Delivery status
    """
    if not prompt and not text:
        return {"status": "error", "message": "Provide either prompt or text"}
    if demo_mode:
        return {"status": "demo", "message": "[DEMO] Announcement would be spoken"}

    payload = {"prompt": prompt, "text": text, "persona": persona}
    try:
        redis = get_async_redis()
        receivers = await redis.publish(_ANNOUNCE_CHANNEL, json.dumps(payload))
    except RedisError as exc:
        logger.error("announce_tool_error_001: publish failed: %s", exc)
        return {"status": "error", "message": str(exc)}

    logger.info("announce_tool_001: published to %d subscriber(s)", receivers)
    return {"status": "success", "receivers": receivers}
