"""Prompt-cache breakpoint helpers (ARCHIE-180).

Prompt builders mark the stable prompt prefix with an internal ``cacheable``
flag. Anthropic models on OpenRouter need an explicit ``cache_control`` block to
cache that prefix; every other provider must simply have the marker stripped
before the request is sent (the OpenAI Responses API rejects unknown message
keys). This module is the single place that resolves the marker.
"""

from typing import Any


CACHE_MARKER = "cacheable"


def apply_cache_breakpoints(
    messages: list[dict[str, Any]], model: str
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with the internal cache marker resolved.

    For Anthropic models (``anthropic/*``) the content of marked messages is
    converted to a content-block list carrying ``cache_control: ephemeral``.
    For all providers the internal marker key is removed.
    """
    use_cache_control = model.startswith("anthropic/")
    result: list[dict[str, Any]] = []
    for message in messages:
        clean = {key: value for key, value in message.items() if key != CACHE_MARKER}
        if (
            use_cache_control
            and message.get(CACHE_MARKER)
            and isinstance(clean.get("content"), str)
        ):
            clean["content"] = [
                {
                    "type": "text",
                    "text": clean["content"],
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        result.append(clean)
    return result
