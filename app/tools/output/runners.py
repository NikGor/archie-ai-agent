"""Streaming token collection for Stage 3 LLM calls."""

import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any
from ...backend.llm.base import LLMClient
from ...utils.stream_utils import JsonPathExtractor


async def stream_and_collect(
    client: LLMClient,
    messages: list[dict],
    model: str,
    response_model: type,
    previous_response_id: str | None,
    extractor: JsonPathExtractor,
    on_chunk: Callable[[str], Awaitable[None]] | None = None,
    response_id_out: list[str] | None = None,
    max_output_tokens: int | None = None,
    extra_extractors: (
        Sequence[tuple[Any, Callable[[str], Awaitable[None]]]] | None
    ) = None,
) -> tuple[str, int | None]:
    """Stream LLM response, collect JSON, return (full_json, ttft_ms)."""
    json_parts: list[str] = []
    stream_start = time.monotonic()
    ttft_ms: int | None = None
    kwargs: dict[str, Any] = {}
    if response_id_out is not None:
        kwargs["response_id_out"] = response_id_out
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    async for token in client.create_completion_stream(
        messages=messages,
        model=model,
        response_format=response_model,
        previous_response_id=previous_response_id,
        **kwargs,
    ):
        if ttft_ms is None:
            ttft_ms = int((time.monotonic() - stream_start) * 1000)
        json_parts.append(token)
        chunk = extractor.feed(token)
        if chunk and on_chunk:
            await on_chunk(chunk)
        if extra_extractors:
            for ext, cb in extra_extractors:
                extra_chunk = ext.feed(token)
                if extra_chunk:
                    await cb(extra_chunk)
    return "".join(json_parts), ttft_ms
