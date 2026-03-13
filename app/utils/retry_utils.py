"""Retry utility with exponential backoff for LLM API calls."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any


logger = logging.getLogger(__name__)

RETRY_BACKOFF = (1.0, 2.0, 4.0)


async def call_with_retry(
    func: Callable[[], Any],
    retryable_exceptions: tuple[type[Exception], ...],
    context: str = "",
    max_attempts: int = 3,
    backoff: tuple[float, ...] = RETRY_BACKOFF,
) -> Any:
    """
    Call a synchronous function with exponential backoff retry.

    Retries on specified exception types. Uses asyncio.sleep for backoff
    so it doesn't block the event loop.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except retryable_exceptions as e:
            if attempt == max_attempts:
                logger.error(
                    f"retry_utils_003: {context} failed after {max_attempts} attempts: {e}"
                )
                raise
            wait = backoff[attempt - 1] if attempt - 1 < len(backoff) else backoff[-1]
            logger.warning(
                f"retry_utils_001: {context} attempt {attempt}/{max_attempts} failed "
                f"({type(e).__name__}: {e}), retrying in {wait}s"
            )
            await asyncio.sleep(wait)
    raise RuntimeError(f"retry_utils: unreachable, context={context}")  # pragma: no cover
