"""Retry utility with exponential backoff for LLM API calls."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
)


logger = logging.getLogger(__name__)

RETRY_BACKOFF = (1.0, 2.0, 4.0)


def _wait_from_backoff(backoff: tuple[float, ...]) -> Callable[[RetryCallState], float]:
    def _wait(retry_state: RetryCallState) -> float:
        idx = retry_state.attempt_number - 1
        return backoff[idx] if idx < len(backoff) else backoff[-1]

    return _wait


def _before_sleep(context: str, max_attempts: int) -> Callable[[RetryCallState], None]:
    def _log(retry_state: RetryCallState) -> None:
        e = retry_state.outcome.exception() if retry_state.outcome else None
        wait = retry_state.next_action.sleep if retry_state.next_action else 0
        logger.warning(
            f"retry_utils_001: {context} attempt {retry_state.attempt_number}/{max_attempts} failed "
            f"({type(e).__name__}: {e}), retrying in {wait}s"
        )

    return _log


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
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=_wait_from_backoff(backoff),
            retry=retry_if_exception_type(retryable_exceptions),
            before_sleep=_before_sleep(context, max_attempts),
            sleep=asyncio.sleep,
            reraise=True,
        ):
            with attempt:
                return func()
    except retryable_exceptions as e:
        logger.error(
            f"retry_utils_003: {context} failed after {max_attempts} attempts: {e}"
        )
        raise
