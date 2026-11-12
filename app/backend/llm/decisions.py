"""OpenRouter Decisions transport shared by Jev callers."""

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from app.config import settings


DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
_HTTP_ATTEMPTS = 3


class DecisionsError(RuntimeError):
    """A decision request failed or returned an invalid response."""


async def decide(
    model: str, state: dict, questions: dict, timeout: float = 120
) -> dict:
    """One Decisions API call; raises DecisionsError."""
    body = {"model": model, "state": state, "questions": questions}
    headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(_HTTP_ATTEMPTS),
                wait=wait_exponential(multiplier=1, max=10),
                retry=retry_if_exception_type(httpx.TransportError),
                reraise=True,
            ):
                with attempt:
                    response = await client.post(
                        DECISIONS_URL, json=body, headers=headers
                    )
            response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise DecisionsError(f"Jev {model} failed: {exc}") from exc
    if not isinstance(data, dict) or "answers" not in data:
        raise DecisionsError(f"Jev {model} returned no answers: {str(data)[:300]}")
    return data
