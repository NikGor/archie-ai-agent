"""OpenAI client for API interactions."""

import logging
import os
from collections.abc import AsyncIterator, Callable
from typing import Any

from openai import (
    APIConnectionError,
    AsyncOpenAI,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel

from app.utils.openai_utils import build_openai_args
from app.utils.retry_utils import call_with_retry


logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for OpenAI API interactions."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        logger.info("openai_client_001: Initialized OpenAI client")

    def _log_usage(self, response: Any) -> None:
        """Log token usage from response."""
        try:
            usage = response.usage
            logger.info(
                f"openai_client_004: Usage - Input: \033[33m{usage.input_tokens}\033[0m | "
                f"Output: \033[33m{usage.output_tokens}\033[0m | "
                f"Total: \033[33m{usage.total_tokens}\033[0m | "
                f"Cached: \033[33m{usage.input_tokens_details.cached_tokens}\033[0m"
            )
            logger.info(
                f"openai_client_005: Status: {response.status} | Model: \033[36m{response.model}\033[0m"
            )
        except Exception as e:
            logger.warning(f"openai_client_warning_001: Could not log usage: {e}")

    async def create_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_format: type[BaseModel] | None = None,
        previous_response_id: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
    ) -> Any:
        """Create a completion using OpenAI API with structured outputs."""
        msg_breakdown = {}
        for msg in messages:
            role = msg.get("role", "unknown")
            msg_breakdown[role] = msg_breakdown.get(role, 0) + 1
        logger.info(
            f"openai_client_002: Calling \033[36m{model}\033[0m with \033[33m{len(messages)}\033[0m msgs "
            f"(system: {msg_breakdown.get('system', 0)}, user: {msg_breakdown.get('user', 0)}, assistant: {msg_breakdown.get('assistant', 0)})"
        )
        try:
            openai_args = build_openai_args(
                model=model,
                messages=messages,
                response_format=response_format,
                previous_response_id=previous_response_id,
                tools=tools,
            )

            if previous_response_id:
                logger.info(
                    f"openai_client_003: Using previous response ID: \033[36m{previous_response_id}\033[0m"
                )

            response = await call_with_retry(
                lambda: self.client.responses.parse(**openai_args, timeout=60),
                retryable_exceptions=(
                    RateLimitError,
                    APIConnectionError,
                    InternalServerError,
                ),
                context="openai_client",
            )
            self._log_usage(response)
            return response
        except Exception as e:
            logger.error(f"openai_client_error_001: \033[31m{e!s}\033[0m")
            raise

    async def create_completion_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_format: type[BaseModel] | None = None,
        previous_response_id: str | None = None,
        response_id_out: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens using OpenAI Responses API (responses.stream()).

        Yields raw JSON token strings as they arrive. The caller is responsible
        for reassembling and parsing the full response.

        Note: previous_response_id is passed through for conversation caching.
        If response_id_out is provided (a mutable list), the OpenAI response ID
        (resp_...) will be appended to it from the response.created event.
        """
        args = build_openai_args(
            model=model,
            messages=messages,
            response_format=response_format,
            previous_response_id=previous_response_id,
        )
        logger.info(f"openai_client_006: Starting stream for \033[36m{model}\033[0m")
        try:
            async with self.async_client.responses.stream(**args) as stream:
                async for event in stream:
                    if event.type == "response.created" and response_id_out is not None:
                        resp_id = getattr(event.response, "id", None)
                        if resp_id:
                            response_id_out.append(resp_id)
                    elif event.type == "response.output_text.delta" and getattr(
                        event, "delta", None
                    ):
                        yield event.delta
        except Exception as e:
            logger.error(f"openai_client_error_002: Stream error: \033[31m{e!s}\033[0m")
            raise
