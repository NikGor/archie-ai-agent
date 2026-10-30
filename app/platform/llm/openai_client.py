"""OpenAI client for API interactions."""

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any
from archie_shared.chat.models import ReasoningEffort
from langfuse.openai import AsyncOpenAI, OpenAI
from openai import APIConnectionError, InternalServerError, RateLimitError
from openai.lib._parsing._responses import type_to_text_format_param
from pydantic import BaseModel
from app.config import settings
from app.utils.openai_utils import build_openai_args
from app.utils.retry_utils import call_with_retry


logger = logging.getLogger(__name__)


class TruncatedOutputError(RuntimeError):
    """The model stopped at the output token limit before a valid answer."""


def _openai_usage_to_dict(usage: Any) -> dict[str, int]:
    """Normalize an OpenAI Responses usage object into the stream usage shape."""
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        "cached_tokens": (
            getattr(input_details, "cached_tokens", 0) or 0 if input_details else 0
        ),
        "reasoning_tokens": (
            getattr(output_details, "reasoning_tokens", 0) or 0 if output_details else 0
        ),
    }


class OpenAIClient:
    """Client for OpenAI API interactions."""

    def __init__(self):
        self.api_key = settings.openai_api_key
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
        observation_name: str = "llm-call",
        reasoning_effort: ReasoningEffort | None = None,
    ) -> Any:
        """Create a completion using OpenAI API with structured outputs.

        `observation_name` names the Langfuse generation (ARCHIE-211).
        """
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
                reasoning_effort=reasoning_effort,
            )

            if previous_response_id:
                logger.info(
                    f"openai_client_003: Using previous response ID: \033[36m{previous_response_id}\033[0m"
                )

            response = await call_with_retry(
                lambda: self.client.responses.parse(
                    **openai_args, name=observation_name, timeout=60
                ),
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
        max_output_tokens: int | None = None,
        usage_out: dict[str, int] | None = None,
        observation_name: str = "llm-call",
        reasoning_effort: ReasoningEffort | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens using OpenAI Responses API (responses.create(stream=True)).

        Yields raw JSON token strings as they arrive. The caller is responsible
        for reassembling and parsing the full response.

        Note: previous_response_id is passed through for conversation caching.
        If response_id_out is provided (a mutable list), the OpenAI response ID
        (resp_...) will be appended to it from the response.created event. If
        usage_out is provided, token usage is written into it from the
        response.completed event (ARCHIE-180).

        ARCHIE-211: create(stream=True) instead of responses.stream(), because only
        create() lets langfuse.openai name the generation (`observation_name`).
        Truncation is detected from the response.incomplete event (ARCHIE-151).
        """
        args = build_openai_args(
            model=model,
            messages=messages,
            response_format=response_format,
            previous_response_id=previous_response_id,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        text_format = args.pop("text_format", None)
        if text_format:
            args["text"] = {"format": type_to_text_format_param(text_format)}
        logger.info(f"openai_client_006: Starting stream for \033[36m{model}\033[0m")
        try:
            stream = await self.async_client.responses.create(
                **args, stream=True, name=observation_name
            )
            async with stream as events:
                async for event in events:
                    if event.type == "response.created" and response_id_out is not None:
                        response_id_out.append(event.response.id)
                    elif event.type == "response.completed" and usage_out is not None:
                        usage = event.response.usage
                        if usage:
                            usage_out.update(_openai_usage_to_dict(usage))
                    elif event.type == "response.incomplete":
                        details = event.response.incomplete_details
                        reason = details.reason if details else "unknown"
                        # ARCHIE-321: a cut-off call is still paid; budgets must see it
                        if usage_out is not None and event.response.usage:
                            usage_out.update(
                                _openai_usage_to_dict(event.response.usage)
                            )
                        logger.error(
                            f"openai_client_error_003: Stream truncated before valid JSON "
                            f"(reason: {reason}, max_output_tokens={max_output_tokens})"
                        )
                        raise TruncatedOutputError(
                            "LLM response was cut off before completing a valid answer "
                            "(output too long for the configured token limit)"
                        )
                    elif event.type == "response.output_text.delta" and event.delta:
                        yield event.delta
        except Exception as e:
            logger.error(f"openai_client_error_002: Stream error: \033[31m{e!s}\033[0m")
            raise
