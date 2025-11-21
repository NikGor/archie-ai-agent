"""OpenAI client for API interactions."""

import logging
import os
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI, pydantic_function_tool
from pydantic import BaseModel


logger = logging.getLogger(__name__)
load_dotenv()


class OpenAIClient:
    """Client for OpenAI API interactions."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        logger.info("openai_client_001: Initialized OpenAI client")

    async def create_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_format: type[BaseModel],
        tools: list[type[BaseModel]] | None = None,
        tool_choice: str | None = None,
        previous_response_id: str | None = None,
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
            openai_args = {
                "model": model,
                "input": messages,
                "text_format": response_format,
            }
            if tools:
                openai_args["tools"] = [pydantic_function_tool(t) for t in tools]
            if tool_choice:
                openai_args["tool_choice"] = tool_choice
            if previous_response_id:
                openai_args["previous_response_id"] = previous_response_id
                logger.info(
                    f"openai_client_003: Using previous response ID: \033[36m{previous_response_id}\033[0m"
                )
            response = self.client.responses.parse(**openai_args)
            self._log_usage(response)
            return response
        except Exception as e:
            logger.error(f"openai_client_error_001: \033[31m{e!s}\033[0m")
            raise

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
