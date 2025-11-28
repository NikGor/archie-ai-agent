"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  DEPRECATED: Use OpenRouterClient instead                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import logging
import os
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types


logger = logging.getLogger(__name__)
load_dotenv()


class GeminiClient:
    """Client for Google Gemini API interactions."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        logger.info("gemini_client_001: Initialized Gemini client")

    def _log_usage(self, response: Any, model: str) -> None:
        """Log token usage from response."""
        try:
            usage = getattr(response, "usage_metadata", None)
            if usage:
                in_tok = getattr(usage, "prompt_token_count", None)
                out_tok = getattr(usage, "candidates_token_count", None)
                total = getattr(usage, "total_token_count", None)
                cached = getattr(usage, "cached_content_token_count", None)
            else:
                in_tok = out_tok = total = cached = None
            logger.info(
                f"gemini_client_004: Usage - Input: \033[33m{in_tok}\033[0m | "
                f"Output: \033[33m{out_tok}\033[0m | "
                f"Total: \033[33m{total}\033[0m | "
                f"Cached: \033[33m{cached}\033[0m"
            )
            logger.info(
                f"gemini_client_005: Status: completed | Model: \033[36m{model}\033[0m"
            )
        except Exception as e:
            logger.warning(f"gemini_client_warning_001: Could not log usage: {e}")

    async def create_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_format: type[BaseModel],
        previous_response_id: str | None = None,
    ) -> Any:
        """Create a completion using Gemini API with structured outputs."""
        msg_breakdown = {}
        for msg in messages:
            role = msg.get("role", "unknown")
            msg_breakdown[role] = msg_breakdown.get(role, 0) + 1
        logger.info(
            f"gemini_client_002: Calling \033[36m{model}\033[0m with \033[33m{len(messages)}\033[0m msgs "
            f"(system: {msg_breakdown.get('system', 0)}, user: {msg_breakdown.get('user', 0)}, assistant: {msg_breakdown.get('assistant', 0)})"
        )
        try:
            # Extract system instruction from first message
            system_instruction = None
            if messages and messages[0]["role"] == "system":
                system_instruction = [types.Part.from_text(text=messages[0]["content"])]
                messages = messages[1:]

            # Convert messages to Gemini format
            contents = []
            for msg in messages:
                if msg["role"] == "user":
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=msg["content"])],
                        )
                    )
                elif msg["role"] == "assistant":
                    contents.append(
                        types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=msg["content"])],
                        )
                    )

            # Get Pydantic schema and remove client-side fields
            schema = response_format.model_json_schema()
            if "properties" in schema:
                schema["properties"].pop("llm_trace", None)
                schema["properties"].pop("response_id", None)
                if "required" in schema:
                    schema["required"] = [
                        field
                        for field in schema["required"]
                        if field not in ["llm_trace", "response_id"]
                    ]

            # Configure generation with direct JSON schema
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=schema,
                system_instruction=system_instruction,
            )

            # Generate response
            logger.info("gemini_client_003: Calling generate_content")
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )

            self._log_usage(response, model)
            return response

        except Exception as e:
            logger.error(f"gemini_client_error_001: \033[31m{e!s}\033[0m")
            raise
