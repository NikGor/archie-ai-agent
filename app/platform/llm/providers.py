"""LLM provider adapters (ARCHIE-303).

Everything that differs between providers lives here: the client, how a
response is parsed, how a tool schema and a multimodal user message are built,
and whether the provider keeps the conversation on its side. Callers ask the
adapter for a capability instead of comparing provider names.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
from archie_shared.chat.models import ImageAttachment
from pydantic import BaseModel
from app.utils.llm_parser import (
    ParsedLLMResponse,
    parse_openai_response,
    parse_openrouter_response,
)
from app.utils.tools_utils import openai_parse, oss_parse
from .base import LLMClient


class ProviderAdapter(ABC):
    """One LLM provider; the registry keeps a single instance per provider."""

    name: str
    # The provider keeps the conversation and continues it from a response id,
    # so the chat history must not be inlined again (OpenAI Responses API).
    threads_by_response_id: bool = False

    def __init__(self) -> None:
        self._client: LLMClient | None = None

    def client(self) -> LLMClient:
        """Shared client, built on first use (clients read API keys)."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def accepts_response_id(self, response_id: str) -> bool:
        """True if `response_id` can continue a conversation on this provider."""
        return False

    @abstractmethod
    def _build_client(self) -> LLMClient: ...

    @abstractmethod
    def parse_response(
        self, raw_response: Any, expected_type: type[BaseModel]
    ) -> ParsedLLMResponse: ...

    @abstractmethod
    def tool_schema(self, func: Callable) -> dict[str, Any]: ...

    @abstractmethod
    def image_part(self, image: ImageAttachment) -> dict[str, Any]: ...

    @abstractmethod
    def text_part(self, text: str) -> dict[str, Any]: ...

    def user_content(
        self, text: str, images: list[ImageAttachment] | None
    ) -> str | list[dict[str, Any]]:
        """Plain text, or a content-part list when images are attached."""
        if not images:
            return text
        return [self.text_part(text), *(self.image_part(img) for img in images)]


class OpenAIProvider(ProviderAdapter):
    name = "openai"
    threads_by_response_id = True
    RESPONSE_ID_PREFIX = "resp_"

    def _build_client(self) -> LLMClient:
        from .openai_client import OpenAIClient

        return OpenAIClient()

    def accepts_response_id(self, response_id: str) -> bool:
        return response_id.startswith(self.RESPONSE_ID_PREFIX)

    def parse_response(
        self, raw_response: Any, expected_type: type[BaseModel]
    ) -> ParsedLLMResponse:
        return parse_openai_response(raw_response, expected_type)

    def tool_schema(self, func: Callable) -> dict[str, Any]:
        return openai_parse(func)

    def text_part(self, text: str) -> dict[str, Any]:
        return {"type": "input_text", "text": text}

    def image_part(self, image: ImageAttachment) -> dict[str, Any]:
        return {
            "type": "input_image",
            "image_url": f"data:{image.media_type};base64,{image.data}",
        }


class OpenRouterProvider(ProviderAdapter):
    """Chat-completions API: no server-side threads, `text` / `image_url` parts."""

    name = "openrouter"

    def _build_client(self) -> LLMClient:
        from .openrouter_client import OpenRouterClient

        return OpenRouterClient()

    def parse_response(
        self, raw_response: Any, expected_type: type[BaseModel]
    ) -> ParsedLLMResponse:
        return parse_openrouter_response(raw_response, expected_type)

    def tool_schema(self, func: Callable) -> dict[str, Any]:
        return oss_parse(func)

    def text_part(self, text: str) -> dict[str, Any]:
        return {"type": "text", "text": text}

    def image_part(self, image: ImageAttachment) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{image.media_type};base64,{image.data}"},
        }
