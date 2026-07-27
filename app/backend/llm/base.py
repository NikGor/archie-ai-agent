"""Common interface implemented by every LLM provider client."""

from collections.abc import AsyncIterator
from typing import Any, Protocol
from pydantic import BaseModel


class LLMClient(Protocol):
    """Structural interface shared by OpenAIClient and OpenRouterClient.

    `tools` deliberately stays untyped: OpenAIClient accepts raw callables
    (schema-generated internally) while OpenRouterClient accepts pre-built
    schema dicts — a real difference between the two, not one worth forcing
    into a shared shape.
    """

    async def create_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_format: type[BaseModel] | None = None,
        previous_response_id: str | None = None,
        tools: Any = None,
    ) -> Any: ...

    def create_completion_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_format: type[BaseModel] | None = None,
        previous_response_id: str | None = None,
        response_id_out: list[str] | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...
