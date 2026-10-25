"""Registry of LLM provider adapters (ARCHIE-303).

A new provider is one ProviderAdapter subclass added to PROVIDERS; no other
module compares provider names.
"""

import logging
from typing import Any
from pydantic import BaseModel
from app.utils.llm_parser import ParsedLLMResponse
from .base import LLMClient
from .providers import OpenAIProvider, OpenRouterProvider, ProviderAdapter


logger = logging.getLogger(__name__)

PROVIDERS: dict[str, ProviderAdapter] = {
    adapter.name: adapter for adapter in (OpenAIProvider(), OpenRouterProvider())
}


def get_provider(name: str) -> ProviderAdapter:
    """Adapter of provider `name`; raises ValueError for an unknown provider."""
    try:
        return PROVIDERS[name]
    except KeyError:
        logger.error(f"llm_registry_error_001: Unknown provider: \033[31m{name}\033[0m")
        raise ValueError(f"Unsupported provider: {name}") from None


def get_client(provider: str) -> LLMClient:
    """Shared client singleton of `provider`."""
    return get_provider(provider).client()


def parse_llm_response(
    raw_response: Any, provider: str, expected_type: type[BaseModel]
) -> ParsedLLMResponse:
    """Parse a raw response of `provider` into the unified ParsedLLMResponse."""
    return get_provider(provider).parse_response(raw_response, expected_type)
