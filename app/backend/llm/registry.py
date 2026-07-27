"""Single lazy registry of LLM provider client singletons.

Replaces the two independent client dicts that used to be built separately
in AgentFactory.__init__ and at create_output_tool module scope.
"""

from .base import LLMClient


_clients: dict[str, LLMClient] = {}


def get_client(provider: str) -> LLMClient:
    """Return the shared client singleton for `provider`, constructing it on first use."""
    if provider not in _clients:
        _clients[provider] = _build_client(provider)
    return _clients[provider]


def _build_client(provider: str) -> LLMClient:
    if provider == "openai":
        from ..openai_client import OpenAIClient

        return OpenAIClient()
    if provider == "openrouter":
        from ..openrouter_client import OpenRouterClient

        return OpenRouterClient()
    raise ValueError(f"Unsupported provider: {provider}")
