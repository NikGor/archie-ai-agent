"""Utility for resolving LLM provider from model name."""

from ..config import MODEL_PROVIDERS, VISION_MODELS


def get_provider_for_model(model: str) -> str:
    """Return provider name ('openai', 'openrouter', 'gemini') for a given model.

    Falls back to 'openai' for unknown model names.
    """
    for provider, models in MODEL_PROVIDERS.items():
        if model in models:
            return provider
    return "openai"


def supports_vision(model: str) -> bool:
    """Return True if the model accepts image input."""
    return model in VISION_MODELS
