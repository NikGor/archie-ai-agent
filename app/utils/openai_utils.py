"""Utility functions for processing OpenAI responses."""

import logging
from typing import Any

from ..models.openai_models import OpenAIFullResponse, ParsedResponse

logger = logging.getLogger(__name__)


def extract_response_metrics(raw_response: Any) -> dict[str, Any]:
    """
    Extract useful metrics from OpenAI response for monitoring and analytics.

    Args:
        raw_response: The raw response object from OpenAI API

    Returns:
        dict: Dictionary containing response metrics
    """
    try:
        usage = raw_response.usage

        metrics = {
            "response_id": raw_response.id,
            "model": raw_response.model,
            "created_at": raw_response.created_at,
            "status": raw_response.status,
            "temperature": raw_response.temperature,
            "top_p": raw_response.top_p,
            "parallel_tool_calls": raw_response.parallel_tool_calls,
            "tools_available": len(raw_response.tools),
            "service_tier": raw_response.service_tier,
            "token_usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "cached_tokens": usage.input_tokens_details.cached_tokens,
                "reasoning_tokens": usage.output_tokens_details.reasoning_tokens,
                "cache_hit_rate": (
                    usage.input_tokens_details.cached_tokens / usage.input_tokens * 100
                    if usage.input_tokens > 0
                    else 0
                ),
            },
            "billing": raw_response.billing,
        }

        return metrics

    except Exception as e:
        logger.error(f"openai_utils_error_001: \033[31m{e}\033[0m")
        return {"extraction_error": str(e)}


def validate_response_structure(raw_response: Any) -> bool:
    """Validate the structure of OpenAI response."""
    try:
        _ = raw_response.output[0].content[0].parsed
        return True
    except (AttributeError, IndexError, TypeError):
        return False


def create_openai_full_response_model(raw_response: Any) -> OpenAIFullResponse:
    """Create an OpenAIFullResponse model from raw response."""
    if not validate_response_structure(raw_response):
        raise ValueError("Invalid OpenAI response structure")

    parsed_response = ParsedResponse(**raw_response.__dict__)
    return OpenAIFullResponse(response=parsed_response)
