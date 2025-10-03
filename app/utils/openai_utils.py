"""Utility functions for processing OpenAI responses."""

import logging
from typing import Any

from archie_shared.chat.models import InputTokensDetails, LllmTrace, OutputTokensDetails

from ..models.openai_models import ResponseUsage

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


def get_usage_from_openai_response(raw_response: Any):
    """Extract usage information directly from OpenAI response."""
    return raw_response.usage


def get_agent_response_from_openai(raw_response: Any):
    """Extract AgentResponse directly from OpenAI response."""
    return raw_response.output[0].content[0].parsed


def create_llm_trace_from_response_usage(
    usage: ResponseUsage, model: str, total_cost: float = 0.0
) -> LllmTrace:
    """
    Convert ResponseUsage to LllmTrace for tracking purposes.

    Args:
        usage: ResponseUsage object from OpenAI response
        model: Model name used for generation
        total_cost: Calculated cost of the request (defaults to 0.0)

    Returns:
        LllmTrace: Converted tracking information
    """
    return LllmTrace(
        model=model,
        input_tokens=usage.input_tokens,
        input_tokens_details=InputTokensDetails(
            cached_tokens=usage.input_tokens_details.cached_tokens
        ),
        output_tokens=usage.output_tokens,
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens
        ),
        total_tokens=usage.total_tokens,
        total_cost=total_cost,
    )


def create_llm_trace_from_openai_response(
    raw_response: Any, total_cost: float = 0.0
) -> LllmTrace:
    """
    Create LllmTrace directly from OpenAI response.

    Args:
        raw_response: Raw OpenAI response object
        total_cost: Calculated cost of the request (defaults to 0.0)

    Returns:
        LllmTrace: Tracking information extracted from response
    """
    usage = raw_response.usage
    return LllmTrace(
        model=raw_response.model,
        input_tokens=usage.input_tokens,
        input_tokens_details=InputTokensDetails(
            cached_tokens=usage.input_tokens_details.cached_tokens
        ),
        output_tokens=usage.output_tokens,
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens
        ),
        total_tokens=usage.total_tokens,
        total_cost=total_cost,
    )
