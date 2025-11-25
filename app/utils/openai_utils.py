import logging
from typing import Any
from pydantic import BaseModel
from archie_shared.chat.models import InputTokensDetails, LllmTrace, OutputTokensDetails


logger = logging.getLogger(__name__)


def build_openai_args(
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[BaseModel],
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    """Build arguments for OpenAI client.responses.parse()."""
    args = {
        "model": model,
        "input": messages,
        "text_format": response_format,
    }

    if previous_response_id:
        args["previous_response_id"] = previous_response_id

    # Add reasoning parameters for thinking models
    if model.startswith(("o1", "o3", "gpt-5")):
        args["reasoning"] = {"effort": "medium", "summary": "auto"}
        logger.info(
            f"openai_utils_001: Added reasoning params for model \033[36m{model}\033[0m"
        )

    return args


def create_llm_trace_from_openai_response(
    raw_response: Any, total_cost: float = 0.0
) -> LllmTrace:
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
