import logging
from collections.abc import Callable
from typing import Any
from pydantic import BaseModel
from app.utils.tools_utils import openai_responses_parse


logger = logging.getLogger(__name__)


def build_openai_args(
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[BaseModel] | None = None,
    previous_response_id: str | None = None,
    tools: list[Callable[..., Any]] | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Build arguments for OpenAI client.responses.parse()."""
    args: dict[str, Any] = {
        "model": model,
        "input": messages,
    }

    if max_output_tokens is not None:
        args["max_output_tokens"] = max_output_tokens

    if response_format:
        args["text_format"] = response_format

    if previous_response_id and previous_response_id.startswith("resp_"):
        args["previous_response_id"] = previous_response_id

    if tools:
        args["tools"] = [openai_responses_parse(func) for func in tools]

    # Add reasoning parameters for thinking models
    if model.startswith(("o1", "o3", "gpt-5")):
        args["reasoning"] = {"effort": "medium", "summary": "auto"}
        logger.info(
            f"openai_utils_001: Added reasoning params for model \033[36m{model}\033[0m"
        )

    return args
