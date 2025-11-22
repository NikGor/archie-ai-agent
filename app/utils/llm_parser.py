import logging
from typing import Any
from pydantic import BaseModel
from archie_shared.chat.models import InputTokensDetails, LllmTrace, OutputTokensDetails


logger = logging.getLogger(__name__)


class ParsedLLMResponse:
    """Unified parsed response from any LLM provider."""

    def __init__(
        self,
        parsed_content: BaseModel,
        llm_trace: LllmTrace,
        response_id: str | None = None,
        has_function_call: bool = False,
        function_name: str | None = None,
        function_arguments: dict[str, Any] | None = None,
    ):
        self.parsed_content = parsed_content
        self.llm_trace = llm_trace
        self.response_id = response_id
        self.has_function_call = has_function_call
        self.function_name = function_name
        self.function_arguments = function_arguments


def parse_openai_response(
    raw_response: Any,
    expected_type: type[BaseModel],
) -> ParsedLLMResponse:
    """
    Parse OpenAI response into unified structure.

    OpenAI response structure:
    - output[0].type = "message" | "function_call"
    - output[0].content[0].parsed = structured Pydantic object
    - usage.input_tokens / output_tokens / total_tokens
    - usage.input_tokens_details.cached_tokens
    - usage.output_tokens_details.reasoning_tokens

    Args:
        raw_response: Raw response from OpenAI client.responses.parse()
        expected_type: Expected Pydantic model type

    Returns:
        ParsedLLMResponse: Unified parsed response
    """
    logger.info("llm_parser_001: Parsing OpenAI response")

    usage = raw_response.usage
    llm_trace = LllmTrace(
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
        total_cost=0.0,
    )

    output_item = raw_response.output[0]
    has_function_call = output_item.type == "function_call"
    function_name = None
    function_arguments = None
    parsed_content = None

    if has_function_call:
        function_name = output_item.name
        function_arguments = output_item.arguments
        logger.info(
            f"llm_parser_002: Function call detected: \033[36m{function_name}\033[0m"
        )
    else:
        parsed_content = output_item.content[0].parsed
        logger.info(
            f"llm_parser_003: Parsed content type: \033[36m{type(parsed_content).__name__}\033[0m"
        )

    return ParsedLLMResponse(
        parsed_content=parsed_content,
        llm_trace=llm_trace,
        response_id=raw_response.id if hasattr(raw_response, "id") else None,
        has_function_call=has_function_call,
        function_name=function_name,
        function_arguments=function_arguments,
    )


def parse_gemini_response(
    raw_response: Any,
    expected_type: type[BaseModel],
) -> ParsedLLMResponse:
    """
    Parse Gemini response into unified structure.

    Args:
        raw_response: Raw response from Gemini client.models.generate_content()
        expected_type: Expected Pydantic model type

    Returns:
        ParsedLLMResponse: Unified parsed response
    """
    logger.info("llm_parser_004: Parsing Gemini response")

    usage_metadata = raw_response.usage_metadata
    llm_trace = LllmTrace(
        model=(
            raw_response.model_version
            if hasattr(raw_response, "model_version")
            else "gemini"
        ),
        input_tokens=usage_metadata.prompt_token_count,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=usage_metadata.candidates_token_count,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=usage_metadata.total_token_count,
        total_cost=0.0,
    )

    candidate = raw_response.candidates[0]
    part = candidate.content.parts[0]

    has_function_call = (
        hasattr(part, "function_call") and part.function_call is not None
    )
    function_name = None
    function_arguments = None
    parsed_content = None

    if has_function_call:
        function_name = part.function_call.name
        function_arguments = dict(part.function_call.args)
        logger.info(
            f"llm_parser_005: Function call detected: \033[36m{function_name}\033[0m"
        )
    else:
        if hasattr(raw_response, "parsed") and raw_response.parsed:
            raw_parsed = raw_response.parsed
            if isinstance(raw_parsed, dict):
                parsed_content = expected_type.model_validate(raw_parsed)
                logger.info(
                    f"llm_parser_006: Converted dict to Pydantic: \033[36m{type(parsed_content).__name__}\033[0m"
                )
            else:
                parsed_content = raw_parsed
                logger.info(
                    f"llm_parser_006b: Using pre-parsed content: \033[36m{type(parsed_content).__name__}\033[0m"
                )
        elif hasattr(part, "text") and part.text:
            parsed_content = expected_type.model_validate_json(part.text)
            logger.info(
                f"llm_parser_007: Parsed content from text: \033[36m{type(parsed_content).__name__}\033[0m"
            )
        else:
            logger.warning(
                "llm_parser_warning_001: No text or parsed content in Gemini response"
            )

    return ParsedLLMResponse(
        parsed_content=parsed_content,
        llm_trace=llm_trace,
        response_id=(
            raw_response.response_id if hasattr(raw_response, "response_id") else None
        ),
        has_function_call=has_function_call,
        function_name=function_name,
        function_arguments=function_arguments,
    )


def parse_llm_response(
    raw_response: Any,
    provider: str,
    expected_type: type[BaseModel],
) -> ParsedLLMResponse:
    """
    Unified parser for any LLM provider response.

    Args:
        raw_response: Raw response from LLM provider
        provider: Provider name ("openai" or "gemini")
        expected_type: Expected Pydantic model type

    Returns:
        ParsedLLMResponse: Unified parsed response
    """
    if provider == "openai":
        return parse_openai_response(raw_response, expected_type)
    elif provider == "gemini":
        return parse_gemini_response(raw_response, expected_type)
    else:
        logger.error(
            f"llm_parser_error_001: Unknown provider: \033[31m{provider}\033[0m"
        )
        raise ValueError(f"Unsupported provider: {provider}")
