"""Shared tail of the Stage 3 execution paths: coerce -> build Content ->
validate URLs -> assemble AgentResponse. All four create_output paths
(plain/level2/ui stream, blocking) converge here after parsing.
"""

import logging
from typing import Any
from archie_shared.chat.models import LllmTrace
from pydantic import ValidationError
from ...models.output_models import AgentResponse, UIResponse
from ...utils.llm_parser import build_content_from_parsed
from ...utils.url_validator import validate_and_fix_urls
from .ui_repair import clear_card_image_prompts, sanitize_chart_items


logger = logging.getLogger(__name__)


async def build_agent_response(
    parsed_content: Any,
    response_format: str,
    llm_trace: LllmTrace | None,
    response_id: str | None = None,
    ttft_ms: int | None = None,
    no_image: bool = False,
) -> AgentResponse:
    """Turn a parsed LLM response into the final AgentResponse.

    For ui_answer, `parsed_content` may be an instance of a dynamically
    filtered response model (see schema_filter.build_filtered_ui_response);
    it is coerced back to the full UIResponse before the UI band-aids run.
    """
    if response_format == "ui_answer":
        try:
            parsed_content = UIResponse.model_validate(parsed_content.model_dump())
        except ValidationError as e:
            logger.error(f"create_output_007: UIResponse coercion failed: {e}")
            raise
        sanitize_chart_items(parsed_content)
        if no_image:
            clear_card_image_prompts(parsed_content)

    content = build_content_from_parsed(
        parsed_content=parsed_content,
        response_format=response_format,
    )
    content = await validate_and_fix_urls(content)  # ARCHIE-123

    result = AgentResponse(
        content=content,
        sgr=parsed_content.sgr,
        llm_trace=llm_trace,
        response_id=response_id,
        ttft_ms=ttft_ms,
    )
    content_text = str(result.content) if result.content else ""
    logger.info(
        f"create_output_005: Created response with length: \033[33m{len(content_text)}\033[0m"
    )
    logger.info(
        f"create_output_006: UI reasoning: \033[35m{result.sgr.ui_reasoning}\033[0m"
    )
    return result
