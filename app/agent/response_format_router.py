"""Select Auto's concrete output format from the completed command context."""

import asyncio
import logging
from typing import Literal
from pydantic import BaseModel, Field
from app.backend.llm.decisions import DecisionsError, decide
from app.config import settings
from app.models.tool_models import ToolResult
from .prompt_compiler import PromptCompiler


logger = logging.getLogger(__name__)
ResponseFormat = Literal["plain", "level2_answer", "ui_answer"]


class FormatAnswer(BaseModel):
    choice: ResponseFormat = Field(
        description="Concrete response format selected by Jev"
    )
    probabilities: dict[ResponseFormat, float] = Field(
        description="Probability per format"
    )


async def select_response_format(
    user_input: str,
    command_summary: str,
    tool_results: list[ToolResult],
    intents: list[str],
    chat_history: str | None = None,
) -> ResponseFormat:
    compiler = PromptCompiler()
    questions = {
        "response_format": {
            "type": "choice",
            "instructions": compiler.compile("response_format_router"),
            "criteria": {
                name: compiler.compile(f"response_format_{name}")
                for name in ("plain", "level2_answer", "ui_answer")
            },
        }
    }
    state = {
        "user_request": user_input,
        "chat_history": chat_history or "",
        "command_summary": command_summary,
        "tool_results": [result.model_dump(mode="json") for result in tool_results],
        "ui_intents": intents,
    }
    try:
        async with asyncio.timeout(settings.auto_response_timeout):
            data = await decide(
                settings.auto_response_model,
                state,
                questions,
                timeout=settings.auto_response_timeout,
            )
        answer = FormatAnswer.model_validate(data["answers"]["response_format"])
    except (DecisionsError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        logger.warning(
            f"response_format_router_001: Jev failed; using level2_answer: {exc}"
        )
        return "level2_answer"
    logger.info(
        f"response_format_router_002: Selected {answer.choice}, probabilities={answer.probabilities}"
    )
    return answer.choice
