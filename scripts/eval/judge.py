"""LLM-as-judge for qualitative evaluation of ui_answer responses."""

import json
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .eval_types import EvalCase


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI()


class _JudgeOutput(BaseModel):
    ui_optimality: int = Field(
        ge=0,
        le=10,
        description=(
            "Score 0-10: Were the right UI component types chosen for the request? "
            "10 = perfect specialized types (weather_card, location_card, event_form, etc.), "
            "5 = generic types used where specialized exist, "
            "0 = completely wrong or no structured components."
        ),
    )
    content_informativeness: int = Field(
        ge=0,
        le=10,
        description=(
            "Score 0-10: How useful and complete is the response content? "
            "10 = comprehensive, actionable, all relevant info present, "
            "5 = partial or surface-level, "
            "0 = empty, irrelevant, or unhelpful."
        ),
    )
    prompt_following: int = Field(
        ge=0,
        le=10,
        description=(
            "Score 0-10: Did the response follow ui_answer design guidelines? "
            "10 = quick_actions present and relevant, forms rendered for creation intents, "
            "interactive components used, no walls of text, "
            "5 = partial compliance, "
            "0 = ignores guidelines (static text dump, missing forms, no interactivity)."
        ),
    )
    reasoning: str = Field(
        description="1-3 sentence explanation justifying the scores."
    )


@dataclass
class JudgeResult:
    ui_optimality: int
    content_informativeness: int
    prompt_following: int
    reasoning: str

    @property
    def mean_score(self) -> float:
        return (
            self.ui_optimality + self.content_informativeness + self.prompt_following
        ) / 3


def _serialize_ui_answer(content: Any) -> str:
    """Build compact JSON-serializable summary of UIAnswer for the judge."""
    ui_answer = getattr(content, "ui_answer", None)
    if ui_answer is None:
        return json.dumps({"error": "no ui_answer found in response"})

    items_summary = []
    for item in getattr(ui_answer, "items", []):
        entry: dict[str, Any] = {"type": item.type, "layout": item.layout_hint}
        c = item.content
        # Grab a few key fields per type to give the judge context
        if hasattr(c, "title"):
            entry["title"] = getattr(c, "title", None)
        if hasattr(c, "text"):
            t = getattr(c, "text", None)
            entry["text_snippet"] = (t[:120] + "…") if t and len(t) > 120 else t
        if hasattr(c, "cards"):
            cards = getattr(c, "cards", []) or []
            entry["card_count"] = len(cards)
            entry["card_types"] = list({type(card).__name__ for card in cards})
        items_summary.append(entry)

    qab = getattr(ui_answer, "quick_action_buttons", None)
    buttons_info = None
    if qab:
        buttons = getattr(qab, "buttons", []) or []
        buttons_info = [getattr(b, "label", None) for b in buttons]

    intro = getattr(ui_answer, "intro_text", None)
    intro_text = None
    if intro:
        t = getattr(intro, "text", None)
        intro_text = (t[:200] + "…") if t and len(t) > 200 else t

    return json.dumps(
        {
            "intro_text": intro_text,
            "items": items_summary,
            "quick_action_buttons": buttons_info,
        },
        ensure_ascii=False,
        indent=2,
    )


_SYSTEM_PROMPT = """\
You are a strict quality evaluator for an AI assistant that generates UI-rich responses.
The assistant uses structured UI components instead of plain text wherever possible.

Component types and their ideal use cases:
- weather_card: weather forecasts and conditions
- location_card: places, addresses, restaurants, venues
- card_grid: multiple results (use specialized card types inside, not generic)
- event_form: calendar event creation
- note_form: note/reminder creation
- email_form: email composition
- table: structured comparisons, data with rows/columns
- chart: statistical or trend data
- text_answer: supplementary explanation only (NOT as primary content)
- image: diagrams, visual explanations, illustrations

Design guidelines the assistant MUST follow:
- ALWAYS render a form (event_form / note_form) when user intent is creation — do NOT describe in text
- ALWAYS include quick_action_buttons (2-3 logical follow-up actions)
- NEVER use text_answer as the only component for queries that suit structured data
- Use specialized card types, not generic cards, when available
- Responses should be interactive and scannable, not walls of text

Score strictly. Do not give 10 unless the response is genuinely excellent.\
"""


async def evaluate_response(
    case: EvalCase,
    response: Any,
    judge_model: str = "gpt-4.1-mini",
) -> JudgeResult | None:
    """Call LLM judge to score ui_optimality, content_informativeness, prompt_following."""
    try:
        response_summary = _serialize_ui_answer(response.content)
        user_message = (
            f"User request: {case.input}\n\n" f"Response structure:\n{response_summary}"
        )
        result = await _get_client().beta.chat.completions.parse(
            model=judge_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=_JudgeOutput,
            temperature=0,
        )
        parsed = result.choices[0].message.parsed
        if parsed is None:
            return None
        return JudgeResult(
            ui_optimality=parsed.ui_optimality,
            content_informativeness=parsed.content_informativeness,
            prompt_following=parsed.prompt_following,
            reasoning=parsed.reasoning,
        )
    except Exception as e:
        return JudgeResult(
            ui_optimality=0,
            content_informativeness=0,
            prompt_following=0,
            reasoning=f"Judge error: {e}",
        )
