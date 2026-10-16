"""Judge for stored chat turns on TypeSafe Jev (ARCHIE-213).

Jev is a decision model (OpenRouter Decisions API): it gets the turn as state and
typed questions, and returns typed answers with probabilities — no free text.
Per criterion it answers a red / orange / green score and picks the main reason
from a fixed list. Tool and skill criteria are asked only when the turn's
Langfuse trace is known.
"""

import functools
import json
import logging
from dataclasses import dataclass
from typing import get_args
import httpx
from archie_shared.monitoring.models import (
    MonitorCriterion,
    MonitorCriterionScore,
    MonitorGrade,
    MonitorMessage,
    MonitorReason,
)
from pydantic import BaseModel, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from app.agent.prompt_builder import PromptBuilder
from app.config import settings
from app.evals.prompt_descriptions import PROMPT_DESCRIPTIONS
from app.tools.tool_factory import ToolFactory
from app.utils.skill_utils import list_skills
from .traces import TraceTurn


logger = logging.getLogger(__name__)

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
GRADE_ORDER: tuple[MonitorGrade, ...] = ("red", "orange", "green")
NO_PROBLEM = "no_problem"
SKILL_FITS_QUESTION = "skill_fits"
_HTTP_ATTEMPTS = 3
_HISTORY_MESSAGES = 6
_HISTORY_TEXT_CHARS = 400
_RESPONSE_CHARS = 10000
# Long strings are cut ever shorter until the response fits; the JSON stays valid
_STRING_LIMITS = (600, 300, 120)
_TOOL_ARGUMENT_CHARS = 300
# Stored UI fields that carry no meaning for the judge; image_prompt may hold
# base64 image data after the frontend generates the image (ARCHIE-173)
_SKIPPED_KEYS = {"image_prompt", "image_url", "icon", "color", "palette"}
# Catalog tool schemas are built for this model's parser; any OpenAI model works
_CATALOG_MODEL = "gpt-4.1"


class JudgeError(RuntimeError):
    """The decisions call failed or returned an unexpected answer."""


@dataclass(frozen=True)
class Criterion:
    """One judged criterion: the question, the three grade levels and the reasons."""

    question: str
    red: str
    orange: str
    green: str
    reasons: dict[MonitorReason, str]
    # Normalized Jev score (0 red .. 1 green) -> grade. Calibrated per criterion on
    # 58 real turns against an LLM judge (ARCHIE-213); Jev's score spread differs by question.
    red_below: float = 0.4
    orange_below: float = 0.7
    needs_trace: bool = False


CRITERIA: dict[MonitorCriterion, Criterion] = {
    "helpfulness": Criterion(
        question="Did the assistant response solve what the user asked in the last message?",
        red="The user did not get what they asked for",
        orange="Partly solved, or solved with extra friction",
        green="Fully solved",
        reasons={
            "helpless_refusal": "Refused or said it cannot, but the tools can serve the request",
            "needless_question": "Asked back although the request was clear enough to act",
            "off_topic": "Answered a different question",
            "partial_answer": "Answered only part of the request",
            "ignored_context": "Ignored earlier messages of the conversation",
            "wrong_language": "Answered in a different language than the user",
        },
        red_below=0.4,
        orange_below=0.7,
    ),
    "correctness": Criterion(
        question=(
            "Is the response free of visible errors? Judge only what the response itself "
            "shows; do not mark a fact wrong only because you cannot verify it."
        ),
        red="A clear factual error, made-up data presented as real, a leaked error or a wrong action",
        orange="A small inaccuracy",
        green="Nothing visibly wrong",
        reasons={
            "factual_error": "States a fact that is clearly wrong by common knowledge",
            "invented_data": (
                "Presents made-up data (names, numbers, results) as real; "
                "data the user asked to simulate or show as an example is fine"
            ),
            "realtime_without_search": (
                "Gives real-time data (weather, schedules, news, prices) "
                "although tool_calls show no search or data tool"
            ),
            "leaked_error": "Shows an error, stack trace or technical text to the user",
            "contradiction": "Contradicts itself, the tool results or the conversation",
            "wrong_action": "Did a different action than asked (for example the wrong device)",
        },
        red_below=0.25,
        orange_below=0.45,
    ),
    "tool_choice": Criterion(
        question="Did the assistant call the right tools with the right arguments (see tool_calls)?",
        red="A needed tool was not called, or the wrong tool was used",
        orange="An extra call or slightly wrong arguments",
        green="Right tools and arguments",
        reasons={
            "missing_tool": "A tool the request needs was not called",
            "wrong_tool": "Called a tool that does not fit the request",
            "bad_arguments": "Right tool, wrong or poor arguments",
            "extra_tool": "Called tools the request did not need",
            "tool_error_ignored": "A tool failed and the response hides or ignores it",
        },
        needs_trace=True,
    ),
    "skill_usage": Criterion(
        question=(
            "A skill from agent_catalog fits this request. Did the assistant load it with "
            "skill_loader_tool and follow it?"
        ),
        red="The fitting skill was not loaded",
        orange="Loaded, but the response does not follow it well",
        green="Loaded and followed",
        reasons={
            "skill_not_loaded": "The fitting skill was not loaded",
            "wrong_skill": "Loaded a skill that does not fit",
            "skill_not_followed": "Loaded the skill but the response ignores its rules",
        },
        needs_trace=True,
    ),
    "format_fit": Criterion(
        question=(
            "Does the response fit its response_format and purpose? A string that ends "
            "with '…' was shortened by the monitor; it is not a defect."
        ),
        red="Broken, empty or wrong format",
        orange="A minor format issue",
        green="Fits the format well",
        reasons={
            "broken_ui": "UI structure is broken or items are empty",
            "empty_response": "No useful content",
            "wrong_element": "A generic UI element where a specific one fits (for example weather_card)",
            "missing_quick_actions": "No useful quick action buttons where the format expects them",
            "too_long": "Too long or a wall of text for the format",
            "wrong_format": "Content does not match the response format",
        },
        red_below=0.35,
        orange_below=0.65,
    ),
}


class JudgeResult(BaseModel):
    criteria: list[MonitorCriterionScore] = Field(description="Score per criterion")
    model: str = Field(description="Jev version that answered")
    cost: float | None = Field(default=None, description="Cost in USD")


def worst_grade(scores: list[MonitorCriterionScore]) -> MonitorGrade:
    """Overall grade of a turn: its worst graded criterion."""
    grades = {score.grade for score in scores if score.grade}
    return next((grade for grade in GRADE_ORDER if grade in grades), "green")


# Reasons that need the turn's tool calls; not offered when the trace is unknown
_TRACE_REASONS: set[MonitorReason] = {"realtime_without_search"}


def grade_from_score(score: float, criterion: Criterion) -> MonitorGrade:
    if score < criterion.red_below:
        return "red"
    if score < criterion.orange_below:
        return "orange"
    return "green"


@functools.cache
def agent_catalog() -> str:
    schemas = ToolFactory().get_tool_schemas(_CATALOG_MODEL, "ui_answer")
    tools = "\n".join(
        f"- {schema['name']}: {schema['description'].strip().splitlines()[0]}"
        for schema in schemas
    )
    skills = "\n".join(
        f"- {skill['name']}: {skill['description']}" for skill in list_skills()
    )
    return (
        "Archie is a personal AI assistant for a smart home and daily life. "
        "Stage 1 picks tools and loads skills, Stage 2 runs tools, Stage 3 writes "
        f"the response.\nTools:\n{tools}\nSkills:\n{skills or '- none'}"
    )


def _compact(value: object, limit: int = _STRING_LIMITS[0]) -> object:
    """Content without empty and visual-only fields, strings cut to `limit`."""
    if isinstance(value, dict):
        return {
            key: _compact(item, limit)
            for key, item in value.items()
            if key not in _SKIPPED_KEYS and item not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [_compact(item, limit) for item in value]
    if isinstance(value, str) and len(value) > limit:
        return f"{value[:limit]}…"
    return value


def _response_json(content: dict) -> str:
    """Valid JSON of the response; never cut mid-structure (a cut JSON made the
    first judge report a broken UI)."""
    for limit in _STRING_LIMITS:
        text = json.dumps(_compact(content, limit), ensure_ascii=False)
        if len(text) <= _RESPONSE_CHARS:
            break
    return text


def _content(message: MonitorMessage) -> dict:
    return json.loads(message.content_json or "{}")


def _short_text(message: MonitorMessage) -> str:
    content = _content(message)
    text = content.get("text") or json.dumps(_compact(content), ensure_ascii=False)
    return text[:_HISTORY_TEXT_CHARS]


def _format_tool_calls(trace: TraceTurn | None) -> str:
    if trace is None:
        return "Unknown: no trace for this turn. Do not judge tool use."
    if not trace.tool_calls:
        return "No tools were called."
    return "\n".join(
        f"- {call.name}({call.arguments[:_TOOL_ARGUMENT_CHARS]})"
        + ("" if call.success is None else f" -> success={call.success}")
        for call in trace.tool_calls
    )


def build_state(
    messages: list[MonitorMessage], index: int, trace: TraceTurn | None
) -> dict[str, str]:
    """Jev state for the assistant message messages[index]."""
    response = messages[index]
    history = messages[max(0, index - _HISTORY_MESSAGES) : index]
    content = _content(response)
    content_format = content.get("content_format", "plain")
    purpose = PROMPT_DESCRIPTIONS.get(
        PromptBuilder.format_spec_name(content_format), ""
    )
    return {
        "agent_catalog": agent_catalog(),
        "conversation": "\n".join(
            f"[{m.created_at}] {m.role}: {_short_text(m)}" for m in history
        )
        or "(empty)",
        "tool_calls": _format_tool_calls(trace),
        "response_format": f"{content_format}: {purpose}",
        "assistant_response": _response_json(content),
    }


def build_questions(trace: TraceTurn | None) -> dict[str, dict]:
    """Score and reason questions per applicable criterion."""
    questions: dict[str, dict] = {}
    for name, criterion in CRITERIA.items():
        if criterion.needs_trace and trace is None:
            continue
        questions[name] = {
            "type": "score",
            "instructions": criterion.question,
            "criteria": [
                {"value": "red", "description": criterion.red},
                {"value": "orange", "description": criterion.orange},
                {"value": "green", "description": criterion.green},
            ],
        }
        reasons = {
            code: text
            for code, text in criterion.reasons.items()
            if trace is not None or code not in _TRACE_REASONS
        }
        questions[f"{name}_reason"] = {
            "type": "choice",
            "instructions": f"Main problem for this question: {criterion.question}",
            "criteria": {**reasons, NO_PROBLEM: "No problem"},
        }
    if trace is not None:
        questions[SKILL_FITS_QUESTION] = {
            "type": "noul",
            "instructions": "Does a skill from agent_catalog fit the user's request?",
            "criteria": {
                "true": "A listed skill fits",
                "false": "No listed skill fits",
            },
        }
    return questions


def parse_answers(answers: dict) -> list[MonitorCriterionScore]:
    skill_fits = answers.get(SKILL_FITS_QUESTION, {}).get("noul", 0.0) >= 0.5
    scores = []
    for name in get_args(MonitorCriterion):
        answer = answers.get(name)
        if answer is None or (name == "skill_usage" and not skill_fits):
            scores.append(MonitorCriterionScore(criterion=name))
            continue
        # Jev scores a 3-level scale as a position 0..2
        score = answer["score"] / (len(GRADE_ORDER) - 1)
        grade = grade_from_score(score, CRITERIA[name])
        reason = answers[f"{name}_reason"]["choice"]
        scores.append(
            MonitorCriterionScore(
                criterion=name,
                grade=grade,
                reason=None if grade == "green" or reason == NO_PROBLEM else reason,
                score=round(score, 3),
            )
        )
    return scores


async def judge_turn(
    messages: list[MonitorMessage], index: int, trace: TraceTurn | None, model: str
) -> JudgeResult:
    body = {
        "model": model,
        "state": build_state(messages, index, trace),
        "questions": build_questions(trace),
    }
    headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(_HTTP_ATTEMPTS),
                wait=wait_exponential(multiplier=1, max=10),
                retry=retry_if_exception_type(httpx.TransportError),
                reraise=True,
            ):
                with attempt:
                    response = await client.post(
                        DECISIONS_URL, json=body, headers=headers
                    )
            response.raise_for_status()
        data = response.json()
        criteria = parse_answers(data["answers"])
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise JudgeError(f"Jev {model} failed: {exc}") from exc
    return JudgeResult(
        criteria=criteria,
        model=data.get("model", model),
        cost=data["usage"].get("cost"),
    )
