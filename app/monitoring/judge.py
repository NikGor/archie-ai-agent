"""LLM judge for stored chat turns (ARCHIE-213).

The judge sees the conversation so far, the judged response, the turn's tool and
skill calls from Langfuse, and the agent catalog (tools, skills, response format
purpose), and grades each criterion red / orange / green.
"""

import functools
import json
import logging
from typing import get_args
from archie_shared.monitoring.models import (
    MonitorCriterion,
    MonitorCriterionScore,
    MonitorGrade,
    MonitorMessage,
)
from pydantic import BaseModel, ConfigDict, Field
from app.agent.prompt_builder import PromptBuilder
from app.backend.llm.registry import get_client
from app.evals.prompt_descriptions import PROMPT_DESCRIPTIONS
from app.tools.tool_factory import ToolFactory
from app.utils.skill_utils import list_skills
from .traces import TraceTurn


logger = logging.getLogger(__name__)

GRADE_ORDER: tuple[MonitorGrade, ...] = ("red", "orange", "green")
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
    """The judge call failed or returned an invalid verdict."""


class CriterionVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grade: MonitorGrade | None = Field(
        description="null only if the criterion does not apply to this turn"
    )
    comment: str = Field(description="One short sentence in Russian")


class JudgeVerdict(BaseModel):
    """Structured judge output; field order makes the model reason first."""

    model_config = ConfigDict(extra="forbid")

    analysis: str = Field(
        description="What the user wanted, what the assistant did, what went wrong"
    )
    helpfulness: CriterionVerdict = Field(
        description="The response solves the user's request; no helpless refusal"
    )
    correctness: CriterionVerdict = Field(
        description="No wrong facts, invented data, leaked errors or contradictions"
    )
    tool_choice: CriterionVerdict = Field(
        description="Right tools with right arguments; no needed tool skipped, no useless calls"
    )
    skill_usage: CriterionVerdict = Field(
        description="A matching skill was loaded before acting; null if no skill fits"
    )
    format_fit: CriterionVerdict = Field(
        description="Response fits its format and purpose (UI elements, buttons, length)"
    )
    summary: str = Field(
        description="One or two sentences in Russian: the main problem, or why the turn is good"
    )

    def criteria(self) -> list[MonitorCriterionScore]:
        return [
            MonitorCriterionScore(
                criterion=name,
                grade=getattr(self, name).grade,
                comment=getattr(self, name).comment,
            )
            for name in get_args(MonitorCriterion)
        ]


def worst_grade(scores: list[MonitorCriterionScore]) -> MonitorGrade:
    """Overall grade of a turn: its worst graded criterion."""
    grades = {score.grade for score in scores if score.grade}
    return next((grade for grade in GRADE_ORDER if grade in grades), "green")


_SYSTEM_PROMPT = """\
You review answers of Archie, a personal AI assistant for a smart home and daily life.
Archie answers in three stages: Stage 1 picks tools (and loads skills with skill_loader_tool),
Stage 2 runs the tools, Stage 3 writes the response in the requested format.

Grade each criterion:
- green: good, nothing to fix.
- orange: partly good; a minor mistake, an extra or a missing step that hurts the answer a little.
- red: the user did not get what they asked for; a helpless refusal of a request the tools can serve;
  wrong or invented facts; a wrong action (for example the wrong device); a leaked error;
  an empty or broken UI; real-time data (weather, news, schedules, prices) given without a search tool.

Rules:
- Judge only the last assistant response. Use earlier messages as context.
- tool_choice and skill_usage: if "Tool calls" says the trace is not available, set grade to null.
  Then do not lower other grades because tool use is unknown; judge only what the response shows.
- skill_usage: set grade to null if no listed skill fits the request.
- A short answer without tools is green for small talk and for facts that do not change.
- A string that ends with "…" was shortened by the monitor, not by Archie. It is not a defect.
- Write all comments and the summary in Russian.

Agent catalog
Tools:
{tools}

Skills (load with skill_loader_tool):
{skills}
"""


@functools.cache
def _system_prompt() -> str:
    schemas = ToolFactory().get_tool_schemas(_CATALOG_MODEL, "ui_answer")
    tools = "\n".join(
        f"- {schema['name']}: {schema['description'].strip().splitlines()[0]}"
        for schema in schemas
    )
    skills = "\n".join(
        f"- {skill['name']}: {skill['description']}" for skill in list_skills()
    )
    return _SYSTEM_PROMPT.format(tools=tools, skills=skills or "- none")


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
    """Valid JSON of the response; never cut mid-structure (ARCHIE-213: a cut
    JSON made the judge report a broken UI)."""
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
        return "Trace is not available for this turn."
    if not trace.tool_calls:
        return "No tools were called."
    return "\n".join(
        f"- {call.name}({call.arguments[:_TOOL_ARGUMENT_CHARS]})"
        + ("" if call.success is None else f" -> success={call.success}")
        for call in trace.tool_calls
    )


def build_turn_prompt(
    messages: list[MonitorMessage], index: int, trace: TraceTurn | None
) -> str:
    """User prompt for the assistant message messages[index]."""
    response = messages[index]
    history = messages[max(0, index - _HISTORY_MESSAGES) : index]
    content = _content(response)
    content_format = content.get("content_format", "plain")
    purpose = PROMPT_DESCRIPTIONS.get(
        PromptBuilder.format_spec_name(content_format), ""
    )
    response_json = _response_json(content)
    lines = [f"[{m.created_at}] {m.role}: {_short_text(m)}" for m in history]
    return (
        f"Conversation before the response:\n{chr(10).join(lines) or '(empty)'}\n\n"
        f"Tool calls of this turn:\n{_format_tool_calls(trace)}\n\n"
        f"Response format: {content_format} — {purpose}\n"
        f"Assistant response [{response.created_at}] (Content JSON):\n{response_json}"
    )


async def judge_turn(prompt: str, model: str) -> tuple[JudgeVerdict, float | None]:
    """Verdict and its cost in USD (None if the provider does not report it)."""
    try:
        # Jev and other judge models are OpenRouter-only
        response = await get_client("openrouter").create_completion(
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": prompt},
            ],
            model=model,
            response_format=JudgeVerdict,
            observation_name="monitor-judge",
        )
        verdict = JudgeVerdict.model_validate_json(response.choices[0].message.content)
    except Exception as exc:
        raise JudgeError(f"Judge {model} failed: {exc}") from exc
    return verdict, getattr(response.usage, "cost", None)
