"""Judge for stored chat turns on TypeSafe Jev (ARCHIE-213).

Each turn is judged against the rules of the prompt specs that produced it
(see spec_rules.py): one Jev question per rule — followed / partly / violated /
not applicable. Jev is a decision model (OpenRouter Decisions API): it returns
typed answers with probabilities, no free text. All questions of a turn go in
one request and are answered in parallel.
"""

import functools
import json
import logging
from archie_shared.monitoring.models import (
    MonitorGrade,
    MonitorMessage,
    MonitorRuleScore,
)
from pydantic import BaseModel, Field
from app.agent.prompt_builder import PromptBuilder
from app.evals.prompt_descriptions import PROMPT_DESCRIPTIONS
from app.platform.llm.decisions import DecisionsError as JudgeError
from app.platform.llm.decisions import decide as _decide
from app.agent.tool_access import chat_tool_names
from app.tools.catalog import tool_registry
from app.utils.skill_utils import list_skills
from .outcome import GOAL_KEY as OUTCOME_GOAL_KEY
from .outcome import TITLE_KEY, build_outcome_questions, parse_outcome, parse_title
from .spec_rules import SpecRule, rules_for_turn, spec_versions
from .traces import TraceTurn


logger = logging.getLogger(__name__)

GRADE_ORDER: tuple[MonitorGrade, ...] = ("red", "orange", "green")
FOLLOWED = "followed"
PARTLY = "partly"
VIOLATED = "violated"
NOT_APPLICABLE = "not_applicable"
ANSWERS = {
    FOLLOWED: "The response follows the rule",
    PARTLY: "The response follows the rule only in part",
    VIOLATED: "The response breaks the rule",
    NOT_APPLICABLE: "The rule does not concern this request or response",
}
# Probability -> grade, calibrated on 58 real turns (ARCHIE-213)
_RED_FROM = 0.7  # P(violated)
_ORANGE_FROM = 0.6  # P(violated) + P(partly)
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


class JudgeResult(BaseModel):
    rules: list[MonitorRuleScore] = Field(description="Score per judged rule")
    spec_versions: dict[str, str] = Field(
        description="Spec -> fingerprint of the version that produced the turn "
        "(from its trace), else of the judged version"
    )
    model: str = Field(description="Jev version that answered")
    cost: float | None = Field(default=None, description="Cost in USD")


def worst_grade(scores: list[MonitorRuleScore]) -> MonitorGrade:
    """Overall grade of a turn: its worst graded rule."""
    grades = {score.grade for score in scores if score.grade}
    return next((grade for grade in GRADE_ORDER if grade in grades), "green")


def grade_from_probabilities(probabilities: dict[str, float]) -> MonitorGrade | None:
    if max(probabilities, key=probabilities.get) == NOT_APPLICABLE:
        return None
    violated = probabilities.get(VIOLATED, 0.0)
    if violated >= _RED_FROM:
        return "red"
    if violated + probabilities.get(PARTLY, 0.0) >= _ORANGE_FROM:
        return "orange"
    return "green"


@functools.cache
def _tool_lines() -> str:
    schemas = tool_registry.schemas(_CATALOG_MODEL, chat_tool_names("ui_answer"))
    return "\n".join(
        f"- {schema['name']}: {schema['description'].strip().splitlines()[0]}"
        for schema in schemas
    )


@functools.cache
def _current_skill_lines() -> tuple[str, ...]:
    return tuple(
        f"- {skill['name']}: {skill['description']}" for skill in list_skills()
    )


def agent_catalog(skills: list[str] | None = None) -> str:
    """Tools and skills for the judge. `skills`: the skill lines of the turn's
    Stage 1 prompt; None (no Stage 1 call in the trace) uses the current ones."""
    skill_lines = "\n".join(_current_skill_lines() if skills is None else skills)
    return (
        "Archie is a personal AI assistant for a smart home and daily life. "
        "Stage 1 picks tools and loads skills, Stage 2 runs tools, Stage 3 writes "
        f"the response.\nTools:\n{_tool_lines()}\nSkills:\n{skill_lines or '- none'}"
    )


def turn_spec_versions(content_format: str, trace: TraceTurn | None) -> dict[str, str]:
    """Spec -> version that produced the turn, from its trace; else the judged
    version (ARCHIE-256: a backlog run must not put old turns on the new version)."""
    produced = trace.spec_versions if trace else {}
    return {
        spec: produced.get(spec, judged)
        for spec, judged in spec_versions(content_format, trace is not None).items()
    }


def _compact(value: object, limit: int = _STRING_LIMITS[0]) -> object:
    """Content without empty and visual-only fields, strings cut to `limit`."""
    if isinstance(value, dict):
        return {
            key: _compact(item, limit)
            for key, item in value.items()
            if key not in _SKIPPED_KEYS
            # Link fields (open_map_url, ...) drive buttons and are never shown as
            # text; the judge read them as raw URLs in the answer
            and not key.endswith("url") and item not in (None, "", [], {})
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


def judged_format(content: dict, requested: str | None = None) -> str:
    """The response format the turn was produced in.

    Messages store the requested format since ARCHIE-213; older ones do not.
    For those it is recognized from the text: voice answers are stored as
    content_format "plain" but were produced by the ssml or gemini_tts spec, and
    judging them by format_plain flagged their markup as broken rules.
    """
    if requested:
        return requested
    content_format = content.get("content_format", "plain")
    text = (content.get("text") or "").lstrip()
    if content_format == "plain" and text.startswith("<speak"):
        return "ssml"
    if content_format == "plain" and "# Director's note" in text:
        return "gemini_tts"
    return content_format


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
    messages: list[MonitorMessage],
    index: int,
    trace: TraceTurn | None,
    title: str | None = None,
) -> dict[str, str]:
    """Jev state for the assistant message messages[index]."""
    response = messages[index]
    history = messages[max(0, index - _HISTORY_MESSAGES) : index]
    content = _content(response)
    content_format = judged_format(content, response.response_format)
    purpose = PROMPT_DESCRIPTIONS.get(
        PromptBuilder.format_spec_name(content_format), ""
    )
    user_messages = [m for m in history if m.role == "user"]
    next_message = _next_user_message(messages, index)
    return {
        **({"conversation_title": title} if title else {}),
        "agent_catalog": agent_catalog(trace.skills_offered if trace else None),
        "conversation": "\n".join(
            f"[{m.created_at}] {m.role}: {_short_text(m)}" for m in history
        )
        or "(empty)",
        # Separate field: with the whole history only, Jev mixed up which message
        # the response answers (e.g. the language of an older message)
        "last_user_message": _short_text(user_messages[-1]) if user_messages else "",
        "next_user_message": _short_text(next_message) if next_message else "(none)",
        "tool_calls": _format_tool_calls(trace),
        "response_format": f"{content_format}: {purpose}",
        "assistant_response": _response_json(content),
    }


def _next_user_message(
    messages: list[MonitorMessage], index: int
) -> MonitorMessage | None:
    """The user's reply to messages[index], if the conversation has one."""
    following = messages[index + 1 : index + 2]
    return following[0] if following and following[0].role == "user" else None


def _question_key(index: int) -> str:
    # Positional keys: rule ids may repeat across specs and contain any characters
    return f"rule_{index}"


def _condition_note(rule: SpecRule) -> str:
    if rule.condition is None:
        return ""
    return (
        f"The rule applies only if {rule.condition}. "
        f"If this is not the case, answer: {NOT_APPLICABLE}.\n"
    )


def build_questions(rules: list[SpecRule]) -> dict[str, dict]:
    return {
        _question_key(index): {
            "type": "choice",
            "instructions": (
                f"Rule from the {rule.spec} prompt that the assistant had to follow:\n"
                f"{rule.text}\n{_condition_note(rule)}"
                "Did the assistant response follow this rule?"
            ),
            "criteria": ANSWERS,
        }
        for index, rule in enumerate(rules)
    }


def _cap_guidance(grade: MonitorGrade | None, rule: SpecRule) -> MonitorGrade | None:
    """Style advice can make a turn orange, never red."""
    return "orange" if rule.guidance and grade == "red" else grade


def parse_answers(rules: list[SpecRule], answers: dict) -> list[MonitorRuleScore]:
    scores = []
    for index, rule in enumerate(rules):
        probabilities = answers[_question_key(index)]["probabilities"]
        scores.append(
            MonitorRuleScore(
                spec=rule.spec,
                rule_id=rule.rule_id,
                text=rule.text,
                grade=_cap_guidance(grade_from_probabilities(probabilities), rule),
                probability=round(max(probabilities.values()), 3),
            )
        )
    return scores


def _is_first_assistant_turn(messages: list[MonitorMessage], index: int) -> bool:
    return all(m.role != "assistant" for m in messages[:index])


async def judge_turn(
    messages: list[MonitorMessage],
    index: int,
    trace: TraceTurn | None,
    model: str,
    title: str | None = None,
) -> JudgeResult:
    """`title`: the conversation title; judged on the first assistant turn only."""
    content_format = judged_format(
        _content(messages[index]), messages[index].response_format
    )
    user_messages = [m for m in messages[:index] if m.role == "user"]
    rules = rules_for_turn(
        content_format,
        trace is not None,
        _short_text(user_messages[-1]) if user_messages else "",
    )
    ask_title = bool(title) and _is_first_assistant_turn(messages, index)
    data = await _decide(
        model,
        build_state(messages, index, trace, title if ask_title else None),
        {
            **build_questions(rules),
            **build_outcome_questions(
                ANSWERS, _next_user_message(messages, index) is not None, ask_title
            ),
        },
    )
    answers = data["answers"]
    try:
        goal_grade = grade_from_probabilities(
            answers[OUTCOME_GOAL_KEY]["probabilities"]
        )
        title_grade = (
            grade_from_probabilities(answers[TITLE_KEY]["probabilities"])
            if TITLE_KEY in answers
            else None
        )
        scores = (
            parse_outcome(answers, goal_grade)
            + parse_title(answers, title_grade)
            + parse_answers(rules, answers)
        )
    except (KeyError, ValueError) as exc:
        raise JudgeError(f"Jev {model} returned unexpected answers: {exc}") from exc
    return JudgeResult(
        rules=scores,
        spec_versions=turn_spec_versions(content_format, trace),
        model=data.get("model", model),
        cost=data["usage"].get("cost"),
    )


async def judge_rules(
    messages: list[MonitorMessage],
    index: int,
    rule_keys: list[str],
    model: str,
    response_format: str,
) -> list[MonitorRuleScore]:
    """Grades of the given 'spec:rule_id' rules only (eval rules_followed check).

    `response_format` is the requested format: Content does not keep it (a
    markdown answer is stored as content_format "plain").
    Without a trace: only rules visible in the response (assistant_prompt and
    format specs) can be judged; other keys are skipped.
    """
    content_format = response_format
    rules = [
        rule
        for rule in rules_for_turn(content_format, trace_found=False)
        if rule.key in rule_keys
    ]
    if not rules:
        return []
    data = await _decide(
        model, build_state(messages, index, None), build_questions(rules)
    )
    try:
        return parse_answers(rules, data["answers"])
    except (KeyError, ValueError) as exc:
        raise JudgeError(f"Jev {model} returned unexpected answers: {exc}") from exc
