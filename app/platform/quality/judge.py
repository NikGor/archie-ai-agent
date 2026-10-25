"""Jev questions and grades for spec rules (ARCHIE-213, ARCHIE-296).

One Jev question per rule — followed / partly / violated / not applicable.
Jev is a decision model (OpenRouter Decisions API): typed answers with
probabilities, no free text. All questions of a record go in one request.
"""

from archie_shared.monitoring.models import MonitorGrade, MonitorRuleScore
from pydantic import BaseModel, Field
from ..llm.decisions import DecisionsError
from .rules import SpecRule


JudgeError = DecisionsError

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


class JudgeResult(BaseModel):
    rules: list[MonitorRuleScore] = Field(description="Score per judged rule")
    spec_versions: dict[str, str] = Field(
        description="Spec -> fingerprint of the version that produced the record "
        "(from its trace), else of the judged version"
    )
    model: str = Field(description="Jev version that answered")
    cost: float | None = Field(default=None, description="Cost in USD")


def worst_grade(scores: list[MonitorRuleScore]) -> MonitorGrade:
    grades = {score.grade for score in scores if score.grade}
    return next((grade for grade in GRADE_ORDER if grade in grades), "green")


def grade_from_probabilities(probabilities: dict[str, float]) -> MonitorGrade | None:
    """None when the rule does not apply to the record."""
    if max(probabilities, key=probabilities.get) == NOT_APPLICABLE:
        return None
    violated = probabilities.get(VIOLATED, 0.0)
    if violated >= _RED_FROM:
        return "red"
    if violated + probabilities.get(PARTLY, 0.0) >= _ORANGE_FROM:
        return "orange"
    return "green"


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
    """Style advice can make a record orange, never red."""
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
