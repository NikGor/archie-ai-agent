"""Outcome checks of a turn, beside the prompt spec rules (ARCHIE-213).

Spec rules say how Archie must answer; outcome checks say what the user got.
A bad outcome with no broken rule points to a gap in the specs.

- goal: did the response solve the request, and if not, why (fixed reasons);
- user_reaction: what the user's next message says about the response — a
  rephrased request or a correction means the turn failed. Asked only when the
  next message exists.
"""

from archie_shared.monitoring.models import (
    MonitorGrade,
    MonitorOutcomeReason,
    MonitorRuleScore,
)


OUTCOME_SPEC = "outcome"
GOAL_RULE = "solves_request"
REACTION_RULE = "user_reaction"
GOAL_TEXT = "The response solves what the user asked in the last message."
REACTION_TEXT = (
    "The user's next message does not repeat the request or point out a mistake."
)
NO_PROBLEM = "no_problem"

GOAL_REASONS: dict[MonitorOutcomeReason, str] = {
    "helpless_refusal": "Refused or said it cannot, although the request can be served",
    "needless_question": "Asked back although the request was clear enough to act",
    "off_topic": "Answered a different question",
    "partial_answer": "Answered only part of the request",
    "ignored_context": "Ignored earlier messages, e.g. a follow-up like 'and tomorrow?'",
}

# User reaction -> grade and reason
REACTIONS: dict[str, tuple[str, MonitorGrade, MonitorOutcomeReason | None]] = {
    "satisfied": ("Thanks, agrees or confirms the answer worked", "green", None),
    "continued": ("Asks a follow-up that builds on the answer", "green", None),
    "new_topic": ("Starts a different, unrelated request", "green", None),
    "rephrased": (
        "Repeats or rephrases the same request, because the answer did not give it",
        "orange",
        "rephrased_request",
    ),
    "corrected": (
        "Says the answer is wrong, not what was asked, or complains",
        "red",
        "user_correction",
    ),
}

# Below this probability a rephrase / correction reading is too unsure to grade:
# a short follow-up like "and in Tbilisi?" was read as a rephrase at 0.37
_REACTION_MIN_PROBABILITY = 0.5

GOAL_KEY = "outcome_goal"
GOAL_REASON_KEY = "outcome_goal_reason"
REACTION_KEY = "outcome_reaction"


def build_outcome_questions(
    answers: dict[str, str], has_next_message: bool
) -> dict[str, dict]:
    """`answers`: the followed / partly / violated / not_applicable options."""
    questions = {
        GOAL_KEY: {
            "type": "choice",
            "instructions": f"{GOAL_TEXT} Did the response achieve this?",
            "criteria": answers,
        },
        GOAL_REASON_KEY: {
            "type": "choice",
            "instructions": "If the response did not fully solve the request: the main reason.",
            "criteria": {**GOAL_REASONS, NO_PROBLEM: "The request is solved"},
        },
    }
    if has_next_message:
        questions[REACTION_KEY] = {
            "type": "choice",
            "instructions": (
                "Read next_user_message: how did the user react to the assistant response?"
            ),
            "criteria": {name: text for name, (text, _, _) in REACTIONS.items()},
        }
    return questions


def parse_outcome(
    answers: dict, goal_grade: MonitorGrade | None
) -> list[MonitorRuleScore]:
    """Goal score (graded by the caller from its probabilities) and user reaction."""
    goal_probabilities = answers[GOAL_KEY]["probabilities"]
    reason = answers[GOAL_REASON_KEY]["choice"]
    scores = [
        MonitorRuleScore(
            spec=OUTCOME_SPEC,
            rule_id=GOAL_RULE,
            text=GOAL_TEXT,
            grade=goal_grade,
            probability=round(max(goal_probabilities.values()), 3),
            reason=reason
            if goal_grade in ("red", "orange") and reason != NO_PROBLEM
            else None,
        )
    ]
    if REACTION_KEY in answers:
        reaction = answers[REACTION_KEY]
        _, grade, reaction_reason = REACTIONS[reaction["choice"]]
        probability = max(reaction["probabilities"].values())
        if grade != "green" and probability < _REACTION_MIN_PROBABILITY:
            grade, reaction_reason = "green", None
        scores.append(
            MonitorRuleScore(
                spec=OUTCOME_SPEC,
                rule_id=REACTION_RULE,
                text=REACTION_TEXT,
                grade=grade,
                probability=round(probability, 3),
                reason=reaction_reason,
            )
        )
    else:
        scores.append(
            MonitorRuleScore(
                spec=OUTCOME_SPEC, rule_id=REACTION_RULE, text=REACTION_TEXT
            )
        )
    return scores
