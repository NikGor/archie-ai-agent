"""Judge for swarm agent messages on TypeSafe Jev (ARCHIE-297): one question
per swarm spec rule, the same questions and grades as for chat turns
(platform/quality/judge.py)."""

import json
import logging
from datetime import UTC, datetime
from archie_shared.monitoring.models import SwarmMonitorRecord, SwarmTurnEvaluation
from archie_shared.swarm.models import SwarmMessage
from app.agents.swarm.agents import SWARM_AGENTS
from app.agents.swarm.messages import format_chat_line
from app.platform.llm.decisions import decide as _decide
from app.platform.quality.judge import (
    JudgeError,
    build_questions,
    parse_answers,
    worst_grade,
)
from app.platform.quality.rules import SpecRule
from .rules import rules_for_message, spec_versions


logger = logging.getLogger(__name__)

_ARGUMENT_CHARS = 300


def _agent_line(agent_id: str) -> str:
    profile = SWARM_AGENTS.get(agent_id)
    if profile is None:
        return f"@{agent_id}"
    return (
        f"@{profile.agent_id} ({profile.display_name}, {profile.role} agent): "
        f"{profile.description}. Tools: {', '.join(profile.tools) or 'none'}"
    )


def _actions(message: SwarmMessage) -> str:
    if not message.actions:
        return "No actions with side effects."
    return "\n".join(
        f"- {action.tool_name}({action.arguments_json[:_ARGUMENT_CHARS]}) -> {action.status}"
        + (f": {action.summary}" if action.summary else "")
        for action in message.actions
    )


def _agent_message(message: SwarmMessage) -> str:
    if message.kind == "check":
        return "(no message: the check found nothing new and stayed silent)"
    tag = f" [{message.importance} alert]" if message.importance else ""
    return f"[{message.kind}]{tag} {message.text}"


def build_state(record: SwarmMonitorRecord) -> dict[str, str]:
    """Jev state for one swarm agent message."""
    message = record.message
    return {
        "system": (
            "Archie's swarm: small agents in one chat, each with its own area. "
            "An agent reacts to a message, may call tools, and writes a short reply; "
            "a watcher agent runs spontaneous checks and writes an alert only if "
            "something is new."
        ),
        "agent": _agent_line(message.sender),
        "swarm_chat": "\n".join(format_chat_line(m) for m in record.history)
        or "(empty)",
        "trigger": format_chat_line(record.trigger)
        if record.trigger
        else "(none: a spontaneous check or a scheduled alert)",
        "actions": _actions(message),
        "assistant_response": _agent_message(message),
    }


def rules_of(record: SwarmMonitorRecord, rule_keys: list[str] | None = None) -> list[SpecRule]:
    rules = rules_for_message(record.message, record.trigger)
    return [rule for rule in rules if rule_keys is None or rule.key in rule_keys]


async def judge_record(
    record: SwarmMonitorRecord,
    run_id: str,
    model: str,
    rule_keys: list[str] | None = None,
) -> SwarmTurnEvaluation:
    """Grade a swarm message by its specs' rules (all, or only `rule_keys`)."""
    rules = rules_of(record, rule_keys)
    data = await _decide(model, build_state(record), build_questions(rules))
    try:
        scores = parse_answers(rules, data["answers"])
    except (KeyError, ValueError) as exc:
        raise JudgeError(f"Jev {model} returned unexpected answers: {exc}") from exc
    message = record.message
    return SwarmTurnEvaluation(
        run_id=run_id,
        message_id=message.message_id,
        agent_id=message.sender,
        grade=worst_grade(scores),
        rules=scores,
        spec_versions=spec_versions(message, record.trigger),
        judge_model=data.get("model", model),
        judge_cost=data.get("usage", {}).get("cost"),
        evaluated_at=datetime.now(UTC).isoformat(),
    )


def evaluation_summary(evaluation: SwarmTurnEvaluation) -> str:
    return json.dumps(
        {r.rule_id: r.grade for r in evaluation.rules if r.grade in ("red", "orange")},
        ensure_ascii=False,
    )
