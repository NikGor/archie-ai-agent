"""Swarm spec rules a swarm message is judged by (ARCHIE-297).

A swarm message is judged by the specs that produced it: swarm_base and the
agent's own spec for every agent message, swarm_watch for a check, the council
specs for opinions and decisions. Templated rules (roster, tools, intros) and
rules whose effect the journal does not show (reads, memory writes) are not
judged.
"""

import functools
from archie_shared.swarm.models import SwarmMessage
from app.agents.swarm.agents import SWARM_AGENTS
from app.agents.swarm.ticker import WATCH_AGENTS
from app.platform.quality.rules import (
    RulePolicy,
    SpecRule,
    library,
    select_spec_rules,
    spec_fingerprints,
)


BASE_SPEC = "swarm_base"
WATCH_SPEC = "swarm_watch"
COUNCIL_SPEC = "swarm_group_council"
OPINION_SPEC = "swarm_council_opinion"
CONSUL_SPEC = "swarm_consul"
COUNCIL_RULES_SPEC = "swarm_council_rules"
COUNCIL_ALERT_SPEC = "swarm_council_alert"

AGENT_KINDS = frozenset({"report", "action", "proposal"})
WATCH_KINDS = frozenset({"alert", "check"})

SWARM_RULE_POLICY = RulePolicy(
    excluded=frozenset(
        {
            # input format and job descriptions, not behavior
            "swarm_chat_input",
            "swarm_chat_job",
            "swarm_cron_job",
            "swarm_voice_job",
            "swarm_display_job",
            "swarm_marcus_knows",
            # council views (ARCHIE-318): judged through swarm_opinion_own_lens
            *(
                f"swarm_lens_{member}_{part}"
                for member in ("business", "bro", "flirty", "marcus")
                for part in ("value", "doubt", "trade")
            ),
            "swarm_alert_intro",
            # council mechanics of the structured output
            "swarm_consul_more_rounds",
            "swarm_consul_next_round_true",
            "swarm_consul_next_round_false",
            # reads and memory writes: the journal keeps only side-effect actions,
            # so "facts come from tools" cannot be checked without the reads
            "swarm_tools_then_reply",
            "swarm_no_invented_facts",
            "swarm_watch_fresh_data",
            "swarm_watch_today_note",
            "swarm_watch_follow_memory",
            "swarm_watch_remember_today",
            "swarm_watch_home_read",
            "swarm_watch_mail_search",
            "swarm_watch_mail_ids",
            "swarm_watch_calendar_read",
            "swarm_decision_remember",
        }
    ),
    guidance=frozenset({"swarm_reply_short", "swarm_no_raw_numbers", "swarm_persona_scope"}),
    conditions={
        "swarm_mention_to_act": "the agent is mentioned and asked to act",
        "swarm_plan_step_is_task": "the trigger is a plan step (task) for this agent",
        "swarm_empty_reply": "the agent has nothing new to add",
        "swarm_own_area": "the request is outside the agent's area",
        "swarm_watch_importance": "the check wrote an alert",
    },
)


def agent_spec(agent_id: str) -> str | None:
    name = f"swarm_agent_{agent_id}"
    return name if name in library().specs else None


def _is_alert_trigger(trigger: SwarmMessage | None) -> bool:
    return trigger is not None and trigger.kind == "alert"


def specs_for_message(message: SwarmMessage, trigger: SwarmMessage | None) -> list[str]:
    """Specs that produced a swarm agent message."""
    own = [spec for spec in [agent_spec(message.sender)] if spec]
    if message.kind == "opinion":
        return [BASE_SPEC, COUNCIL_SPEC, OPINION_SPEC]
    if message.kind == "decision":
        final = COUNCIL_ALERT_SPEC if _is_alert_trigger(trigger) else COUNCIL_RULES_SPEC
        return [BASE_SPEC, COUNCIL_SPEC, CONSUL_SPEC, final]
    if message.kind in WATCH_KINDS and message.sender in WATCH_AGENTS.values():
        return [BASE_SPEC, *own, WATCH_SPEC]
    return [BASE_SPEC, *own]


def judge_context(message: SwarmMessage, trigger: SwarmMessage | None) -> dict:
    """Values for the specs' `when` conditions."""
    task = next(
        (task for task, agent in WATCH_AGENTS.items() if agent == message.sender), None
    )
    return {
        "agent": SWARM_AGENTS.get(message.sender),
        "task": task,
        "alert": _is_alert_trigger(trigger),
        # A decision in the journal is the last word of the council
        "last": True,
    }


def rules_for_message(
    message: SwarmMessage, trigger: SwarmMessage | None
) -> list[SpecRule]:
    context = judge_context(message, trigger)
    return [
        rule
        for spec in specs_for_message(message, trigger)
        for rule in select_spec_rules(spec, context, SWARM_RULE_POLICY)
    ]


def spec_versions(message: SwarmMessage, trigger: SwarmMessage | None) -> dict[str, str]:
    return spec_fingerprints(specs_for_message(message, trigger))


def case_specs(agent_id: str, stage: str) -> list[str]:
    """Specs a swarm eval case runs: an agent turn or a check."""
    own = [spec for spec in [agent_spec(agent_id)] if spec]
    return [BASE_SPEC, *own, *([WATCH_SPEC] if stage == "swarm_watch" else [])]


@functools.cache
def case_rules(agent_id: str, stage: str, watch_task: str | None) -> tuple[SpecRule, ...]:
    """Judged rules a swarm case can protect."""
    context = {
        "agent": SWARM_AGENTS.get(agent_id),
        "task": watch_task,
        "alert": False,
        "last": True,
    }
    return tuple(
        rule
        for spec in case_specs(agent_id, stage)
        for rule in select_spec_rules(spec, context, SWARM_RULE_POLICY)
    )
