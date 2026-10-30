"""Eval of the AgentConfigurator (ARCHIE-323): Configuration Validity, the
spawn | direct decision, Tool Selection Accuracy, the profile, and a Jev check
that the brief and completion criteria fit the task.

Cases are ephemeral-kind cases with response_format "configurator"; the input
is a ConfiguratorScenario as JSON. No agent runs and no tool executes.
"""

import json
from typing import Literal
from archie_shared.evals.models import EvalAssertionFailure
from pydantic import BaseModel, Field
from app.agents.ephemeral.configurator import AgentConfigurator, ConfigurationResult
from app.agents.ephemeral.dynamic import skill_names
from app.platform.llm.decisions import decide
from app.platform.llm.profiles import ModelProfile
from app.platform.runtime.budget import Budget
from app.platform.runtime.spawn import SpawnContext
from app.platform.runtime.spec import CallerPolicy


TARGET = "configurator"
JUDGE_MODEL = "~typesafe/jev-latest"


class ConfiguratorScenario(BaseModel):
    task: str = Field(description="Task given to the configurator")
    context: str = Field("", description="Facts of the caller")
    caller_tools: list[str] = Field(description="Tools the caller may grant")
    max_profile: ModelProfile = Field("BALANCED", description="Caller ceiling")
    may_request_approval: bool = Field(False, description="Caller can ask approval")
    expect_decision: Literal["spawn", "direct"] = Field(description="Right decision")
    expect_tools: list[str] = Field(
        default_factory=list, description="Tools a spawned spec must include"
    )
    expect_profiles: list[ModelProfile] = Field(
        default_factory=list, description="Acceptable profiles; empty: any"
    )


def scenario_policy(scenario: ConfiguratorScenario, run_id: str) -> CallerPolicy:
    return CallerPolicy.from_spawn(
        SpawnContext(source="eval", parent_run_id=run_id, demo_mode=True),
        scenario.caller_tools,
        max_profile=scenario.max_profile,
        may_request_approval=scenario.may_request_approval,
        budget=Budget(
            max_tokens=40000, max_cost_usd=0.05, max_seconds=90, max_tool_rounds=3
        ),
    )


def failure(key: str, expected, actual) -> EvalAssertionFailure:
    return EvalAssertionFailure(
        key=key,
        expected=json.dumps(expected, ensure_ascii=False),
        actual=json.dumps(actual, ensure_ascii=False),
    )


def check(
    scenario: ConfiguratorScenario, result: ConfigurationResult
) -> list[EvalAssertionFailure]:
    """Deterministic checks; the judge is separate."""
    failures = []
    if result.decision == "failed":
        failures.append(failure("configuration_valid", True, result.errors))
    if result.decision != scenario.expect_decision:
        failures.append(failure("decision", scenario.expect_decision, result.decision))
    if result.spec is None:
        return failures
    missing = set(scenario.expect_tools) - set(result.spec.tools)
    if missing:
        failures.append(failure("tools", scenario.expect_tools, result.spec.tools))
    if not set(result.spec.tools) <= set(scenario.caller_tools):
        failures.append(
            failure("tools_allowed", scenario.caller_tools, result.spec.tools)
        )
    profile = result.spec.model.profile
    if scenario.expect_profiles and profile not in scenario.expect_profiles:
        failures.append(failure("profile", scenario.expect_profiles, profile))
    return failures


async def judge_fit(
    scenario: ConfiguratorScenario, result: ConfigurationResult, model: str
) -> bool:
    """Jev: do the brief and the criteria fit this task?"""
    assert result.spec is not None
    data = await decide(
        model,
        {
            "task": scenario.task,
            "brief": result.spec.instructions,
            "completion_criteria": result.spec.objective.completion_criteria,
        },
        {
            "fit": {
                "type": "choice",
                "instructions": "Do the brief and the completion criteria fit this task, and can a reader check each criterion against a result?",
                "criteria": {
                    "fits": "Specific to the task; every criterion is checkable",
                    "generic": "Generic, off-task, or criteria cannot be checked",
                },
            }
        },
    )
    probabilities = data["answers"]["fit"]["probabilities"]
    return max(probabilities, key=probabilities.get) == "fits"


async def run_scenario(
    raw: str, run_id: str, model: str, judge: bool, judge_model: str | None
) -> tuple[ConfigurationResult, list[EvalAssertionFailure]]:
    scenario = ConfiguratorScenario.model_validate_json(raw)
    result = await AgentConfigurator().configure(
        scenario.task,
        scenario.context,
        scenario_policy(scenario, run_id),
        skill_names(),
        model,
    )
    failures = check(scenario, result)
    if (
        judge
        and result.spec is not None
        and not await judge_fit(scenario, result, judge_model or JUDGE_MODEL)
    ):
        failures.append(failure("brief_fits_task", "fits", "generic"))
    return result, failures
