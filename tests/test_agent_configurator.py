"""AgentConfigurator: task -> validated AgentSpec or direct (ARCHIE-323)."""

from archie_shared.chat.models import LllmTrace
from app.agents.ephemeral.configurator import (
    AgentConfigurator,
    ConfigurationDraft,
    SpecDraft,
)
from app.agents.ephemeral.dynamic import DynamicAgents
from app.agents.ephemeral.engine import AgentRuntime
from app.platform.runtime.budget import Budget
from app.platform.runtime.completion import Completion
from app.platform.runtime.results import CriterionCheck, FindingsReply
from app.platform.runtime.spawn import SpawnContext
from app.platform.runtime.spec import CallerPolicy
from app.tools.catalog import tool_registry


def policy(**updates) -> CallerPolicy:
    values = {
        "budget": Budget(
            max_tokens=40000, max_cost_usd=0.05, max_seconds=60, max_tool_rounds=3
        ),
        "max_profile": "BALANCED",
    }
    values.update(updates)
    return CallerPolicy.from_spawn(
        SpawnContext(source="chat", parent_run_id="turn-1"),
        ["google_search_tool", "google_places_search_tool"],
        **values,
    )


def draft(**updates) -> SpecDraft:
    values = {
        "name": "place_compare",
        "role": "Researcher",
        "purpose": "Compare places",
        "completion_criteria": ["Both places rated"],
        "instructions": "Search each place, compare ratings.",
        "profile": "BALANCED",
        "effort": "medium",
        "profile_reason": "two sources",
        "tools": ["google_places_search_tool"],
        "skills": [],
        "facts": [],
        "output_schema": "findings_v1",
        "max_iterations": 3,
        "max_tool_calls": 4,
        "max_cost_usd": 0.03,
        "permissions_mode": "read_only",
        "approval_tools": [],
    }
    values.update(updates)
    return SpecDraft(**values)


class Fake:
    def __init__(self, *drafts):
        self.drafts = list(drafts)
        self.messages = []

    async def __call__(self, messages, schema, model, cap, effort=None):
        self.messages.append(list(messages))
        content = self.drafts.pop(0)
        return Completion(
            content=content,
            trace=LllmTrace(
                model=model,
                input_tokens=500,
                output_tokens=100,
                total_tokens=600,
                total_cost=0.001,
            ),
        )


async def configure(*drafts, caller=None):
    fake = Fake(*drafts)
    configurator = AgentConfigurator(tool_registry, fake)
    return await configurator.configure(
        "Compare two cafes", "", caller or policy(), []
    ), fake


async def test_spawn_returns_a_validated_spec_with_the_original_task():
    result, _ = await configure(
        ConfigurationDraft(decision="spawn", reason="two sources", spec=draft())
    )
    assert result.decision == "spawn"
    assert result.spec.objective.task == "Compare two cafes"
    assert result.spec.tools == ["google_places_search_tool"]
    assert result.cost_usd == 0.001 and result.attempts == 1


async def test_direct_creates_no_spec():
    result, _ = await configure(
        ConfigurationDraft(decision="direct", reason="one search", spec=None)
    )
    assert result.decision == "direct" and result.spec is None


async def test_llm_cannot_grant_tools_or_budget_beyond_the_caller():
    greedy = ConfigurationDraft(
        decision="spawn",
        reason="x",
        spec=draft(
            tools=["smarthome_control_tool"], max_cost_usd=10.0, profile="MAXIMUM"
        ),
    )
    fixed = ConfigurationDraft(
        decision="spawn", reason="x", spec=draft(max_cost_usd=10.0, profile="MAXIMUM")
    )
    result, fake = await configure(greedy, fixed)
    assert result.attempts == 2
    assert "Validation failed" in fake.messages[1][-1]["content"]
    assert result.spec.tools == ["google_places_search_tool"]
    assert result.spec.budget.max_cost_usd == 0.05
    assert result.spec.model.profile == "BALANCED"


async def test_two_invalid_drafts_fail_with_reason():
    bad = ConfigurationDraft(
        decision="spawn", reason="x", spec=draft(tools=["notes_tool"])
    )
    result, _ = await configure(bad, bad)
    assert result.decision == "failed" and result.errors


def test_capabilities_list_only_tools_of_the_caller():
    caps = AgentConfigurator(tool_registry).capabilities(policy())
    assert {t["name"] for t in caps["tools"]} == {
        "google_search_tool",
        "google_places_search_tool",
    }
    assert caps["profiles"] == ["FAST", "BALANCED"]
    assert "ha_diagnosis_v1" not in {s["name"] for s in caps["output_schemas"]}


async def test_delegate_records_configurator_cost_and_runs_the_agent():
    fake = Fake(
        ConfigurationDraft(decision="spawn", reason="two sources", spec=draft(tools=[]))
    )

    async def agent_call(messages, schema, model, cap, effort=None):
        return Completion(
            content=FindingsReply(
                summary="A is better",
                findings=["A 4.8"],
                proposed_actions=[],
                criteria=[
                    CriterionCheck(
                        criterion="Both places rated", met=True, evidence="x"
                    )
                ],
            ),
            trace=LllmTrace(
                model=model,
                input_tokens=10,
                output_tokens=10,
                total_tokens=20,
                total_cost=0.002,
            ),
        )

    agents = DynamicAgents(
        runtime=AgentRuntime(tool_registry, agent_call),
        configurator=AgentConfigurator(tool_registry, fake),
    )
    record = await agents.delegate("Compare two cafes", "", policy())
    assert record.state == "completed"
    assert record.configuration["decision"] == "spawn"
    assert record.configurator_cost_usd == 0.001
    assert abs(record.cost_usd - 0.003) < 1e-9
    kinds = [e.kind for e in record.events]
    assert kinds == [
        "task_created",
        "agent_configured",
        "agent_started",
        "agent_completed",
    ]

    direct = DynamicAgents(
        configurator=AgentConfigurator(
            tool_registry,
            Fake(ConfigurationDraft(decision="direct", reason="one call", spec=None)),
        )
    )
    record = await direct.delegate("Weather now", "", policy())
    assert record.state == "completed" and record.stop_reason == "direct"
