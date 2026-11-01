"""Agent systems: AgentSystemSpec validator and the four topologies (ARCHIE-338, ARCHIE-339)."""

import asyncio
import pytest
from app.agents.ephemeral.composer import SystemDraft, to_system
from app.agents.ephemeral.configurator import ConfigurationResult
from app.agents.ephemeral.system import SystemRuntime, result_facts
from app.agents.systems import SystemService, constructor_policy, constructor_tools
from app.platform.runtime.runs import AgentRunRecord
from app.platform.runtime.spec import DELEGATION_TOOLS, AgentSpec
from app.platform.runtime.system import (
    DIMENSIONS,
    AgentSystemSpec,
    check_system,
    validate_system,
)
from app.tools.catalog import tool_registry


def agent(name: str = "researcher", **updates) -> dict:
    spec = {
        "identity": {"name": name, "role": "Researcher", "purpose": "Find facts"},
        "objective": {"task": "Find facts", "completion_criteria": ["Facts found"]},
        "model": {"profile": "BALANCED", "effort": "medium"},
        "tools": ["google_search_tool"],
        "budget": {"max_cost_usd": 0.05},
        "output": {"schema_name": "findings_v1"},
    }
    spec.update(updates)
    return spec


def system(
    topology: str = "centralized", nodes: list[dict] | None = None, **updates
) -> dict:
    spec = {
        "name": "test_system",
        "topology": topology,
        "nodes": nodes or [{"id": "main", "role": "agent", "spec": agent()}],
    }
    spec.update(updates)
    return spec


def check(spec: dict):
    return check_system(spec, constructor_policy(False), tool_registry)


def test_centralized_system_is_valid():
    result = check(system())
    assert result.valid, result.errors
    assert result.spec.nodes[0].spec.identity.name == "researcher"


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"lifecycle": "durable"}, "lifecycle durable"),
        ({"activation": "event_driven"}, "activation event_driven"),
        ({"activation": "proactive"}, "activation proactive"),
        ({"autonomy": {"external_actions": "execute"}}, "autonomy execute"),
        ({"autonomy": {"set_goals": True}}, "autonomy set_goals"),
        ({"autonomy": {"spawn_agents": True}}, "autonomy spawn_agents"),
        ({"lifecycle": "continuous"}, "continuous needs activation scheduled"),
        ({"activation": "scheduled"}, "needs a schedule"),
        ({"dynamism": "static"}, "static needs a fixed task"),
    ],
)
def test_unavailable_options_are_errors(updates, error):
    result = check(system(**updates))
    assert not result.valid
    assert any(error in e for e in result.errors), result.errors


def test_every_unavailable_option_has_a_reason():
    for dimension in DIMENSIONS.values():
        for option in dimension.options:
            assert option.available == (not option.reason)


def test_scheduled_system_needs_valid_cron_and_task():
    bad = check(
        system(activation="scheduled", schedule={"cron": "every hour x"}, task="t")
    )
    assert any("invalid cron" in e for e in bad.errors)
    good = check(
        system(
            activation="scheduled",
            lifecycle="continuous",
            schedule={"cron": "0 * * * *"},
            task="Check the weather",
        )
    )
    assert good.valid, good.errors


def test_workflow_rejects_cycles_and_unknown_dependencies():
    nodes = [
        {"id": "a", "spec": agent("a"), "depends_on": ["b"]},
        {"id": "b", "spec": agent("b"), "depends_on": ["a"]},
    ]
    assert any("cycle" in e for e in check(system("workflow", nodes)).errors)
    nodes[1]["depends_on"] = ["x"]
    assert any(
        "unknown dependency x" in e for e in check(system("workflow", nodes)).errors
    )


def test_roles_must_fit_the_topology():
    nodes = [
        {"id": "boss", "role": "supervisor", "spec": agent("boss")},
        {"id": "p", "role": "peer", "spec": agent("p")},
    ]
    errors = check(system("hierarchical", nodes)).errors
    assert any("role peer does not fit" in e for e in errors)
    assert any("1 to 4 workers" in e for e in errors)


def test_vote_needs_an_aggregator():
    nodes = [
        {"id": "p1", "role": "peer", "spec": agent("p1")},
        {"id": "p2", "role": "peer", "spec": agent("p2")},
    ]
    result = check(system("decentralized", nodes, coordination={"protocol": "vote"}))
    assert any("vote needs an aggregator" in e for e in result.errors)


def test_node_specs_go_through_the_agent_validator():
    nodes = [
        {"id": "main", "role": "agent", "spec": agent(tools=["delegate_agent_tool"])}
    ]
    errors = check(system(nodes=nodes)).errors
    assert any(
        e.startswith("node main:") and "delegate_agent_tool" in e for e in errors
    )


def test_autonomy_none_forbids_proposals():
    spec = agent(
        tools=["notes_tool"],
        permissions={"mode": "propose_actions", "approval_tools": ["notes_tool"]},
    )
    nodes = [{"id": "main", "role": "agent", "spec": spec}]
    assert not check(system(nodes=nodes)).valid
    assert check(system(nodes=nodes, autonomy={"external_actions": "propose"})).valid


def test_without_planning_agents_get_one_tool_round():
    result = check(system(autonomy={"plan": False}))
    assert result.spec.nodes[0].spec.execution.max_iterations == 1


def test_dynamic_nodes_need_a_step_instead_of_a_spec():
    nodes = [{"id": "main", "role": "agent"}]
    assert any("needs an agent spec" in e for e in check(system(nodes=nodes)).errors)
    nodes[0]["step"] = "Find the facts"
    assert check(system(nodes=nodes, dynamism="dynamic")).valid


def test_constructor_never_grants_delegation_or_exclusive_tools():
    tools = set(constructor_tools())
    assert not tools & DELEGATION_TOOLS
    assert "agent_system_tool" not in tools
    assert "swarm_alert_tool" not in tools


def test_composer_draft_becomes_a_checked_system():
    node = {
        "name": "a",
        "role": "R",
        "purpose": "P",
        "completion_criteria": ["c"],
        "instructions": "i",
        "profile": "FAST",
        "effort": "low",
        "profile_reason": "",
        "tools": ["google_search_tool"],
        "skills": [],
        "facts": [],
        "output_schema": "findings_v1",
        "max_iterations": 2,
        "max_tool_calls": 2,
        "max_cost_usd": 0.02,
        "permissions_mode": "read_only",
        "approval_tools": [],
    }
    draft = SystemDraft.model_validate(
        {
            "name": "pair",
            "topology": "workflow",
            "reason": "two steps",
            "protocol": "independent",
            "debate_rounds": 0,
            "nodes": [
                {
                    "id": "find",
                    "role": "agent",
                    "step": "Find",
                    "depends_on": [],
                    "spec": node,
                },
                {
                    "id": "check",
                    "role": "agent",
                    "step": "Check",
                    "depends_on": ["find"],
                    "spec": {**node, "name": "b"},
                },
            ],
        }
    )
    policy = constructor_policy(False)
    result = check_system(to_system(draft, "Task", policy), policy, tool_registry)
    assert result.valid, result.errors
    assert result.spec.task == "Task"


class FakeConfigurator:
    async def configure(self, *_args, allow_direct=True):
        assert allow_direct is False
        return ConfigurationResult(
            decision="spawn",
            reason="dynamic",
            spec=AgentSpec.model_validate(agent("built")),
            cost_usd=0.001,
        )


class FakeAgents:
    """Records every step; replies by node phase."""

    def __init__(self, replies: dict[str, dict] | None = None) -> None:
        self.calls: list[tuple[str, AgentSpec]] = []
        self.replies = replies or {}
        self.configurator = FakeConfigurator()

    async def run(self, task, spec, policy, context=""):
        node, phase = context.split(":")[1:]
        self.calls.append((f"{node}:{phase}", spec))
        await asyncio.sleep(0)
        reply = self.replies.get(f"{node}:{phase}") or {
            "summary": f"{node} {phase} done",
            "findings": [f"{node} fact"],
            "proposed_actions": [],
        }
        return AgentRunRecord(
            run_id=f"{node}-{phase}",
            parent_run_id=policy.parent_run_id,
            idempotency_key=context,
            source=policy.source,
            task=task,
            state="completed",
            result=reply,
            cost_usd=0.01,
        )


async def run_system(spec: dict, agents: FakeAgents, task: str = "Task"):
    policy = constructor_policy(True)
    validated = validate_system(spec, policy, tool_registry)
    runtime = SystemRuntime(agents)
    return await runtime.run(validated, task, policy, "manual")


async def test_workflow_runs_dependencies_first_and_passes_results():
    nodes = [
        {"id": "a", "spec": agent("a")},
        {"id": "b", "spec": agent("b")},
        {"id": "c", "spec": agent("c"), "depends_on": ["a", "b"]},
    ]
    agents = FakeAgents()
    record = await run_system(system("workflow", nodes), agents)
    order = [name for name, _ in agents.calls]
    assert order.index("c:step") == 2
    facts = agents.calls[2][1].context.facts
    assert any(f.startswith("a (completed)") for f in facts)
    assert record.state == "completed"
    assert record.result["summary"] == "c step done"
    assert record.cost_usd == pytest.approx(0.03)


async def test_hierarchical_plans_works_and_combines():
    nodes = [
        {"id": "boss", "role": "supervisor", "spec": agent("boss")},
        {"id": "w1", "role": "worker", "spec": agent("w1")},
        {"id": "w2", "role": "worker", "spec": agent("w2")},
    ]
    plan = {
        "summary": "plan",
        "assignments": [
            {"worker": "w2", "task": "Check prices"},
            {"worker": "ghost", "task": "x"},
        ],
        "proposed_actions": [],
    }
    agents = FakeAgents({"boss:plan": plan})
    record = await run_system(system("hierarchical", nodes), agents)
    names = [name for name, _ in agents.calls]
    assert names == ["boss:plan", "w2:work", "boss:synthesis"]
    assert agents.calls[0][1].output.schema_name == "plan_v1"
    assert agents.calls[1][1].objective.task.startswith("Check prices")
    assert record.state == "completed"


async def test_decentralized_debate_then_vote():
    nodes = [
        {"id": "p1", "role": "peer", "spec": agent("p1")},
        {"id": "p2", "role": "peer", "spec": agent("p2")},
        {"id": "judge", "role": "aggregator", "spec": agent("judge")},
    ]
    spec = system(
        "decentralized", nodes, coordination={"protocol": "debate", "debate_rounds": 1}
    )
    agents = FakeAgents()
    record = await run_system(spec, agents)
    names = sorted(name for name, _ in agents.calls[:4])
    assert names == ["p1:round_0", "p1:round_1", "p2:round_0", "p2:round_1"]
    p1_round_1 = next(s for n, s in agents.calls if n == "p1:round_1")
    assert any(f.startswith("p2") for f in p1_round_1.context.facts)
    assert not any(f.startswith("p1") for f in p1_round_1.context.facts)
    assert agents.calls[-1][0] == "judge:aggregate"
    assert record.result["summary"] == "judge aggregate done"


async def test_dynamic_nodes_are_configured_before_the_run():
    spec = system(
        dynamism="dynamic", nodes=[{"id": "main", "role": "agent", "step": "Find"}]
    )
    agents = FakeAgents()
    record = await run_system(spec, agents)
    assert record.configured["main"]["identity"]["name"] == "built"
    assert record.cost_usd == pytest.approx(0.011)


async def test_system_budget_stops_the_next_step():
    nodes = [
        {"id": "a", "spec": agent("a")},
        {"id": "b", "spec": agent("b"), "depends_on": ["a"]},
    ]
    agents = FakeAgents()
    record = await run_system(
        system("workflow", nodes, budget={"max_cost_usd": 0.005}), agents
    )
    assert record.state == "failed"
    assert record.stop_reason == "budget_exhausted"
    assert [n for n, _ in agents.calls] == ["a:step"]


async def test_proposed_actions_wait_for_approval():
    reply = {
        "summary": "turn off",
        "findings": [],
        "proposed_actions": [
            {"tool_name": "smarthome_control_tool", "arguments": [], "reason": "empty"}
        ],
    }
    agents = FakeAgents({"main:run": reply})
    record = await run_system(system(), agents)
    assert record.state == "waiting_for_approval"
    assert record.proposed_actions[0]["node"] == "main"


def test_static_system_runs_its_fixed_task():
    spec = AgentSystemSpec.model_validate(system(dynamism="static", task="Fixed"))
    assert SystemService.task_of(spec, "Other") == "Fixed"
    with pytest.raises(ValueError):
        SystemService.task_of(AgentSystemSpec.model_validate(system()), " ")


def test_result_facts_are_short():
    run = AgentRunRecord(
        run_id="r",
        parent_run_id="p",
        idempotency_key="k",
        source="s",
        task="t",
        state="completed",
        result={"summary": "x" * 2000, "findings": ["f"] * 10},
    )
    facts = result_facts("node", run)
    assert len(facts) == 6
    assert all(len(f) <= 600 for f in facts)
