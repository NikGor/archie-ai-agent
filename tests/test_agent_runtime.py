"""Universal AgentSpec runtime and runs (ARCHIE-321, ARCHIE-322, ARCHIE-325)."""

import asyncio
import pytest
from archie_shared.chat.models import LllmTrace
from app.agents.ephemeral.dynamic import DynamicAgents
from app.agents.ephemeral.engine import AgentFactory, AgentRuntime
from app.platform.runtime.budget import Budget
from app.platform.runtime.completion import Completion, InvalidOutputError
from app.platform.runtime.results import (
    AnswerReply,
    AnswerTurn,
    CriterionCheck,
    FindingsReply,
    FindingsTurn,
)
from app.platform.runtime.runs import RunManager
from app.platform.runtime.spawn import SpawnContext
from app.platform.runtime.spec import CallerPolicy
from app.platform.tools.models import Parameter, ToolCall, ToolResult
from app.tools.catalog import tool_registry


def spec(**updates) -> dict:
    value = {
        "identity": {"name": "researcher", "role": "Researcher", "purpose": "Facts"},
        "objective": {
            "task": "Find the opening hours",
            "completion_criteria": ["Hours found"],
        },
        "instructions": "Search once, then answer.",
        "model": {"profile": "FAST"},
        "tools": ["google_search_tool"],
        "output": {"schema_name": "findings_v1"},
    }
    value.update(updates)
    return value


def policy(**updates) -> CallerPolicy:
    values = {
        "budget": Budget(
            max_tokens=50000, max_cost_usd=0.05, max_seconds=30, max_tool_rounds=3
        ),
        "max_profile": "BALANCED",
        "may_request_approval": True,
    }
    values.update(updates)
    return CallerPolicy.from_spawn(
        SpawnContext(source="chat", parent_run_id="turn-1"),
        [
            "google_search_tool",
            "google_places_search_tool",
            "notes_tool",
            "events_tool",
        ],
        **values,
    )


def done(content, cost=0.0001, tokens=100):
    return Completion(
        content=content,
        trace=LllmTrace(
            model="t",
            input_tokens=tokens,
            output_tokens=0,
            total_tokens=tokens,
            total_cost=cost,
        ),
    )


def met(text="Hours found"):
    return [CriterionCheck(criterion=text, met=True, evidence="search result")]


def search(query="hours"):
    return ToolCall(
        tool_name="google_search_tool", arguments=[Parameter(name="query", value=query)]
    )


def findings_turn(*calls):
    return FindingsTurn(
        summary="",
        findings=[],
        criteria=[],
        proposed_actions=[],
        tool_calls=list(calls),
    )


def findings_reply(**updates):
    values = {
        "summary": "Open 9-18",
        "findings": ["9-18"],
        "criteria": met(),
        "proposed_actions": [],
    }
    values.update(updates)
    return FindingsReply(**values)


class Script:
    """Fake LLM: returns prepared replies and records the models it was called with."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.models = []
        self.efforts = []

    async def __call__(self, messages, schema, model, cap, effort=None):
        self.models.append(model)
        self.efforts.append(effort)
        self.caps = getattr(self, "caps", []) + [cap]
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return await reply()
        return reply if isinstance(reply, Completion) else done(reply)


def tools_result(output=None):
    async def run(calls, _context):
        return [
            ToolResult(tool_name=c.tool_name, output=output or {"text": "9-18"})
            for c in calls
        ]

    return run


def service(script, run_tools=None, manager=None):
    runtime = AgentRuntime(tool_registry, script, run_tools or tools_result())
    return DynamicAgents(
        manager=manager or RunManager(),
        runtime=runtime,
        factory=AgentFactory(tool_registry),
    )


async def test_two_specs_run_on_one_class_without_new_code():
    script = Script(
        findings_turn(search()),
        findings_reply(),
        AnswerTurn(
            summary="Paris",
            answer="Paris",
            sources=["fact"],
            criteria=met("Capital named"),
            proposed_actions=[],
            tool_calls=[],
        ),
    )
    agents = service(script)
    first = await agents.run("Find the opening hours", spec(), policy())
    second = await agents.run(
        "Name the capital",
        spec(
            identity={"name": "answerer", "role": "Answerer", "purpose": "One answer"},
            objective={
                "task": "Name the capital",
                "completion_criteria": ["Capital named"],
            },
            tools=[],
            output={"schema_name": "answer_v1"},
        ),
        policy(),
    )
    assert first.state == "completed" and first.result["findings"] == ["9-18"]
    assert second.state == "completed" and second.result["answer"] == "Paris"
    assert first.tools_used == ["google_search_tool"] and not second.tools_used
    assert type(agents.runtime) is AgentRuntime


async def test_parallel_runs_do_not_share_evidence_or_budget():
    gate = asyncio.Event()

    async def slow():
        await gate.wait()
        return done(findings_reply(), cost=0.002)

    agents = service(
        Script(findings_turn(search("a")), slow, findings_turn(search("b")), slow)
    )
    first = asyncio.create_task(agents.run("task a", spec(), policy()))
    second = asyncio.create_task(agents.run("task b", spec(), policy()))
    await asyncio.sleep(0.05)
    gate.set()
    a, b = await asyncio.gather(first, second)
    assert a.run_id != b.run_id
    assert len(a.evidence) == 1 and len(b.evidence) == 1
    assert a.cost_usd == pytest.approx(0.0021) and b.cost_usd == pytest.approx(0.0021)


@pytest.mark.parametrize(
    "updates,replies,reason",
    [
        ({"execution": {"timeout_seconds": 0.2}}, ["turn", "hang"], "timeout"),
        ({}, ["turn", "costly"], "budget_exhausted"),
        ({}, ["turn", "wordy"], "budget_exhausted"),
        ({"execution": {"max_tool_calls": 1}}, ["two_calls"], "max_tool_calls"),
    ],
)
async def test_limits_stop_the_run_with_state_reason_and_evidence(
    updates, replies, reason
):
    async def hang():
        await asyncio.Event().wait()

    prepared = {
        "hang": hang,
        "turn": findings_turn(search()),
        "reply": findings_reply(),
        "two_calls": findings_turn(search("a"), search("b")),
        "costly": done(findings_reply(), cost=1.0),
        "wordy": done(findings_reply(), tokens=10**6),
    }
    record = await service(Script(*[prepared[r] for r in replies])).run(
        "task", spec(**updates), policy()
    )
    assert record.state == "failed"
    assert record.stop_reason == reason
    kinds = [e.kind for e in record.events]
    assert kinds[-1] == "agent_failed"
    if reason in ("budget_exhausted", "timeout"):
        assert "budget_exhausted" in kinds and record.evidence


async def test_iteration_limit_forces_the_final_reply():
    script = Script(findings_turn(search("a")), findings_reply())
    record = await service(script).run(
        "task", spec(execution={"max_iterations": 1}), policy()
    )
    assert record.state == "completed" and record.iterations == 1


async def test_repeated_tool_call_stops_with_no_progress():
    script = Script(
        findings_turn(search()), findings_turn(search()), findings_turn(search())
    )
    record = await service(script).run(
        "task", spec(model={"profile": "BALANCED"}), policy()
    )
    assert record.state == "failed" and record.stop_reason == "no_progress"
    assert "loop_guard" in [e.kind for e in record.events]


async def test_same_delegation_in_the_same_parent_returns_the_existing_run():
    manager = RunManager()
    agents = service(Script(findings_turn(), findings_reply()), manager=manager)
    agents.runtime.call = Script(findings_reply())
    first = await agents.run("task", spec(), policy())
    second = await agents.run("task", spec(), policy())
    assert first.run_id == second.run_id
    other_parent = policy()
    other_parent.parent_run_id = "turn-2"
    agents.runtime.call = Script(findings_reply())
    third = await agents.run("task", spec(), other_parent)
    assert third.run_id != first.run_id


async def test_cancel_by_run_id_keeps_collected_evidence():
    entered = asyncio.Event()

    async def hang():
        entered.set()
        await asyncio.Event().wait()

    manager = RunManager()
    agents = service(Script(findings_turn(search()), hang), manager=manager)
    task = asyncio.create_task(agents.run("task", spec(), policy()))
    await entered.wait()
    run_id = next(iter(manager.tasks))
    assert await manager.cancel(run_id)
    record = await task
    assert record.state == "cancelled"
    assert record.evidence[0]["output"] == {"text": "9-18"}
    assert [e.kind for e in record.events][-1] == "agent_cancelled"


async def test_write_tool_never_runs_and_becomes_an_approval_request():
    executed = []

    async def run_tools(calls, _context):
        executed.extend(c.tool_name for c in calls)
        return []

    write = ToolCall(
        tool_name="notes_tool", arguments=[Parameter(name="action", value="create")]
    )
    script = Script(findings_turn(write), findings_reply())
    record = await service(script, run_tools).run(
        "task",
        spec(
            tools=["notes_tool"],
            permissions={"mode": "propose_actions", "approval_tools": ["notes_tool"]},
        ),
        policy(),
    )
    assert executed == []
    assert record.state == "waiting_for_approval"
    assert record.result["proposed_actions"][0]["tool_name"] == "notes_tool"
    assert "approval_requested" in [e.kind for e in record.events]


async def test_tool_outside_the_spec_is_blocked_at_execution():
    script = Script(
        findings_turn(ToolCall(tool_name="smarthome_control_tool", arguments=[])),
        findings_reply(),
    )
    record = await service(script).run("task", spec(), policy())
    assert record.policy_violations and record.evidence[0]["success"] is False


async def test_invalid_json_is_retried_then_escalated_one_step():
    script = Script(
        InvalidOutputError("bad json"), InvalidOutputError("bad json"), findings_reply()
    )
    record = await service(script).run(
        "task", spec(tools=[], execution={"retries": 1}), policy()
    )
    assert record.state == "completed"
    assert record.retries == 1
    assert record.escalations[0]["reason"] == "invalid_output"
    assert record.escalations[0]["to_profile"] == "BALANCED"
    assert "model_escalated" in [e.kind for e in record.events]


async def test_unmet_criteria_escalate_once_within_the_ceiling():
    unmet = findings_reply(
        criteria=[CriterionCheck(criterion="Hours found", met=False, evidence="none")]
    )
    script = Script(unmet, unmet)
    record = await service(script).run("task", spec(tools=[]), policy())
    assert [e["reason"] for e in record.escalations] == ["criteria_failed"]
    capped = await service(Script(unmet)).run(
        "task 2", spec(tools=[], model={"profile": "BALANCED"}), policy()
    )
    assert capped.escalations == []


async def test_invalid_spec_fails_the_run_with_reason():
    record = await service(Script()).run(
        "task", spec(tools=["smarthome_control_tool"]), policy()
    )
    assert record.state == "failed" and record.stop_reason == "invalid_spec"


async def test_record_has_model_choice_and_stats():
    manager = RunManager()
    script = Script(findings_reply())
    record = await service(script, manager=manager).run(
        "task", spec(tools=[]), policy()
    )
    assert record.model == "gpt-5.6-luna" and record.effort == "low"
    assert "profile FAST" in record.model_reason
    assert script.efforts == ["low"]
    assert record.spec_fingerprint and record.prompt_version
    stats = manager.stats()
    assert stats["runs"] == 1 and stats["success_rate"] == 1.0


async def test_runs_api_lists_shows_and_cancels(monkeypatch):
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    import app.platform.runtime.runs as runs
    from app.runtime_endpoints import router

    manager = RunManager()
    monkeypatch.setattr(runs, "_manager", manager)
    record = await service(Script(findings_reply()), manager=manager).run(
        "task", spec(tools=[]), policy()
    )
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    listed = client.get("/runtime/runs").json()["runs"]
    assert listed[0]["run_id"] == record.run_id and listed[0]["model"] == "gpt-5.6-luna"
    full = client.get(f"/runtime/runs/{record.run_id}").json()
    assert full["events"][0]["kind"] == "task_created"
    assert client.post(f"/runtime/runs/{record.run_id}/cancel").json()["cancelled"] is False
    assert client.get("/runtime/runs/nope").status_code == 404


async def test_reasoning_effort_raises_the_output_cap():
    script = Script(findings_reply())
    await service(script).run("task", spec(tools=[], model={"profile": "FAST"}), policy())
    assert script.efforts == ["low"] and script.caps[0] > 2048


async def test_truncated_stream_becomes_invalid_output_with_its_cost(monkeypatch):
    from app.platform.llm.openai_client import TruncatedOutputError
    from app.platform.runtime import completion

    class Client:
        async def create_completion_stream(self, usage_out, **_kwargs):
            usage_out.update({"input_tokens": 100, "output_tokens": 4096})
            raise TruncatedOutputError("cut off")
            yield ""

    monkeypatch.setattr(completion, "get_client", lambda _provider: Client())
    with pytest.raises(InvalidOutputError) as raised:
        await completion.complete([], FindingsReply, "gpt-5.4", 4096, "high")
    assert raised.value.trace.output_tokens == 4096


def test_only_the_first_json_object_of_a_reply_counts():
    from app.platform.runtime.completion import first_object

    assert first_object('{"a": 1}{"a": 2}', "m") == '{"a": 1}'
    assert first_object(' {"a": 1} ', "m") == ' {"a": 1} '
    assert first_object("not json", "m") == "not json"
