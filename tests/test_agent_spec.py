"""AgentSpec v1 and its deterministic validator (ARCHIE-320)."""

import pytest
import app.agents.swarm.tools  # noqa: F401  registers the swarm tool group
from app.platform.runtime.budget import Budget
from app.platform.runtime.spawn import SpawnContext
from app.platform.runtime.spec import (
    AgentSpec,
    CallerPolicy,
    Execution,
    SpecValidationError,
    validate_spec,
)
from app.tools.catalog import tool_registry


def draft(**updates) -> dict:
    spec = {
        "identity": {
            "name": "researcher",
            "role": "Researcher",
            "purpose": "Find facts",
        },
        "objective": {
            "task": "Compare two places",
            "completion_criteria": ["Both rated"],
        },
        "instructions": "Use the places search.",
        "model": {"profile": "BALANCED", "effort": "medium", "reason": "two sources"},
        "tools": ["google_places_search_tool"],
        "output": {"schema_name": "findings_v1"},
    }
    spec.update(updates)
    return spec


def policy(**updates) -> CallerPolicy:
    values = {
        "tools": ["google_places_search_tool", "google_search_tool", "notes_tool"],
        "budget": Budget(
            max_tokens=20000, max_cost_usd=0.05, max_seconds=60, max_tool_rounds=3
        ),
        "limits": Execution(
            max_iterations=3, max_tool_calls=6, timeout_seconds=60, retries=1
        ),
        "max_profile": "BALANCED",
    }
    values.update(updates)
    caller = SpawnContext(source="chat", parent_run_id="turn-1")
    return CallerPolicy.from_spawn(caller, **values)


def test_valid_spec_passes_with_version_and_fingerprint():
    result = validate_spec(draft(), policy(), tool_registry)
    assert result.spec.spec_version == "1"
    assert len(result.spec.fingerprint()) == 16
    assert result.spec.budget.max_cost_usd == 0.05


@pytest.mark.parametrize(
    "updates",
    [
        {"code": "import os"},
        {"output": {"schema_name": "findings_v1", "json_schema": {"type": "object"}}},
        {"output": {"schema_name": "my schema"}},
        {"spec_version": "2"},
        {"model": {"profile": "GODLIKE"}},
    ],
)
def test_strict_schema_rejects_extra_fields_schemas_and_code(updates):
    with pytest.raises(SpecValidationError):
        validate_spec(draft(**updates), policy(), tool_registry)


@pytest.mark.parametrize(
    "updates,error",
    [
        ({"tools": ["unknown_tool"]}, "unknown tool"),
        ({"tools": ["smarthome_status_tool"]}, "not allowed"),
        ({"tools": ["notes_tool"]}, "approval"),
        ({"tools": ["spawn_agent_tool"]}, "recursive"),
        ({"skills": ["no_such_skill"]}, "unknown skill"),
        ({"output": {"schema_name": "secret_v1"}}, "unknown output schema"),
    ],
)
def test_validator_rejects_what_the_caller_cannot_grant(updates, error):
    with pytest.raises(SpecValidationError, match=error):
        validate_spec(draft(**updates), policy(), tool_registry, skills=["football"])


def test_write_tool_needs_approval_right_and_listing():
    spec = draft(
        tools=["notes_tool"],
        permissions={"mode": "propose_actions", "approval_tools": ["notes_tool"]},
    )
    with pytest.raises(SpecValidationError, match="approval"):
        validate_spec(spec, policy(), tool_registry)
    result = validate_spec(spec, policy(may_request_approval=True), tool_registry)
    assert result.spec.permissions.approval_tools == ["notes_tool"]
    unlisted = draft(tools=["notes_tool"], permissions={"mode": "propose_actions"})
    with pytest.raises(SpecValidationError, match="approval_tools"):
        validate_spec(unlisted, policy(may_request_approval=True), tool_registry)


def test_budget_limits_and_profile_are_capped_to_the_caller():
    spec = draft(
        budget={"max_cost_usd": 5.0, "max_tokens": None},
        execution={
            "max_iterations": 8,
            "max_tool_calls": 24,
            "timeout_seconds": 600,
            "retries": 3,
        },
        model={"profile": "MAXIMUM"},
    )
    result = validate_spec(spec, policy(), tool_registry)
    assert result.spec.budget.max_cost_usd == 0.05
    assert result.spec.budget.max_tokens == 20000
    assert result.spec.execution.max_iterations == 3
    assert result.spec.execution.timeout_seconds == 60
    assert result.spec.model.profile == "BALANCED"
    assert len(result.adjustments) == 3


def test_depth_limit_and_delegation_tools_never_reach_policy():
    caller = SpawnContext(source="chat", parent_run_id="p", depth=1)
    deep = CallerPolicy.from_spawn(
        caller, ["delegate_agent_tool", "google_search_tool"], budget=Budget()
    )
    assert "delegate_agent_tool" not in deep.tools
    with pytest.raises(SpecValidationError, match="depth"):
        validate_spec(draft(tools=["google_search_tool"]), deep, tool_registry)


def test_spec_model_rejects_unknown_nested_fields():
    with pytest.raises(Exception):
        AgentSpec.model_validate(draft(context={"facts": [], "secrets": "x"}))


def test_effort_is_capped_with_the_profile():
    result = validate_spec(
        draft(model={"profile": "DEEP", "effort": "high"}),
        policy(max_effort="medium"),
        tool_registry,
    )
    assert result.spec.model.profile == "BALANCED"
    assert result.spec.model.effort == "medium"
