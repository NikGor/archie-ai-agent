"""Fixed AgentSpec presets (ARCHIE-321). The HA diagnosis of the swarm runs on
the same runtime as a configured agent; it is also the fixed baseline of the
Dynamic vs Fixed benchmark (ARCHIE-326)."""

from app.agents.ephemeral.models import SpawnRequest
from app.platform.runtime.budget import Budget
from app.platform.runtime.spec import (
    AgentSpec,
    Execution,
    Identity,
    ModelChoice,
    Objective,
    OutputSpec,
)


HA_DIAGNOSIS_SPECS = ("ephemeral_base", "ephemeral_ha_diagnosis")
HA_FINAL_TASK = "Report the diagnosis and the evidence gaps. Do not take action."
HA_EVIDENCE_BYTES = 4000
TOOL_CALLS_PER_ROUND = 4


def ha_diagnosis_spec(request: SpawnRequest, budget: Budget) -> AgentSpec:
    """Preset spec; `budget` is already capped by the service."""
    rounds = budget.max_tool_rounds or 0
    return AgentSpec(
        identity=Identity(
            name="ha_diagnosis",
            role="Home Assistant diagnostician",
            purpose="Read-only diagnosis of a Home Assistant fault for the swarm",
        ),
        objective=Objective(
            task=request.goal,
            completion_criteria=["The diagnosis states its evidence and gaps"],
        ),
        model=ModelChoice(profile="BALANCED", reason="fixed preset"),
        tools=list(request.tools),
        context={"facts": [request.context] if request.context else []},
        execution=Execution(
            max_iterations=min(8, max(1, rounds)),
            max_tool_calls=min(24, TOOL_CALLS_PER_ROUND * max(1, rounds)),
            timeout_seconds=min(600, max(request.ttl, 0.001)),
            retries=0,
        ),
        budget=budget.model_copy(
            update={"max_seconds": min(budget.max_seconds, request.ttl)}
        ),
        output=OutputSpec(schema_name="ha_diagnosis_v1"),
    )
