"""Execute one finite read-only HA diagnosis as a preset on the AgentSpec
runtime (ARCHIE-300, ARCHIE-321) and keep evidence when a limit ends it."""

import asyncio
import json
import uuid
from app.agents.ephemeral.engine import (
    AgentFactory,
    AgentRunResult,
    AgentRuntime,
    RunCancelledError,
    ToolCallRunner,
    model_override,
)
from app.agents.ephemeral.messages import build_messages
from app.agents.ephemeral.models import (
    DEFAULT_COST_USD,
    DEFAULT_SECONDS,
    DEFAULT_TOKENS,
    DEFAULT_TOOL_ROUNDS,
    READ_TOOLS,
    EphemeralResult,
    EphemeralUsage,
    SpawnRequest,
    validate_budget_limits,
)
from app.agents.ephemeral.presets import (
    HA_DIAGNOSIS_SPECS,
    HA_EVIDENCE_BYTES,
    HA_FINAL_TASK,
    ha_diagnosis_spec,
)
from app.config import settings
from app.platform.prompts.compiler import PromptCompiler
from app.platform.runtime.budget import Budget
from app.platform.runtime.completion import (  # noqa: F401  re-exported for callers
    BudgetEstimateError,
    Completion,
    CompletionCall,
    complete,
    estimate_call,
)
from app.platform.runtime.spawn import SpawnContext
from app.platform.runtime.spec import CallerPolicy
from app.platform.tools.executor import execute_tool_calls
from app.platform.tools.models import ToolCall, ToolContext, ToolResult
from app.platform.tools.registry import ToolRegistry
from app.tools.catalog import tool_registry


LIMIT_STOPS = frozenset(
    {"timeout", "budget_exhausted", "max_tool_calls", "no_progress"}
)


class EphemeralCancelledError(asyncio.CancelledError):
    """Cancellation with retained evidence for terminal result publication."""

    def __init__(self, result: EphemeralResult) -> None:
        super().__init__("Ephemeral run cancelled")
        self.result = result


def capped_budget(requested: Budget, ceiling: Budget) -> Budget:
    """Missing requested limits never remove the service's mandatory caps."""
    validate_budget_limits(requested)
    validate_budget_limits(ceiling)
    values = {}
    defaults = Budget(
        max_tokens=DEFAULT_TOKENS,
        max_cost_usd=DEFAULT_COST_USD,
        max_seconds=DEFAULT_SECONDS,
        max_tool_rounds=DEFAULT_TOOL_ROUNDS,
    )
    for name, default in defaults.model_dump().items():
        cap = getattr(ceiling, name)
        cap = default if cap is None else min(cap, default)
        value = getattr(requested, name)
        values[name] = cap if value is None else min(value, cap)
    return Budget(**values)


class EphemeralRunner:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        call: CompletionCall | None = None,
        run_tools: ToolCallRunner | None = None,
        compiler: PromptCompiler | None = None,
        model: str | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.registry = registry or tool_registry
        self.compiler = compiler or PromptCompiler()
        self.factory = AgentFactory(self.registry, self.compiler)
        self.runtime = AgentRuntime(
            self.registry, call or complete, run_tools or self._run_tools
        )
        self.model = model or settings.swarm_model
        self.budget = budget or Budget()
        validate_budget_limits(self.budget)

    async def _run_tools(
        self, calls: list[ToolCall], context: ToolContext
    ) -> list[ToolResult]:
        return await execute_tool_calls(
            calls, self.registry, allowed=READ_TOOLS, context=context
        )

    async def run(
        self, request: SpawnRequest, caller: SpawnContext, run_id: str | None = None
    ) -> EphemeralResult:
        identifier = run_id or uuid.uuid4().hex[:16]
        budget = capped_budget(request.budget, self.budget)
        if not set(request.tools) <= READ_TOOLS:
            return self._result(
                identifier,
                caller,
                "error",
                "Diagnosis failed: PermissionError: Only approved read tools are allowed",
            )
        agent = self.factory.create(
            ha_diagnosis_spec(request, budget),
            selection=model_override(self.model),
            messages=build_messages(request, self.model, self.registry, self.compiler),
            base_specs=HA_DIAGNOSIS_SPECS,
            read_tools=READ_TOOLS & frozenset(request.tools),
            final_task=HA_FINAL_TASK,
            evidence_bytes=HA_EVIDENCE_BYTES,
            spend_kind="ephemeral",
        )
        policy = CallerPolicy.from_spawn(caller, READ_TOOLS, budget=budget)
        try:
            outcome = await self.runtime.execute(agent, policy, identifier)
        except RunCancelledError as exc:
            raise EphemeralCancelledError(self._map(exc.result, caller)) from None
        return self._map(outcome, caller)

    def _map(self, outcome: AgentRunResult, caller: SpawnContext) -> EphemeralResult:
        reply = outcome.reply or {}
        if outcome.state in ("completed", "waiting_for_approval"):
            status, summary = (
                "completed",
                reply.get("summary") or "No diagnosis was produced.",
            )
        elif outcome.state == "cancelled":
            status, summary = "partial", outcome.detail
        elif outcome.stop_reason in LIMIT_STOPS:
            status, summary = "partial", f"Stopped at the run limit: {outcome.detail}."
        else:
            status, summary = "error", f"Diagnosis failed: {outcome.detail}"
        findings = list(reply.get("findings") or [])
        if status != "completed":
            findings.extend(
                f"{item.tool_name}: {json.dumps(item.output, ensure_ascii=False)[:1600]}"
                for item in outcome.evidence
            )
        return self._result(
            outcome.run_id,
            caller,
            status,
            summary,
            findings=findings,
            evidence=outcome.evidence,
            usage=EphemeralUsage(
                input_tokens=outcome.usage.input_tokens,
                output_tokens=outcome.usage.output_tokens,
                total_cost=outcome.usage.cost_usd,
                seconds=outcome.usage.seconds,
                tool_rounds=outcome.usage.tool_rounds,
            ),
        )

    @staticmethod
    def _result(
        run_id: str, caller: SpawnContext, status: str, summary: str, **values
    ) -> EphemeralResult:
        return EphemeralResult(
            run_id=run_id,
            parent_run_id=caller.parent_run_id,
            status=status,
            summary=summary,
            **values,
        )
