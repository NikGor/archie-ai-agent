"""Dynamic agents (ARCHIE-321, ARCHIE-323, ARCHIE-325): task -> configurator ->
validated AgentSpec -> runtime, with a run record for every step.

`DynamicAgents.delegate` is what a trusted caller (the chat tool) uses; it is
idempotent per parent, task and context and returns a summary with the run_id.
"""

import logging
from langfuse import get_client
from app.agents.ephemeral.engine import (
    AgentFactory,
    AgentRunResult,
    AgentRuntime,
    RunCancelledError,
)
from app.platform.runtime.runs import AgentRunRecord, RunManager, RunRecorder
from app.platform.runtime.spec import (
    AgentSpec,
    CallerPolicy,
    SpecValidationError,
    validate_spec,
)
from app.platform.tools.registry import ToolRegistry
from app.tools.catalog import tool_registry
from app.utils.skill_utils import list_skills


logger = logging.getLogger(__name__)


def skill_names() -> list[str]:
    return [skill["name"] for skill in list_skills()]


def record_result(recorder: RunRecorder, result: AgentRunResult) -> None:
    """Copy the run outcome into the record and close it with its event."""
    record = recorder.record
    record.state = result.state
    record.stop_reason = result.stop_reason
    record.detail = result.detail
    record.model = result.selection.model
    record.effort = result.selection.effort
    record.model_reason = result.selection.reason
    record.escalations = [e.model_dump(mode="json") for e in result.escalations]
    record.tools_used = sorted({e.tool_name for e in result.evidence if e.success})
    record.llm_calls = result.usage.llm_calls
    record.tool_calls = result.usage.tool_calls
    record.iterations = result.usage.tool_rounds
    record.input_tokens += result.usage.input_tokens
    record.output_tokens += result.usage.output_tokens
    record.cost_usd += result.usage.cost_usd
    record.retries = result.retries
    record.policy_violations = result.policy_violations
    record.loop_guards = result.loop_guards
    if result.error_type:
        record.errors.append(result.detail)
    reply = dict(result.reply or {})
    reply["proposed_actions"] = [
        a.model_dump(mode="json") for a in result.proposed_actions
    ]
    record.result = reply
    record.evidence = [e.model_dump(mode="json") for e in result.evidence]
    recorder.state(result.state)
    if result.state == "completed":
        recorder.event("agent_completed", cost_usd=round(record.cost_usd, 5))
    elif result.state == "failed":
        recorder.event("agent_failed", reason=result.stop_reason, detail=result.detail)
    elif result.state == "cancelled":
        recorder.event("agent_cancelled")


class DynamicAgents:
    """Runs AgentSpec agents for trusted callers of this process."""

    def __init__(
        self,
        manager: RunManager | None = None,
        runtime: AgentRuntime | None = None,
        factory: AgentFactory | None = None,
        registry: ToolRegistry | None = None,
        configurator=None,
    ) -> None:
        self.registry = registry or tool_registry
        self.manager = manager or RunManager()
        self.runtime = runtime or AgentRuntime(self.registry)
        self.factory = factory or AgentFactory(self.registry)
        self.configurator = configurator

    async def run_spec(
        self, recorder: RunRecorder, spec: AgentSpec | dict, policy: CallerPolicy
    ) -> None:
        """Validate, create and execute; the record gets every step."""
        try:
            validated = validate_spec(spec, policy, self.registry, skill_names())
        except SpecValidationError as exc:
            recorder.record.stop_reason = "invalid_spec"
            recorder.record.detail = str(exc)
            recorder.record.errors.extend(exc.errors)
            recorder.state("failed")
            recorder.event("agent_failed", reason="invalid_spec", detail=str(exc))
            return
        agent = self.factory.create(validated.spec, ceiling=policy.max_profile)
        record = recorder.record
        record.spec = validated.spec.model_dump(mode="json")
        record.spec_version = validated.spec.spec_version
        record.spec_fingerprint = validated.spec.fingerprint()
        record.prompt_version = agent.prompt_version
        record.adjustments = validated.adjustments
        recorder.state("configured")
        recorder.event(
            "agent_configured",
            name=validated.spec.identity.name,
            profile=validated.spec.model.profile,
            tools=list(validated.spec.tools),
        )
        recorder.state("running")
        with get_client().start_as_current_observation(
            as_type="agent",
            name="agent-run",
            input=validated.spec.objective.task,
            metadata={
                "run_id": record.run_id,
                "parent_run_id": record.parent_run_id,
                "spec_version": record.spec_version,
                "spec_fingerprint": record.spec_fingerprint,
                "prompt_version": record.prompt_version,
            },
        ) as observation:
            record.trace_id = get_client().get_current_trace_id()
            try:
                result = await self.runtime.execute(
                    agent, policy, record.run_id, recorder
                )
            except RunCancelledError as exc:
                record_result(recorder, exc.result)
                observation.update(output=record.summary())
                raise
            record_result(recorder, result)
            observation.update(output=record.summary())

    async def run(
        self, task: str, spec: AgentSpec | dict, policy: CallerPolicy, context: str = ""
    ) -> AgentRunRecord:
        """Run a given spec (fixed configuration); idempotent per parent and task."""
        recorder, _ = self.manager.create(
            task, policy.parent_run_id, policy.source, context
        )

        async def body(active: RunRecorder) -> None:
            await self.run_spec(active, spec, policy)

        return await self.manager.run(recorder, body)
