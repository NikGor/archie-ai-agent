"""Dynamic agents (ARCHIE-321, ARCHIE-323, ARCHIE-325): task -> configurator ->
validated AgentSpec -> runtime, with a run record for every step.

`DynamicAgents.delegate` is what a trusted caller (the chat tool) uses; it is
idempotent per parent, task and context and returns a summary with the run_id.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from langfuse import get_client
from app.agents.ephemeral.configurator import AgentConfigurator
from app.agents.ephemeral.engine import (
    AgentFactory,
    AgentRunResult,
    AgentRuntime,
    RunCancelledError,
)
from app.platform.runtime.runs import (
    AgentRunRecord,
    EventListener,
    RunManager,
    RunRecorder,
)
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
        configurator: AgentConfigurator | None = None,
    ) -> None:
        self.registry = registry or tool_registry
        self.manager = manager or RunManager()
        self.runtime = runtime or AgentRuntime(self.registry)
        self.factory = factory or AgentFactory(self.registry)
        self.configurator = configurator or AgentConfigurator(self.registry)

    @contextmanager
    def observe(self, recorder: RunRecorder) -> Iterator[None]:
        """One Langfuse agent-run with the configurator and the agent inside."""
        record = recorder.record
        with get_client().start_as_current_observation(
            as_type="agent",
            name="agent-run",
            input=record.task,
            metadata={"run_id": record.run_id, "parent_run_id": record.parent_run_id},
        ) as observation:
            record.trace_id = get_client().get_current_trace_id()
            try:
                yield
            finally:
                observation.update(
                    output=record.summary(),
                    metadata={
                        "run_id": record.run_id,
                        "parent_run_id": record.parent_run_id,
                        "spec_version": record.spec_version,
                        "spec_fingerprint": record.spec_fingerprint,
                        "prompt_version": record.prompt_version,
                        "model": record.model,
                        "state": record.state,
                        "cost_usd": record.cost_usd,
                    },
                )

    async def run_spec(
        self, recorder: RunRecorder, spec: AgentSpec | dict, policy: CallerPolicy
    ) -> None:
        """Validate, create and execute; the record gets every step."""
        try:
            validated = validate_spec(spec, policy, self.registry, skill_names())
        except SpecValidationError as exc:
            self._fail(recorder, "invalid_spec", str(exc), exc.errors)
            return
        agent = self.factory.create(
            validated.spec, ceiling=policy.max_profile, timezone=policy.user_timezone
        )
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
        try:
            result = await self.runtime.execute(agent, policy, record.run_id, recorder)
        except RunCancelledError as exc:
            record_result(recorder, exc.result)
            raise
        record_result(recorder, result)

    @staticmethod
    def _fail(
        recorder: RunRecorder, reason: str, detail: str, errors: list[str]
    ) -> None:
        recorder.record.stop_reason = reason
        recorder.record.detail = detail
        recorder.record.errors.extend(errors)
        recorder.state("failed")
        recorder.event("agent_failed", reason=reason, detail=detail)

    async def run(
        self, task: str, spec: AgentSpec | dict, policy: CallerPolicy, context: str = ""
    ) -> AgentRunRecord:
        """Run a given spec (fixed configuration); idempotent per parent and task."""
        recorder, _ = self.manager.create(
            task, policy.parent_run_id, policy.source, context
        )

        async def body(active: RunRecorder) -> None:
            with self.observe(active):
                await self.run_spec(active, spec, policy)

        return await self.manager.run(recorder, body)

    async def delegate(
        self,
        task: str,
        context: str,
        policy: CallerPolicy,
        listener: EventListener | None = None,
    ) -> AgentRunRecord:
        """Configurator, then the agent; `direct` creates no agent."""
        recorder, existing = self.manager.create(
            task, policy.parent_run_id, policy.source, context
        )
        if not existing:
            recorder.listener = listener

        async def body(active: RunRecorder) -> None:
            with self.observe(active):
                config = await self.configurator.configure(
                    task, context, policy, skill_names()
                )
                record = active.record
                record.configuration = config.model_dump(mode="json", exclude={"spec"})
                record.configurator_cost_usd = config.cost_usd
                record.cost_usd += config.cost_usd
                record.input_tokens += config.input_tokens
                record.output_tokens += config.output_tokens
                if config.decision == "direct":
                    active.event("agent_configured", decision="direct")
                    record.stop_reason = "direct"
                    record.detail = config.reason
                    active.state("completed")
                    active.event("agent_completed", decision="direct")
                    return
                if config.spec is None:
                    self._fail(active, "invalid_spec", config.reason, config.errors)
                    return
                await self.run_spec(active, config.spec, policy)

        return await self.manager.run(recorder, body)
