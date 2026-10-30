"""One runtime for every AgentSpec (ARCHIE-321): AgentFactory -> AgentInstance,
AgentRuntime.execute -> AgentRunResult. Generalized from the ARCHIE-300 runner;
the HA diagnosis of the swarm is a fixed preset on the same runtime.

The tool policy is checked for each call right before it runs. Tools with side
effects never run here: a permitted one becomes a proposed action that waits
for the user's approval.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo
from archie_shared.chat.models import LllmTrace
from pydantic import BaseModel, ConfigDict, Field
from app.platform.llm.profiles import (
    Escalation,
    EscalationReason,
    ModelProfile,
    ModelSelection,
    next_profile,
    select_model,
)
from app.platform.prompts.compiler import PromptCompiler
from app.platform.runtime.budget import (
    Budget,
    BudgetExceededError,
    BudgetMeter,
    daily_spend,
)
from app.platform.runtime.completion import (
    BudgetEstimateError,
    Completion,
    CompletionCall,
    InvalidOutputError,
    check_remaining,
    complete,
    max_output_for,
    output_cap,
)
from app.platform.runtime.results import (
    CriterionCheck,
    ProposedAction,
    ResultSchema,
    result_schema,
)
from app.platform.runtime.runs import RunRecorder, RunState
from app.platform.runtime.spec import AgentSpec, CallerPolicy
from app.platform.runtime.tool_loop import run_tool_loop
from app.platform.tools.executor import execute_tool_calls
from app.platform.tools.models import ToolCall, ToolContext, ToolResult
from app.platform.tools.registry import ToolRegistry, UnknownToolError
from app.tools.catalog import tool_registry
from app.utils.skill_utils import load_skill


logger = logging.getLogger(__name__)

ToolCallRunner = Callable[[list[ToolCall], ToolContext], Awaitable[list[ToolResult]]]
StopReason = Literal[
    "timeout",
    "budget_exhausted",
    "max_tool_calls",
    "no_progress",
    "invalid_output",
    "error",
    "cancelled",
]
BASE_SPEC = "agent_base"
BLOCKED_OUTPUT = {"error": "read-only policy blocked this call"}
# ARCHIE-326: "Do not take action" made agents drop the proposals the task asked for
FINAL_TASK = (
    "Report the result against the completion criteria. "
    "Put each change the task needs in proposed_actions; do not claim it was done."
)
SKILL_CHARS = 4000
# Defaults for a spec whose caller left a limit open
DEFAULT_BUDGET = Budget(
    max_tokens=40000, max_cost_usd=0.10, max_seconds=120, max_tool_rounds=3
)


class LoopGuardError(RuntimeError):
    """A run repeats the same tool call without progress."""

    def __init__(self, reason: Literal["no_progress"], detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


class AgentInstance(BaseModel):
    """A validated spec with a resolved model and its prompt, ready to run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: AgentSpec = Field(description="Validated spec")
    selection: ModelSelection = Field(description="Model of the first attempt")
    ceiling: ModelProfile = Field(description="Highest profile for escalation")
    schema_: ResultSchema = Field(alias="schema", description="Reply schemas")
    messages: list[dict[str, str]] = Field(description="Initial messages")
    read_tools: frozenset[str] = Field(description="Spec tools that only read")
    prompt_version: str = Field(description="Base prompt fingerprint + spec hash")
    final_task: str = Field(FINAL_TASK, description="Ask for the final reply")
    evidence_bytes: int = Field(8000, description="Evidence shown to the model")
    spend_kind: str = Field("agent", description="Daily spend bucket")
    new_evidence_only: bool = Field(
        True, description="A round shows only its own results (presets: all)"
    )


class RunUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    tool_rounds: int = 0
    tool_calls: int = 0
    llm_calls: int = 0


class AgentRunResult(BaseModel):
    run_id: str = Field(description="Run identity")
    parent_run_id: str = Field(description="Parent identity")
    state: RunState = Field(description="Terminal state")
    stop_reason: StopReason | None = Field(None, description="Why it stopped early")
    detail: str = Field("", description="Stop detail")
    error_type: str = Field("", description="Exception class of an error stop")
    reply: dict | None = Field(None, description="Last valid reply")
    evidence: list[ToolResult] = Field(default_factory=list)
    proposed_actions: list[ProposedAction] = Field(default_factory=list)
    criteria: list[CriterionCheck] = Field(default_factory=list)
    usage: RunUsage = Field(default_factory=RunUsage)
    selection: ModelSelection = Field(description="Model of the last attempt")
    escalations: list[Escalation] = Field(default_factory=list)
    retries: int = 0
    policy_violations: list[str] = Field(default_factory=list)
    loop_guards: list[str] = Field(default_factory=list)


class RunCancelledError(asyncio.CancelledError):
    """Cancellation that keeps the collected evidence."""

    def __init__(self, result: AgentRunResult) -> None:
        super().__init__("Agent run cancelled")
        self.result = result


def complete_budget(budget: Budget, defaults: Budget = DEFAULT_BUDGET) -> Budget:
    return Budget(
        **{
            name: value if value is not None else getattr(defaults, name)
            for name, value in budget.model_dump().items()
        }
    )


class AgentFactory:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        compiler: PromptCompiler | None = None,
    ) -> None:
        self.registry = registry or tool_registry
        self.compiler = compiler or PromptCompiler()

    def create(
        self,
        spec: AgentSpec,
        ceiling: ModelProfile | None = None,
        selection: ModelSelection | None = None,
        messages: list[dict[str, str]] | None = None,
        base_specs: tuple[str, ...] = (BASE_SPEC,),
        read_tools: frozenset[str] | None = None,
        timezone: str = "UTC",
        **options,
    ) -> AgentInstance:
        """Trusted presets may pass their own messages, base specs or model."""
        selection = selection or select_model(
            spec.model.profile, spec.model.effort, spec.model.reason
        )
        base = "+".join(self.compiler.fingerprint(name)[:12] for name in base_specs)
        if read_tools is None:
            read_tools = frozenset(
                name for name in spec.tools if self._effect(name) == "read"
            )
        return AgentInstance(
            spec=spec,
            selection=selection,
            ceiling=ceiling or spec.model.profile,
            schema=result_schema(spec.output.schema_name),
            messages=messages
            or self.messages(spec, selection.model, base_specs, timezone),
            read_tools=read_tools,
            prompt_version=f"{base}:{spec.fingerprint()}",
            **options,
        )

    def _effect(self, name: str) -> str:
        try:
            return self.registry.capability(name).effect
        except UnknownToolError:
            return "unknown"

    def messages(
        self,
        spec: AgentSpec,
        model: str,
        base_specs: tuple[str, ...],
        timezone: str = "UTC",
    ) -> list[dict[str, str]]:
        """Fixed platform rules; the brief, task and facts go in as data."""
        system = "\n".join(self.compiler.compile(name) for name in base_specs)
        if spec.tools:
            system += "\nAllowed tools:\n" + json.dumps(
                self.registry.schemas(model, spec.tools), ensure_ascii=False
            )
        for name in spec.skills:
            skill = load_skill(name)
            if skill:
                system += (
                    f"\nSkill {name} (reference data):\n"
                    + str(skill.get("content", ""))[:SKILL_CHARS]
                )
        data = {
            "role": spec.identity.role,
            "purpose": spec.identity.purpose,
            "task": spec.objective.task,
            "completion_criteria": spec.objective.completion_criteria,
            "brief": spec.instructions,
            "facts": spec.context.facts,
            # ARCHIE-326: without the date "tomorrow" or "next week" cannot be resolved
            "now": datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M %A"),
            "timezone": timezone,
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
        ]


class AgentRuntime:
    """Executes any AgentInstance: one class for every configuration."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        call: CompletionCall | None = None,
        run_tools: ToolCallRunner | None = None,
    ) -> None:
        self.registry = registry or tool_registry
        self.call = call or complete
        self.run_tools = run_tools or self._run_tools

    async def _run_tools(
        self, calls: list[ToolCall], context: ToolContext
    ) -> list[ToolResult]:
        return await execute_tool_calls(
            calls, self.registry, allowed={c.tool_name for c in calls}, context=context
        )

    async def execute(
        self,
        agent: AgentInstance,
        caller: CallerPolicy,
        run_id: str | None = None,
        recorder: RunRecorder | None = None,
    ) -> AgentRunResult:
        run = _Run(self, agent, caller, run_id or uuid.uuid4().hex[:16], recorder)
        return await run.execute()


class _Run:
    """State of one execution; parallel runs share nothing."""

    def __init__(
        self,
        runtime: AgentRuntime,
        agent: AgentInstance,
        caller: CallerPolicy,
        run_id: str,
        recorder: RunRecorder | None,
    ) -> None:
        self.runtime = runtime
        self.agent = agent
        self.spec = agent.spec
        self.caller = caller
        self.run_id = run_id
        self.recorder = recorder
        budget = complete_budget(self.spec.budget)
        rounds = min(budget.max_tool_rounds, self.spec.execution.max_iterations)
        self.meter = BudgetMeter(budget.model_copy(update={"max_tool_rounds": rounds}))
        self.selection = agent.selection
        self.evidence: list[ToolResult] = []
        self.proposed: list[ProposedAction] = []
        self.traces: list[LllmTrace] = []
        self.reply: BaseModel | None = None
        self.seen: set[str] = set()
        self.tool_calls = 0
        self.llm_calls = 0
        self.retries = 0
        self.violations: list[str] = []
        self.guards: list[str] = []
        self.escalations: list[Escalation] = []
        self.messages: list[dict[str, str]] = []

    def event(self, kind, **detail) -> None:
        if self.recorder:
            self.recorder.event(kind, **detail)

    async def call(self, schema: type[BaseModel]) -> BaseModel:
        model = self.selection.model
        attempts = self.spec.execution.retries + 1
        for attempt in range(attempts):
            cap = output_cap(
                self.meter,
                self.messages,
                schema,
                model,
                max_output_for(self.selection.effort),
            )
            try:
                effort = self.selection.effort
                completion: Completion = await self.runtime.call(
                    self.messages,
                    schema,
                    model,
                    cap,
                    **({"effort": effort} if effort else {}),
                )
            except InvalidOutputError as exc:
                self._count(exc.trace)
                if attempt + 1 >= attempts:
                    raise
                self.retries += 1
                logger.warning(
                    f"agent_runtime_warning_002: {self.run_id} invalid reply, retry {self.retries}: {exc}"
                )
                self.messages.append(
                    {
                        "role": "user",
                        "content": f"Your last reply did not match the schema ({exc}). Reply again with valid JSON.",
                    }
                )
                continue
            if not isinstance(completion.content, self.agent.schema_.reply):
                raise InvalidOutputError(
                    "The completion did not return the reply schema"
                )
            self.reply = completion.content
            self._count(completion.trace)
            return completion.content
        raise InvalidOutputError("No reply")

    def _count(self, trace: LllmTrace | None) -> None:
        self.llm_calls += 1
        if trace is None:
            return
        self.traces.append(trace)
        daily_spend.add(self.agent.spend_kind, trace.total_cost or 0.0)
        self.meter.add(trace)

    def _allowed_read(self, call: ToolCall, arguments: dict[str, str]) -> bool:
        if call.tool_name not in self.spec.tools:
            return False
        return (
            call.tool_name in self.agent.read_tools
            or self.runtime.registry.is_read_only(call.tool_name, arguments)
        )

    def _block(self, call: ToolCall, reason: str) -> None:
        self.violations.append(f"{call.tool_name}: {reason}")
        self.event("policy_violation", tool=call.tool_name, reason=reason)
        self.evidence.append(
            ToolResult(
                tool_name=call.tool_name,
                output=BLOCKED_OUTPUT,
                success=False,
                error="read-only policy",
            )
        )

    def _propose(self, call: ToolCall, reason: str = "") -> bool:
        permissions = self.spec.permissions
        if (
            permissions.mode != "propose_actions"
            or call.tool_name not in permissions.approval_tools
        ):
            return False
        self.proposed.append(
            ProposedAction(
                tool_name=call.tool_name,
                arguments=call.arguments,
                reason=reason or call.reason,
            )
        )
        self.event("approval_requested", tool=call.tool_name)
        return True

    async def tools(self, turn: BaseModel) -> tuple[str, bool]:
        """Policy for each call right before it runs; loop guards after it."""
        check_remaining(self.meter)
        shown = len(self.evidence) if self.agent.new_evidence_only else 0
        permitted: list[ToolCall] = []
        progress = False
        skipped = 0
        limit = self.spec.execution.max_tool_calls
        for call in getattr(turn, "tool_calls", []):
            arguments = {p.name: p.value for p in call.arguments}
            if self._allowed_read(call, arguments):
                key = json.dumps([call.tool_name, sorted(arguments.items())])
                if key in self.seen:
                    self.guards.append(f"repeat {call.tool_name}")
                    continue
                if self.tool_calls >= limit:
                    skipped += 1
                    continue
                self.seen.add(key)
                self.tool_calls += 1
                permitted.append(call)
            elif self._propose(call):
                progress = True
                self.evidence.append(
                    ToolResult(
                        tool_name=call.tool_name,
                        output={"status": "waiting_for_approval"},
                    )
                )
            else:
                self._block(call, "not allowed for this agent")
        if skipped:
            # The limit ends tool use, not the run: the agent reports what it has
            self.guards.append("max_tool_calls")
            self.event("loop_guard", reason="max_tool_calls", skipped=skipped)
        if permitted:
            progress = True
            for call in permitted:
                self.event("tool_invoked", tool=call.tool_name)
            self.evidence.extend(
                await self.runtime.run_tools(
                    permitted,
                    ToolContext(
                        user_timezone=self.caller.user_timezone,
                        demo_mode=self.caller.demo_mode,
                    ),
                )
            )
        if (
            not progress
            and not skipped
            and any(g.startswith("repeat") for g in self.guards)
        ):
            self.event("loop_guard", reason="no_progress")
            raise LoopGuardError("no_progress", "the same tool call was repeated")
        self.meter.check()
        text = self._visible_evidence(self.evidence[shown:])
        if self.tool_calls >= limit:
            text += (
                "\n[Tool call limit reached. Give the final reply with this evidence.]"
            )
            return text, True
        return text, False

    def _visible_evidence(self, evidence: list[ToolResult] | None = None) -> str:
        """Earlier rounds are already in the messages; a round shows its results."""
        items = self.evidence if evidence is None else evidence
        visible = json.dumps(
            [result.model_dump() for result in items], ensure_ascii=False
        ).encode("utf-8")
        if len(visible) <= self.agent.evidence_bytes:
            return visible.decode("utf-8")
        return visible[: self.agent.evidence_bytes].decode("utf-8", errors="ignore") + (
            "\n[Evidence clipped to the run budget. Full read results remain in the record. "
            "Do not infer facts from the omitted evidence.]"
        )

    def _criteria_failed(self) -> bool:
        checks = getattr(self.reply, "criteria", None)
        if checks is None:
            return False
        wanted = len(self.spec.objective.completion_criteria)
        return len(checks) < wanted or not all(check.met for check in checks)

    async def _attempt(self) -> None:
        schema = self.agent.schema_
        timeout = (
            min(self.spec.execution.timeout_seconds, self.meter.budget.max_seconds)
            - self.meter.seconds
        )
        if timeout <= 0:
            raise TimeoutError("wall time")
        async with asyncio.timeout(timeout):
            await run_tool_loop(
                self.call,
                lambda turn: getattr(turn, "tool_calls", []),
                self.tools,
                self.messages,
                schema.turn if self.meter.budget.max_tool_rounds else schema.reply,
                schema.reply,
                self.agent.final_task,
                self.meter,
            )

    def _escalate(self, reason: EscalationReason, detail: str) -> bool:
        """One step up for a checked problem, once per run, within the ceiling."""
        if self.escalations:
            return False
        step = next_profile(self.selection.profile, reason, self.agent.ceiling, detail)
        if step is None:
            return False
        try:
            self.selection = select_model(
                step.to_profile, self.spec.model.effort, f"escalation: {reason}"
            )
        except LookupError:
            return False
        self.escalations.append(step)
        self.event(
            "model_escalated",
            reason=reason,
            to_profile=step.to_profile,
            model=self.selection.model,
        )
        self.seen.clear()
        return True

    def _restart_messages(self) -> None:
        self.messages = [dict(m) for m in self.agent.messages]
        if self.evidence:
            self.messages.append(
                {
                    "role": "system",
                    "content": "Evidence from an earlier attempt:\n"
                    + self._visible_evidence(),
                }
            )

    async def execute(self) -> AgentRunResult:
        state: RunState = "completed"
        stop: StopReason | None = None
        detail = ""
        error_type = ""
        cancelled = False
        self.messages = [dict(m) for m in self.agent.messages]
        self.event(
            "agent_started",
            model=self.selection.model,
            effort=self.selection.effort,
            profile=self.selection.profile,
        )
        while True:
            try:
                await self._attempt()
                if self._criteria_failed() and self._escalate(
                    "criteria_failed", "the reply marks criteria as not met"
                ):
                    self._restart_messages()
                    continue
                break
            except InvalidOutputError as exc:
                if self._escalate("invalid_output", str(exc)):
                    self._restart_messages()
                    continue
                state, stop, detail = "failed", "invalid_output", str(exc)
            except LoopGuardError as exc:
                if exc.reason == "no_progress" and self._escalate(
                    "no_progress", str(exc)
                ):
                    self._restart_messages()
                    continue
                state, stop, detail = "failed", exc.reason, str(exc)
            except asyncio.CancelledError:
                state, stop, cancelled = "cancelled", "cancelled", True
                detail = "Stopped because the parent run was cancelled."
                logger.info(f"agent_runtime_002: {self.run_id} {detail}")
            except (TimeoutError, BudgetExceededError, BudgetEstimateError) as exc:
                state = "failed"
                stop = (
                    "timeout" if isinstance(exc, TimeoutError) else "budget_exhausted"
                )
                detail = str(exc) or "wall time"
                self.event("budget_exhausted", limit=detail)
                logger.warning(f"agent_runtime_warning_001: {self.run_id} {detail}")
            except Exception as exc:
                state, stop = "failed", "error"
                error_type = type(exc).__name__
                detail = f"{error_type}: {exc}"
                logger.exception(f"agent_runtime_error_001: {self.run_id} {detail}")
            break
        if state == "completed":
            self._collect_reply_actions()
            if self.proposed:
                state = "waiting_for_approval"
        result = self._result(state, stop, detail, error_type)
        if cancelled:
            raise RunCancelledError(result)
        return result

    def _collect_reply_actions(self) -> None:
        """Actions in the final reply count only if the spec may propose them."""
        for action in getattr(self.reply, "proposed_actions", []) or []:
            call = ToolCall.model_construct(
                tool_name=action.tool_name,
                arguments=action.arguments,
                reason=action.reason,
            )
            if any(
                p.tool_name == action.tool_name and p.arguments == action.arguments
                for p in self.proposed
            ):
                continue
            if not self._propose(call, action.reason):
                self.violations.append(f"{action.tool_name}: proposal not permitted")
                self.event("policy_violation", tool=action.tool_name, reason="proposal")

    def _result(
        self, state: RunState, stop: StopReason | None, detail: str, error_type: str
    ) -> AgentRunResult:
        return AgentRunResult(
            run_id=self.run_id,
            parent_run_id=self.caller.parent_run_id,
            state=state,
            stop_reason=stop,
            detail=detail,
            error_type=error_type,
            reply=self.reply.model_dump(mode="json") if self.reply else None,
            evidence=self.evidence,
            proposed_actions=self.proposed,
            criteria=list(getattr(self.reply, "criteria", []) or []),
            usage=RunUsage(
                input_tokens=sum(t.input_tokens for t in self.traces),
                output_tokens=sum(t.output_tokens for t in self.traces),
                cost_usd=self.meter.cost,
                seconds=round(self.meter.seconds, 3),
                tool_rounds=self.meter.rounds,
                tool_calls=self.tool_calls,
                llm_calls=self.llm_calls,
            ),
            selection=self.selection,
            escalations=self.escalations,
            retries=self.retries,
            policy_violations=self.violations,
            loop_guards=self.guards,
        )


def model_override(model: str, profile: ModelProfile = "BALANCED") -> ModelSelection:
    """A trusted preset that names its model (the swarm's HA diagnosis)."""
    return ModelSelection(profile=profile, model=model, reason=f"preset model {model}")
