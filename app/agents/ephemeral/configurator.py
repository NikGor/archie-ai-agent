"""AgentConfigurator (ARCHIE-323): task -> configuration proposal -> validated
AgentSpec. The LLM proposes; `validate_spec` decides. The configurator never
grants tools, budget, depth or approval rights that the caller does not have.

First decision: is an agent needed at all (spawn | direct)?
"""

import json
import logging
from typing import Literal
from pydantic import BaseModel, Field
from app.config import settings
from app.platform.llm.profiles import (
    PROFILES,
    ModelProfile,
    select_model,
    translate_effort,
)
from app.platform.prompts.compiler import PromptCompiler
from app.platform.runtime.budget import Budget, BudgetExceededError, BudgetMeter
from app.platform.runtime.completion import (
    BudgetEstimateError,
    CompletionCall,
    InvalidOutputError,
    complete,
    max_output_for,
    output_cap,
)
from app.platform.runtime.results import RESULT_SCHEMAS
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

PROMPT = "configurator"
# One configurator call: small, fixed budget. The byte-based estimate is about
# 5x the real price (gpt-4.1: ~$0.035 estimated, ~$0.006 paid), ARCHIE-323 eval.
CONFIGURATOR_BUDGET = Budget(max_tokens=40000, max_cost_usd=0.10, max_seconds=60)


class SpecDraft(BaseModel):
    """What the LLM fills; the runtime turns it into an AgentSpec and checks it."""

    name: str = Field(description="Agent slug: lowercase letters, digits, _")
    role: str = Field(description="Who the agent is, one line")
    purpose: str = Field(description="Why the agent exists, one line")
    completion_criteria: list[str] = Field(
        description="1-4 checks that prove the task is done"
    )
    instructions: str = Field(description="Short brief: steps, checks, report")
    profile: ModelProfile = Field(description="Work complexity of the task")
    effort: Literal["low", "medium", "high"] = Field(description="Reasoning effort")
    profile_reason: str = Field(description="Why this profile and effort")
    tools: list[str] = Field(description="Smallest set of listed tools")
    skills: list[str] = Field(description="Listed skills the agent needs")
    facts: list[str] = Field(description="Facts from the context the task needs")
    output_schema: str = Field(description="One of the listed output schemas")
    max_iterations: int = Field(description="Tool rounds")
    max_tool_calls: int = Field(description="Tool calls in the run")
    max_cost_usd: float = Field(description="USD for the run")
    permissions_mode: Literal["read_only", "propose_actions"] = Field(
        description="propose_actions only when the task needs a change"
    )
    approval_tools: list[str] = Field(
        description="Write tools the agent may propose; empty for read_only"
    )


class ConfigurationDraft(BaseModel):
    decision: Literal["spawn", "direct"] = Field(
        description="spawn: create an agent; direct: Archie does it with one tool call or an answer"
    )
    reason: str = Field(description="Why this decision")
    spec: SpecDraft | None = Field(description="Agent configuration; null for direct")


class ConfigurationResult(BaseModel):
    decision: Literal["spawn", "direct", "failed"] = Field(description="Outcome")
    reason: str = Field(description="Decision reason or the failure")
    spec: AgentSpec | None = Field(None, description="Validated spec for spawn")
    adjustments: list[str] = Field(default_factory=list, description="Validator caps")
    errors: list[str] = Field(default_factory=list, description="Validation errors")
    attempts: int = Field(0, description="LLM calls")
    model: str = Field("", description="Configurator model")
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    prompt_version: str = Field("", description="Fingerprint of the configurator spec")


def to_spec(
    draft: SpecDraft, task: str, timeout_seconds: float, retries: int = 1
) -> dict:
    """The original task stays the objective; the draft cannot rewrite it."""
    return {
        "identity": {"name": draft.name, "role": draft.role, "purpose": draft.purpose},
        "objective": {"task": task, "completion_criteria": draft.completion_criteria},
        "instructions": draft.instructions,
        "model": {
            "profile": draft.profile,
            "effort": draft.effort,
            "reason": draft.profile_reason[:300],
        },
        "tools": draft.tools,
        "skills": draft.skills,
        "context": {"facts": draft.facts},
        "execution": {
            "max_iterations": max(1, min(8, draft.max_iterations)),
            "max_tool_calls": max(0, min(24, draft.max_tool_calls)),
            "timeout_seconds": timeout_seconds,
            "retries": retries,
        },
        "budget": {"max_cost_usd": max(0.0, draft.max_cost_usd)},
        "permissions": {
            "mode": draft.permissions_mode,
            "approval_tools": draft.approval_tools,
        },
        "output": {"schema_name": draft.output_schema},
    }


class AgentConfigurator:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        call: CompletionCall | None = None,
        compiler: PromptCompiler | None = None,
    ) -> None:
        self.registry = registry or tool_registry
        self.call = call or complete
        self.compiler = compiler or PromptCompiler()

    def capabilities(self, policy: CallerPolicy) -> dict:
        """Only what the caller may grant: the LLM never sees other tools."""
        tools = []
        for name in self.registry.names():
            if name not in policy.tools:
                continue
            capability = self.registry.capability(name)
            doc = (self.registry.function(name).__doc__ or "").strip().splitlines()
            tools.append(
                {
                    "name": name,
                    "effect": capability.effect,
                    "read_actions": list(capability.read_actions),
                    "description": doc[0] if doc else "",
                }
            )
        ceiling = PROFILES.index(policy.max_profile)
        return {
            "tools": tools,
            "skills": [
                {"name": s["name"], "description": s.get("description", "")}
                for s in list_skills()
            ],
            "output_schemas": [
                {"name": s.name, "description": s.description}
                for s in RESULT_SCHEMAS.values()
                if s.configurable
            ],
            "profiles": list(PROFILES[: ceiling + 1]),
            "ceilings": {
                "max_cost_usd": policy.budget.max_cost_usd,
                "max_iterations": policy.limits.max_iterations,
                "max_tool_calls": policy.limits.max_tool_calls,
            },
            "may_request_approval": policy.may_request_approval,
        }

    def messages(
        self, task: str, context: str, policy: CallerPolicy
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": self.compiler.compile(PROMPT)
                + "\nCapabilities:\n"
                + json.dumps(self.capabilities(policy), ensure_ascii=False),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"task": task, "context": context}, ensure_ascii=False
                ),
            },
        ]

    async def configure(
        self,
        task: str,
        context: str,
        policy: CallerPolicy,
        skills: list[str],
        model: str | None = None,
    ) -> ConfigurationResult:
        """One structured call; an invalid draft gets one repeat with the errors.
        `model` overrides the profile model (eval compares models)."""
        selection = select_model(settings.agent_configurator_profile, "low")
        if model:
            selection = selection.model_copy(
                update={"model": model, "effort": translate_effort(model, "low")}
            )
        meter = BudgetMeter(CONFIGURATOR_BUDGET)
        messages = self.messages(task, context, policy)
        result = ConfigurationResult(
            decision="failed",
            reason="",
            model=selection.model,
            prompt_version=self.compiler.fingerprint(PROMPT),
        )
        for attempt in range(2):
            result.attempts = attempt + 1
            try:
                cap = output_cap(
                    meter,
                    messages,
                    ConfigurationDraft,
                    selection.model,
                    max_output_for(selection.effort),
                )
                completion = await self.call(
                    messages,
                    ConfigurationDraft,
                    selection.model,
                    cap,
                    **({"effort": selection.effort} if selection.effort else {}),
                )
            except (BudgetExceededError, BudgetEstimateError) as exc:
                result.errors = [str(exc)]
                break
            except InvalidOutputError as exc:
                self._count(result, meter, exc.trace)
                result.errors = [str(exc)]
                messages.append(
                    {"role": "user", "content": f"Invalid JSON ({exc}). Reply again."}
                )
                continue
            self._count(result, meter, completion.trace)
            draft = completion.content
            assert isinstance(draft, ConfigurationDraft)
            result.reason = draft.reason
            if draft.decision == "direct":
                result.decision = "direct"
                logger.info(f"agent_configurator_001: direct: {draft.reason}")
                return result
            try:
                if draft.spec is None:
                    raise SpecValidationError(["spawn needs a spec"])
                validated = validate_spec(
                    to_spec(draft.spec, task, policy.limits.timeout_seconds),
                    policy,
                    self.registry,
                    skills,
                )
            except SpecValidationError as exc:
                result.errors = exc.errors
                logger.warning(
                    f"agent_configurator_warning_001: attempt {attempt + 1}: {exc}"
                )
                messages.append(
                    {"role": "assistant", "content": draft.model_dump_json()}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"Validation failed: {exc}. Fix the configuration.",
                    }
                )
                continue
            result.decision = "spawn"
            result.spec = validated.spec
            result.adjustments = validated.adjustments
            result.errors = []
            logger.info(
                f"agent_configurator_002: spawn {validated.spec.identity.name} "
                f"{validated.spec.model.profile} tools={validated.spec.tools}"
            )
            return result
        result.reason = result.reason or "no valid configuration"
        logger.warning(f"agent_configurator_warning_002: failed: {result.errors}")
        return result

    @staticmethod
    def _count(result: ConfigurationResult, meter: BudgetMeter, trace) -> None:
        if trace is None:
            return
        result.input_tokens += trace.input_tokens
        result.output_tokens += trace.output_tokens
        result.cost_usd += trace.total_cost or 0.0
        meter.add(trace)
