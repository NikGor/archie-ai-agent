"""AgentSpec: an agent as configuration, and its deterministic validator (ARCHIE-320).

Contract: Task + AgentSpec + CallerPolicy -> validated runtime -> Run + AgentResult.
A spec from an LLM is untrusted input: `validate_spec` rejects what the caller
may not grant and caps budgets and limits to the caller's ceilings.
"""

import hashlib
import logging
from collections.abc import Iterable
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from app.platform.llm.profiles import PROFILES, LogicalEffort, ModelProfile
from app.platform.runtime.budget import Budget
from app.platform.runtime.results import RESULT_SCHEMAS
from app.platform.runtime.spawn import SpawnContext
from app.platform.tools.registry import ToolRegistry, UnknownToolError


logger = logging.getLogger(__name__)

EFFORTS: tuple[LogicalEffort, ...] = ("low", "medium", "high")

# Tools that create agents or hand work to other agents: never inside a spec (no recursion)
DELEGATION_TOOLS = frozenset(
    {"delegate_agent_tool", "agent_task_tool", "spawn_agent_tool"}
)


class SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Identity(SpecModel):
    name: str = Field(
        min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$", description="Slug"
    )
    role: str = Field(min_length=1, max_length=200, description="Who the agent is")
    purpose: str = Field(min_length=1, max_length=500, description="Why it exists")


class Objective(SpecModel):
    task: str = Field(min_length=1, max_length=2000, description="Task to finish")
    completion_criteria: list[str] = Field(
        min_length=1, max_length=8, description="Checks that prove the task is done"
    )


class ModelChoice(SpecModel):
    profile: ModelProfile = Field(description="Work complexity, not a model name")
    effort: LogicalEffort | None = Field(None, description="Logical reasoning effort")
    reason: str = Field("", max_length=300, description="Why this profile")


class ContextSpec(SpecModel):
    facts: list[str] = Field(
        default_factory=list, max_length=20, description="Input facts; never secrets"
    )
    memory: Literal["none", "user_state"] = Field(
        "none", description="Memory access; none keeps the context minimal"
    )


class Execution(SpecModel):
    max_iterations: int = Field(3, ge=1, le=8, description="LLM tool rounds")
    max_tool_calls: int = Field(6, ge=0, le=24, description="Tool calls in the run")
    timeout_seconds: float = Field(90, gt=0, le=600, description="Wall time")
    retries: int = Field(1, ge=0, le=3, description="Repeats of an invalid reply")


class Permissions(SpecModel):
    mode: Literal["read_only", "propose_actions"] = Field(
        "read_only", description="propose_actions: side effects wait for approval"
    )
    approval_tools: list[str] = Field(
        default_factory=list, description="Write tools the agent may propose"
    )


class OutputSpec(SpecModel):
    schema_name: str = Field(
        pattern=r"^[a-z][a-z0-9_]*_v[0-9]+$", description="Registered result schema"
    )


class AgentSpec(SpecModel):
    spec_version: Literal["1"] = Field("1", description="Version of this format")
    identity: Identity
    objective: Objective
    instructions: str = Field(
        "", max_length=4000, description="Brief for this task; data, not policy"
    )
    model: ModelChoice
    tools: list[str] = Field(default_factory=list, max_length=12)
    skills: list[str] = Field(default_factory=list, max_length=4)
    context: ContextSpec = Field(default_factory=ContextSpec)
    execution: Execution = Field(default_factory=Execution)
    budget: Budget = Field(default_factory=Budget)
    permissions: Permissions = Field(default_factory=Permissions)
    output: OutputSpec
    lifecycle: Literal["ephemeral"] = Field("ephemeral", description="Run lifetime")

    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


class CallerPolicy(BaseModel):
    """What the trusted caller may grant; built from SpawnContext, never by an LLM."""

    source: str = Field(description="Trusted caller identity")
    parent_run_id: str = Field(description="Run that owns the request")
    depth: int = Field(0, ge=0, description="Spawn depth of the caller")
    max_depth: int = Field(1, ge=1, description="Deepest allowed spawn")
    tools: frozenset[str] = Field(description="Tools the caller may grant")
    max_profile: ModelProfile = Field("DEEP", description="Most expensive profile")
    max_effort: LogicalEffort = Field("high", description="Highest logical effort")
    budget: Budget = Field(description="Budget ceiling; every limit is set")
    limits: Execution = Field(default_factory=Execution, description="Limit ceiling")
    may_request_approval: bool = Field(
        False, description="The caller can ask its user to approve proposed actions"
    )
    user_timezone: str = Field("UTC", description="Timezone passed to tools")
    demo_mode: bool = Field(False, description="Tools return demo data")

    @classmethod
    def from_spawn(
        cls, caller: SpawnContext, tools: Iterable[str], **ceilings
    ) -> "CallerPolicy":
        return cls(
            source=caller.source,
            parent_run_id=caller.parent_run_id,
            depth=caller.depth,
            tools=frozenset(tools) - DELEGATION_TOOLS,
            user_timezone=caller.user_timezone,
            demo_mode=caller.demo_mode,
            **ceilings,
        )


class SpecValidationError(ValueError):
    """The spec is invalid or asks for more than the caller may grant."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class ValidatedSpec(BaseModel):
    spec: AgentSpec = Field(description="Spec after caps")
    adjustments: list[str] = Field(
        default_factory=list, description="Caps the validator applied"
    )


def _cap(value: float | None, ceiling: float | None) -> float | None:
    if ceiling is None:
        return value
    return ceiling if value is None else min(value, ceiling)


def _tool_errors(
    spec: AgentSpec, policy: CallerPolicy, registry: ToolRegistry
) -> list[str]:
    errors: list[str] = []
    write_tools = []
    for name in spec.tools:
        try:
            capability = registry.capability(name)
        except UnknownToolError:
            errors.append(f"unknown tool {name}")
            continue
        if name in DELEGATION_TOOLS:
            errors.append(f"{name} creates agents; delegation is not recursive")
        elif name not in policy.tools:
            errors.append(f"{name} is not allowed for caller {policy.source}")
        elif capability.effect == "write" and not capability.read_actions:
            write_tools.append(name)
    errors += [
        f"approval tool {name} is not in tools"
        for name in spec.permissions.approval_tools
        if name not in spec.tools
    ]
    if not write_tools and spec.permissions.mode == "read_only":
        return errors
    if not policy.may_request_approval:
        errors.append(
            "tools with side effects need approval the caller cannot ask: "
            + (", ".join(write_tools) or "propose_actions")
        )
    elif set(write_tools) - set(spec.permissions.approval_tools):
        errors.append("write tools must be listed in permissions.approval_tools")
    return errors


def _capped(spec: AgentSpec, policy: CallerPolicy) -> ValidatedSpec:
    budget = {
        name: _cap(getattr(spec.budget, name), getattr(policy.budget, name))
        for name in Budget.model_fields
    }
    limits = {
        name: min(getattr(spec.execution, name), getattr(policy.limits, name))
        for name in Execution.model_fields
    }
    profile = spec.model.profile
    if PROFILES.index(profile) > PROFILES.index(policy.max_profile):
        profile = policy.max_profile
    effort = spec.model.effort
    if effort and EFFORTS.index(effort) > EFFORTS.index(policy.max_effort):
        effort = policy.max_effort
    adjustments = [
        f"{label} capped: {before} -> {after}"
        for label, before, after in (
            ("budget", spec.budget.model_dump(), budget),
            ("execution", spec.execution.model_dump(), limits),
            ("profile", spec.model.profile, profile),
            ("effort", spec.model.effort, effort),
        )
        if before != after
    ]
    capped = spec.model_copy(
        update={
            "budget": Budget.model_validate(budget),
            "execution": Execution.model_validate(limits),
            "model": spec.model.model_copy(
                update={"profile": profile, "effort": effort}
            ),
        }
    )
    if adjustments:
        logger.info(f"agent_spec_001: {spec.identity.name}: {'; '.join(adjustments)}")
    return ValidatedSpec(spec=capped, adjustments=adjustments)


def validate_spec(
    draft: AgentSpec | dict,
    policy: CallerPolicy,
    registry: ToolRegistry,
    skills: Iterable[str] = (),
) -> ValidatedSpec:
    """Reject unknown or ungranted capabilities; cap budget, limits and profile."""
    try:
        spec = (
            draft if isinstance(draft, AgentSpec) else AgentSpec.model_validate(draft)
        )
    except ValidationError as exc:
        raise SpecValidationError(
            [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        ) from exc
    errors = ["maximum spawn depth reached"] if policy.depth >= policy.max_depth else []
    errors += _tool_errors(spec, policy, registry)
    known_skills = set(skills)
    errors += [f"unknown skill {s}" for s in spec.skills if s not in known_skills]
    if spec.output.schema_name not in RESULT_SCHEMAS:
        errors.append(f"unknown output schema {spec.output.schema_name}")
    if errors:
        logger.warning(
            f"agent_spec_warning_001: Rejected {spec.identity.name}: {errors}"
        )
        raise SpecValidationError(errors)
    return _capped(spec, policy)
