"""Composition of agent systems for the agent constructor (ARCHIE-341): the
constructor caller policy, the catalog of what can be built, validation,
composition, runs and schedules. The API process serves the constructor; the
runtime process runs scheduled systems through `agent_system_tool`."""

import logging
from typing import Literal
from pydantic import BaseModel, Field, JsonValue
from app.agents.ephemeral.composer import AgentSystemComposer, CompositionResult
from app.agents.ephemeral.dynamic import DynamicAgents, skill_names
from app.agents.ephemeral.system import SystemRuntime
from app.backend.cron_scheduler import cron_scheduler
from app.backend.redis_factory import get_async_redis
from app.config import settings
from app.platform.llm.catalog import MODEL_EFFORTS
from app.platform.llm.profiles import PROFILES
from app.platform.runtime.budget import Budget
from app.platform.runtime.completion import CompletionCall
from app.platform.runtime.results import RESULT_SCHEMAS
from app.platform.runtime.spawn import SpawnContext
from app.platform.runtime.spec import (
    DELEGATION_TOOLS,
    EFFORTS,
    CallerPolicy,
    Execution,
)
from app.platform.runtime.system import (
    AUTONOMY_POWERS,
    DIMENSIONS,
    MAX_DEBATE_ROUNDS,
    MAX_NODES,
    MAX_PEERS,
    MAX_WORKERS,
    TOPOLOGY_ROLES,
    AgentSystemSpec,
    SystemCheck,
    check_system,
    validate_system,
)
from app.platform.runtime.system_runner import register_system_runner
from app.platform.runtime.system_runs import RedisSystemRunStore, SystemRunRecord
from app.tools.catalog import tool_registry
from app.utils.skill_utils import list_skills


logger = logging.getLogger(__name__)

# Ceilings of one constructor run: every node and the whole system (ARCHIE-341)
CONSTRUCTOR_BUDGET = Budget(
    max_tokens=400000, max_cost_usd=0.5, max_seconds=600, max_tool_rounds=8
)
CONSTRUCTOR_LIMITS = Execution(
    max_iterations=6, max_tool_calls=16, timeout_seconds=300, retries=1
)


def constructor_tools() -> list[str]:
    """Tools of the chat: exclusive groups and delegation are never granted."""
    return [
        name
        for name in tool_registry.names(include_exclusive=False)
        if name not in DELEGATION_TOOLS
    ]


def constructor_policy(
    demo_mode: bool, source: str = "constructor", parent_run_id: str = "constructor"
) -> CallerPolicy:
    """The user of the constructor: every chat tool, side effects as proposals."""
    return CallerPolicy.from_spawn(
        SpawnContext(
            source=source,
            parent_run_id=parent_run_id,
            user_timezone=settings.default_timezone,
            demo_mode=demo_mode,
        ),
        constructor_tools(),
        budget=CONSTRUCTOR_BUDGET,
        limits=CONSTRUCTOR_LIMITS,
        max_profile="MAXIMUM",
        max_effort="high",
        may_request_approval=True,
    )


class ScheduleInfo(BaseModel):
    job_id: str = Field(description="Cron job of the schedule")
    next_run_at: str | None = Field(None, description="Next run, UTC ISO")
    enabled: bool = Field(True, description="The job runs")


class SystemService:
    def __init__(self, agents: DynamicAgents, call: CompletionCall) -> None:
        self.agents = agents
        self.runtime = SystemRuntime(agents, RedisSystemRunStore(get_async_redis()))
        self.composer = AgentSystemComposer(agents.configurator, call)

    def catalog(self) -> dict[str, JsonValue]:
        """Everything the constructor may offer; availability comes from the validator."""
        tools = []
        for name in constructor_tools():
            capability = tool_registry.capability(name)
            doc = (tool_registry.function(name).__doc__ or "").strip().splitlines()
            tools.append(
                {
                    "name": name,
                    "effect": capability.effect,
                    "read_actions": list(capability.read_actions),
                    "description": doc[0] if doc else "",
                }
            )
        return {
            "dimensions": [d.model_dump() for d in DIMENSIONS.values()],
            "autonomy": [p.model_dump() for p in AUTONOMY_POWERS],
            "topology_roles": {k: sorted(v) for k, v in TOPOLOGY_ROLES.items()},
            "tools": tools,
            "skills": [
                {"name": s["name"], "description": s.get("description", "")}
                for s in list_skills()
            ],
            "profiles": [
                {
                    "name": profile,
                    "models": settings.agent_profiles.get(profile, []),
                }
                for profile in PROFILES
            ],
            "efforts": list(EFFORTS),
            "model_efforts": {k: list(v) for k, v in MODEL_EFFORTS.items()},
            "output_schemas": [
                {"name": s.name, "description": s.description}
                for s in RESULT_SCHEMAS.values()
                if s.configurable
            ],
            "limits": {
                "max_nodes": MAX_NODES,
                "max_workers": MAX_WORKERS,
                "max_peers": MAX_PEERS,
                "max_debate_rounds": MAX_DEBATE_ROUNDS,
                "system_max_cost_usd": CONSTRUCTOR_BUDGET.max_cost_usd,
                "system_max_seconds": CONSTRUCTOR_BUDGET.max_seconds,
                "node": CONSTRUCTOR_LIMITS.model_dump(),
                "daily_cost_usd": settings.agent_daily_cost_usd,
            },
            "timezone": settings.default_timezone,
        }

    def check(self, spec: dict) -> SystemCheck:
        return check_system(
            spec, constructor_policy(False), tool_registry, skill_names()
        )

    async def compose(
        self,
        task: str,
        topology: Literal["centralized", "workflow", "hierarchical", "decentralized"]
        | None,
    ) -> CompositionResult:
        return await self.composer.compose(
            task, constructor_policy(False), skill_names(), topology
        )

    def start(
        self, spec: dict, task: str, demo_mode: bool, trigger: str = "manual"
    ) -> SystemRunRecord:
        """Validate and start; raises SystemValidationError with every error."""
        policy = constructor_policy(demo_mode)
        validated = validate_system(spec, policy, tool_registry, skill_names())
        return self.runtime.start(
            validated, self.task_of(validated, task), policy, trigger
        )

    @staticmethod
    def task_of(spec: AgentSystemSpec, task: str) -> str:
        if spec.dynamism == "static" or not task.strip():
            task = spec.task
        if not task.strip():
            raise ValueError("The run needs a task")
        return task

    async def run_scheduled(
        self, spec: dict, task: str, demo_mode: bool
    ) -> dict[str, JsonValue]:
        policy = constructor_policy(demo_mode, source="constructor_schedule")
        validated = validate_system(spec, policy, tool_registry, skill_names())
        record = await self.runtime.run(
            validated, self.task_of(validated, task), policy, "schedule"
        )
        return record.summary()

    async def schedule(self, spec: dict, demo_mode: bool) -> ScheduleInfo:
        """One cron job per saved system; the job carries the spec snapshot."""
        validated = validate_system(
            spec, constructor_policy(demo_mode), tool_registry, skill_names()
        )
        if validated.activation != "scheduled" or validated.schedule is None:
            raise ValueError("activation scheduled with a schedule is required")
        job = await cron_scheduler.create_job(
            name=f"Agent system {validated.name}",
            tool_name="agent_system_tool",
            arguments={
                "spec_json": validated.model_dump_json(),
                "task": validated.task,
                "demo_mode": demo_mode,
            },
            timezone=validated.schedule.timezone,
            cron_expression=validated.schedule.cron,
        )
        logger.info(f"agent_systems_001: scheduled {validated.name} as {job['id']}")
        return ScheduleInfo(
            job_id=job["id"], next_run_at=job.get("next_run_at"), enabled=True
        )

    @staticmethod
    async def unschedule(job_id: str) -> None:
        await cron_scheduler.delete_job(job_id)
        logger.info(f"agent_systems_002: removed schedule {job_id}")

    @staticmethod
    async def schedule_info(job_id: str) -> ScheduleInfo:
        job = await cron_scheduler.get_job(job_id)
        return ScheduleInfo(
            job_id=job_id,
            next_run_at=job.get("next_run_at"),
            enabled=bool(job.get("enabled")),
        )


_service: SystemService | None = None


def configure_systems(agents: DynamicAgents, call: CompletionCall) -> SystemService:
    global _service
    _service = SystemService(agents, call)
    register_system_runner(_service.run_scheduled)
    logger.info("agent_systems_003: Agent systems ready")
    return _service


def system_service() -> SystemService:
    if _service is None:
        raise RuntimeError("Agent systems are not configured in this process")
    return _service
