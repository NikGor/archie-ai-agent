"""AgentSystemSpec: several AgentSpec agents organized along the five dimensions
of the Solaris research (ARCHIE-338): control topology, activation, lifecycle,
autonomy and organizational dynamism.

Contract: Task + AgentSystemSpec + CallerPolicy -> validated system -> SystemRun.
`check_system` returns every problem at once (the constructor shows them while
the user edits); `validate_system` refuses to run a system with errors. Which
option of a dimension the runtime can execute is decided here only:
`DIMENSIONS` feeds both the validator and the constructor catalog.
"""

import logging
from collections.abc import Iterable
from typing import Literal
from croniter import croniter  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, ValidationError
from app.platform.runtime.spec import (
    AgentSpec,
    CallerPolicy,
    SpecModel,
    SpecValidationError,
    validate_spec,
)
from app.platform.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)

Topology = Literal["centralized", "workflow", "hierarchical", "decentralized"]
Activation = Literal["reactive", "scheduled", "event_driven", "proactive"]
Lifecycle = Literal["ephemeral", "durable", "continuous"]
Dynamism = Literal["static", "configurable", "dynamic"]
NodeRole = Literal["agent", "supervisor", "worker", "peer", "aggregator"]
Protocol = Literal["independent", "debate", "vote"]
ExternalActions = Literal["none", "propose", "execute"]

MAX_NODES = 6
MAX_WORKERS = 4
MAX_PEERS = 4
MAX_DEBATE_ROUNDS = 2


class DimensionOption(BaseModel):
    value: str = Field(description="Option code")
    label: str = Field(description="Name for the constructor")
    description: str = Field(description="What the option means")
    available: bool = Field(description="The runtime can execute it")
    reason: str = Field("", description="Why it is not available")


class Dimension(BaseModel):
    name: str = Field(description="Dimension code")
    label: str = Field(description="Name for the constructor")
    question: str = Field(description="What the dimension measures")
    options: list[DimensionOption]

    def option(self, value: str) -> DimensionOption | None:
        return next((o for o in self.options if o.value == value), None)


def _option(
    value: str, label: str, description: str, reason: str = ""
) -> DimensionOption:
    return DimensionOption(
        value=value,
        label=label,
        description=description,
        available=not reason,
        reason=reason,
    )


DIMENSIONS: dict[str, Dimension] = {
    d.name: d
    for d in (
        Dimension(
            name="topology",
            label="Топология управления",
            question="Кто управляет зависимостями и принимает решения",
            options=[
                _option(
                    "centralized",
                    "Централизованная",
                    "Один агент сам выбирает шаги и tools",
                ),
                _option(
                    "workflow",
                    "Workflow / DAG",
                    "Шаги и зависимости заданы заранее; независимые шаги идут параллельно",
                ),
                _option(
                    "hierarchical",
                    "Иерархическая",
                    "Supervisor раздаёт подзадачи workers и собирает итог",
                ),
                _option(
                    "decentralized",
                    "Децентрализованная",
                    "Равные агенты решают независимо, обмениваются выводами, агрегатор сводит",
                ),
            ],
        ),
        Dimension(
            name="activation",
            label="Активация",
            question="Что запускает работу",
            options=[
                _option("reactive", "По запросу", "Запуск вручную или через API"),
                _option("scheduled", "По расписанию", "Запуск по cron-выражению"),
                _option(
                    "event_driven",
                    "По событию",
                    "Запуск по событию датчика или другого агента",
                    "Нет подписки систем на события дома; события пока видит только рой",
                ),
                _option(
                    "proactive",
                    "Проактивная",
                    "Система сама находит потребность и ставит цель",
                    "Нужно право ставить цели; платформа его не выдаёт",
                ),
            ],
        ),
        Dimension(
            name="lifecycle",
            label="Жизненный цикл",
            question="Сколько живёт логическое состояние",
            options=[
                _option(
                    "ephemeral", "Ephemeral", "Run заканчивается с результатом задачи"
                ),
                _option(
                    "continuous",
                    "Continuous",
                    "Деятельность без точки завершения: много коротких run по расписанию",
                ),
                _option(
                    "durable",
                    "Durable",
                    "Run хранит checkpoint, ждёт пользователя и переживает перезапуск",
                    "Нужны checkpoints и восстановление (Temporal, ARCHIE-301)",
                ),
            ],
        ),
        Dimension(
            name="dynamism",
            label="Организационная динамика",
            question="Может ли меняться состав и схема",
            options=[
                _option(
                    "static",
                    "Статическая",
                    "Агенты и задача зафиксированы; запуск без ввода",
                ),
                _option(
                    "configurable",
                    "Конфигурируемая",
                    "Агенты зафиксированы, задача задаётся при каждом запуске",
                ),
                _option(
                    "dynamic",
                    "Динамическая",
                    "Участников без спеки собирает конфигуратор под задачу при запуске",
                ),
            ],
        ),
    )
}

# Autonomy is a set of separate powers, not a scale (research: mechanisms/activation-lifecycle)
AUTONOMY_POWERS: list[DimensionOption] = [
    _option("choose_tools", "Выбор действий", "Агент сам выбирает tool и аргументы"),
    _option(
        "plan",
        "Планирование",
        "Агент сам строит план из нескольких шагов; без него — один раунд tools",
    ),
    _option(
        "propose",
        "Предлагать изменения",
        "Изменения мира идут как предложения на подтверждение",
    ),
    _option(
        "execute",
        "Исполнять изменения",
        "Агент сам меняет мир",
        "Платформа не исполняет write-tools агентов; только предложения на подтверждение",
    ),
    _option(
        "set_goals",
        "Ставить цели",
        "Агент сам формулирует новые задачи",
        "Право ставить цели не выдаётся",
    ),
    _option(
        "spawn_agents",
        "Порождать агентов",
        "Агент сам создаёт других агентов",
        "Рекурсивный spawn запрещён: состав задаёт система",
    ),
]

TOPOLOGY_ROLES: dict[str, set[str]] = {
    "centralized": {"agent"},
    "workflow": {"agent"},
    "hierarchical": {"supervisor", "worker"},
    "decentralized": {"peer", "aggregator"},
}


class Autonomy(SpecModel):
    plan: bool = Field(True, description="Agents plan several tool rounds")
    external_actions: ExternalActions = Field(
        "none", description="none: read only; propose: changes wait for approval"
    )
    set_goals: bool = Field(False, description="Agents formulate new tasks")
    spawn_agents: bool = Field(False, description="Agents create other agents")


class Coordination(SpecModel):
    protocol: Protocol = Field(
        "independent", description="How peers of a decentralized system agree"
    )
    debate_rounds: int = Field(
        1, ge=0, le=MAX_DEBATE_ROUNDS, description="Peer review rounds for debate"
    )


class Schedule(SpecModel):
    cron: str = Field(min_length=9, max_length=100, description="Five-field cron")
    timezone: str = Field("Europe/Berlin", description="IANA timezone of the cron")


class SystemBudget(SpecModel):
    max_cost_usd: float = Field(0.3, gt=0, le=2.0, description="USD for one run")
    max_seconds: float = Field(300, gt=0, le=1800, description="Wall time of one run")


class SystemNode(SpecModel):
    id: str = Field(
        min_length=1, max_length=32, pattern=r"^[a-z][a-z0-9_]*$", description="Slug"
    )
    role: NodeRole = Field("agent", description="Place of the node in the topology")
    step: str = Field(
        "", max_length=1000, description="What this node does with the task"
    )
    depends_on: list[str] = Field(
        default_factory=list, max_length=MAX_NODES, description="Workflow inputs"
    )
    spec: AgentSpec | None = Field(
        None, description="Agent of the node; null: the configurator builds it"
    )


class AgentSystemSpec(SpecModel):
    system_version: Literal["1"] = Field("1", description="Version of this format")
    name: str = Field(
        min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$", description="Slug"
    )
    description: str = Field("", max_length=500)
    topology: Topology = "centralized"
    activation: Activation = "reactive"
    lifecycle: Lifecycle = "ephemeral"
    dynamism: Dynamism = "configurable"
    autonomy: Autonomy = Field(default_factory=Autonomy)
    coordination: Coordination = Field(default_factory=Coordination)
    nodes: list[SystemNode] = Field(min_length=1, max_length=MAX_NODES)
    task: str = Field("", max_length=2000, description="Fixed task (static, schedule)")
    schedule: Schedule | None = None
    budget: SystemBudget = Field(default_factory=SystemBudget)

    def node(self, node_id: str) -> SystemNode:
        return next(n for n in self.nodes if n.id == node_id)


class SystemCheck(BaseModel):
    valid: bool = Field(description="No errors: the system may run")
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    adjustments: list[str] = Field(default_factory=list, description="Caps applied")
    spec: AgentSystemSpec | None = Field(None, description="Spec after caps")


class SystemValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def _dimension_errors(spec: AgentSystemSpec) -> list[str]:
    errors = []
    for name in DIMENSIONS:
        value = getattr(spec, name)
        option = DIMENSIONS[name].option(value)
        if option is not None and not option.available:
            errors.append(f"{name} {value}: {option.reason}")
    powers = {p.value: p for p in AUTONOMY_POWERS}
    if spec.autonomy.external_actions == "execute":
        errors.append(f"autonomy execute: {powers['execute'].reason}")
    if spec.autonomy.set_goals:
        errors.append(f"autonomy set_goals: {powers['set_goals'].reason}")
    if spec.autonomy.spawn_agents:
        errors.append(f"autonomy spawn_agents: {powers['spawn_agents'].reason}")
    return errors


def _cycle(nodes: list[SystemNode]) -> list[str]:
    """Node ids on a dependency cycle (Kahn's algorithm leaves them)."""
    pending = {n.id: set(n.depends_on) for n in nodes}
    while True:
        ready = [i for i, deps in pending.items() if not deps]
        if not ready:
            return sorted(pending)
        for node_id in ready:
            del pending[node_id]
        for deps in pending.values():
            deps.difference_update(ready)


def _workflow_errors(nodes: list[SystemNode]) -> list[str]:
    errors = ["workflow needs at least two steps"] if len(nodes) < 2 else []
    known = {n.id for n in nodes}
    for n in nodes:
        errors += [
            f"node {n.id}: unknown dependency {d}"
            for d in n.depends_on
            if d not in known
        ]
        if n.id in n.depends_on:
            errors.append(f"node {n.id} depends on itself")
    if not errors and (cycle := _cycle(nodes)):
        errors.append(f"workflow has a cycle: {', '.join(cycle)}")
    return errors


def _team_errors(spec: AgentSystemSpec, count: dict[str, int]) -> list[str]:
    """Hierarchical and decentralized: how many of each role."""
    errors = []
    if spec.topology == "hierarchical":
        if count["supervisor"] != 1:
            errors.append("hierarchical needs exactly one supervisor")
        if not 1 <= count["worker"] <= MAX_WORKERS:
            errors.append(f"hierarchical needs 1 to {MAX_WORKERS} workers")
        if not spec.autonomy.plan:
            errors.append("hierarchical needs autonomy plan: the supervisor plans")
    else:
        if not 2 <= count["peer"] <= MAX_PEERS:
            errors.append(f"decentralized needs 2 to {MAX_PEERS} peers")
        if count["aggregator"] > 1:
            errors.append("decentralized has at most one aggregator")
        if spec.coordination.protocol == "vote" and not count["aggregator"]:
            errors.append("protocol vote needs an aggregator that counts the votes")
    return errors


def _topology_errors(spec: AgentSystemSpec) -> list[str]:
    nodes = spec.nodes
    ids = [n.id for n in nodes]
    errors = [
        f"node id {i} is used twice" for i in sorted(set(ids)) if ids.count(i) > 1
    ]
    roles = TOPOLOGY_ROLES[spec.topology]
    errors += [
        f"node {n.id}: role {n.role} does not fit topology {spec.topology}"
        for n in nodes
        if n.role not in roles
    ]
    if spec.topology == "workflow":
        return errors + _workflow_errors(nodes)
    errors += [
        f"node {n.id}: depends_on is only for workflow" for n in nodes if n.depends_on
    ]
    if spec.topology == "centralized":
        return errors + (
            ["centralized needs exactly one agent"] if len(nodes) != 1 else []
        )
    count = {role: sum(1 for n in nodes if n.role == role) for role in roles}
    return errors + _team_errors(spec, count)


def _activation_errors(spec: AgentSystemSpec) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    if spec.activation == "scheduled":
        if spec.schedule is None:
            errors.append("activation scheduled needs a schedule")
        elif len(spec.schedule.cron.split()) != 5 or not croniter.is_valid(
            spec.schedule.cron
        ):
            errors.append(f"invalid cron expression: {spec.schedule.cron}")
        if not spec.task.strip():
            errors.append("a scheduled run needs a fixed task")
        if spec.lifecycle == "ephemeral":
            warnings.append(
                "regular short runs are lifecycle continuous in the research model"
            )
    elif spec.schedule is not None:
        warnings.append("the schedule is ignored: activation is not scheduled")
    if spec.lifecycle == "continuous" and spec.activation != "scheduled":
        errors.append("lifecycle continuous needs activation scheduled")
    if spec.dynamism == "static" and not spec.task.strip():
        errors.append("dynamism static needs a fixed task")
    return errors, warnings


def node_policy(spec: AgentSystemSpec, policy: CallerPolicy) -> CallerPolicy:
    """Autonomy narrows what the caller grants to every node."""
    update: dict = {
        "may_request_approval": policy.may_request_approval
        and spec.autonomy.external_actions == "propose"
    }
    if not spec.autonomy.plan:
        update["limits"] = policy.limits.model_copy(update={"max_iterations": 1})
    return policy.model_copy(update=update)


def check_system(
    draft: AgentSystemSpec | dict,
    policy: CallerPolicy,
    registry: ToolRegistry,
    skills: Iterable[str] = (),
) -> SystemCheck:
    """Every error, warning and cap of a system; nodes are checked as AgentSpec."""
    try:
        spec = (
            draft
            if isinstance(draft, AgentSystemSpec)
            else AgentSystemSpec.model_validate(draft)
        )
    except ValidationError as exc:
        return SystemCheck(
            valid=False,
            errors=[
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
            ],
        )
    errors = _dimension_errors(spec) + _topology_errors(spec)
    activation_errors, warnings = _activation_errors(spec)
    errors += activation_errors
    adjustments: list[str] = []
    budget = spec.budget.model_copy(
        update={
            "max_cost_usd": min(
                spec.budget.max_cost_usd, policy.budget.max_cost_usd or 0.0
            ),
            "max_seconds": min(
                spec.budget.max_seconds, policy.budget.max_seconds or 0.0
            ),
        }
    )
    if budget != spec.budget:
        adjustments.append(
            f"system budget capped: {spec.budget.model_dump()} -> {budget.model_dump()}"
        )
    granted = node_policy(spec, policy)
    nodes, planned_cost = [], 0.0
    for node in spec.nodes:
        if node.spec is None:
            if spec.dynamism != "dynamic":
                errors.append(
                    f"node {node.id}: needs an agent spec (dynamism {spec.dynamism})"
                )
            elif not node.step.strip():
                errors.append(
                    f"node {node.id}: the configurator needs a step description"
                )
            nodes.append(node)
            continue
        try:
            validated = validate_spec(node.spec, granted, registry, skills)
        except SpecValidationError as exc:
            errors += [f"node {node.id}: {e}" for e in exc.errors]
            nodes.append(node)
            continue
        adjustments += [f"node {node.id}: {a}" for a in validated.adjustments]
        runs = (
            2
            if node.role == "supervisor"
            else 1
            + (
                spec.coordination.debate_rounds
                if node.role == "peer" and spec.coordination.protocol == "debate"
                else 0
            )
        )
        planned_cost += runs * (validated.spec.budget.max_cost_usd or 0.0)
        nodes.append(node.model_copy(update={"spec": validated.spec}))
    if planned_cost > budget.max_cost_usd:
        warnings.append(
            f"node budgets add up to ${planned_cost:.2f} over the system budget "
            f"${budget.max_cost_usd:.2f}: later nodes get what is left"
        )
    capped = spec.model_copy(update={"nodes": nodes, "budget": budget})
    if errors:
        logger.info(f"agent_system_001: {spec.name}: {len(errors)} errors")
    return SystemCheck(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        adjustments=adjustments,
        spec=capped,
    )


def validate_system(
    draft: AgentSystemSpec | dict,
    policy: CallerPolicy,
    registry: ToolRegistry,
    skills: Iterable[str] = (),
) -> AgentSystemSpec:
    check = check_system(draft, policy, registry, skills)
    if not check.valid or check.spec is None:
        raise SystemValidationError(check.errors)
    return check.spec
