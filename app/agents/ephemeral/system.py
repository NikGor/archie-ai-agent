"""Execution of agent systems (ARCHIE-339): one validated AgentSystemSpec, one
system run; every step is an ordinary AgentSpec run of DynamicAgents.

- centralized: the one agent does the task;
- workflow: steps run as soon as their dependencies are done, independent
  steps in parallel; the results of the dependencies are the step's facts;
- hierarchical: the supervisor plans one subtask per worker (plan_v1), the
  workers run in parallel, the supervisor combines their results;
- decentralized: peers answer independently, optional debate rounds with the
  other peers' findings, an optional aggregator merges or counts votes.

Agents get results of other agents as facts (summary and findings), the full
evidence stays in their own run records. The system budget is shared: every
step gets at most what is left.
"""

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from langfuse import get_client
from pydantic import BaseModel, JsonValue
from app.agents.ephemeral.dynamic import DynamicAgents, skill_names
from app.platform.runtime.runs import AgentRunRecord
from app.platform.runtime.spec import AgentSpec, CallerPolicy, Objective
from app.platform.runtime.system import (
    AgentSystemSpec,
    SystemNode,
    node_policy,
)
from app.platform.runtime.system_runs import (
    SYSTEM_TERMINAL_STATES,
    MemorySystemRunStore,
    SystemRunRecord,
    SystemRunStore,
    SystemStep,
)


logger = logging.getLogger(__name__)

FACT_CHARS = 600
FACTS_PER_RESULT = 6
PLAN_CRITERIA = ["Every assignment names a listed worker and one concrete subtask"]
PLAN_TASK = (
    "Plan the work. Give one subtask to each worker that the task needs. "
    "Do not do the subtasks yourself."
)
SYNTHESIS_TASK = (
    "Combine the results of the workers into one result. "
    "Name the gaps and the conflicts between the workers."
)
DEBATE_TASK = (
    "Other peers did the same task. Check their findings against yours, "
    "correct your mistakes and keep only what has evidence."
)
AGGREGATE_TASKS = {
    "independent": "Merge the peer results into one result. List where the peers disagree.",
    "debate": "Merge the peer results into one result. List where the peers still disagree.",
    "vote": "Each peer result is one vote. Choose the answer that most peers support and report the vote count.",
}


class SystemBudgetExceededError(RuntimeError):
    """The system budget is used up before a step."""


def result_facts(label: str, run: AgentRunRecord) -> list[str]:
    """What the next agent learns from a run: summary and supported findings."""
    result = run.result or {}
    head = result.get("summary") or run.detail or run.stop_reason or run.state
    facts = [f"{label} ({run.state}): {head}"]
    items = list(result.get("findings") or [])
    if result.get("answer"):
        items.insert(0, result["answer"])
    facts += [f"{label}: {item}" for item in items[: FACTS_PER_RESULT - 1]]
    return [fact[:FACT_CHARS] for fact in facts]


def with_task(task: str, step: str) -> str:
    return f"{step}\n\nOverall task: {task}" if step.strip() else task


class SystemRuntime:
    """Starts, tracks and cancels system runs of this process."""

    def __init__(
        self, agents: DynamicAgents, store: SystemRunStore | None = None
    ) -> None:
        self.agents = agents
        self.store = store or MemorySystemRunStore()
        self.records: dict[str, SystemRunRecord] = {}
        self.tasks: dict[str, asyncio.Task] = {}

    def start(
        self,
        spec: AgentSystemSpec,
        task: str,
        policy: CallerPolicy,
        trigger: str = "manual",
    ) -> SystemRunRecord:
        """Start in the background; the caller polls `status`."""
        record = SystemRunRecord(
            name=spec.name,
            task=task,
            trigger=trigger,
            demo_mode=policy.demo_mode,
            spec=spec.model_dump(mode="json"),
        )
        record.event("system_started", topology=spec.topology, nodes=len(spec.nodes))
        self.records[record.system_run_id] = record
        policy = policy.model_copy(update={"parent_run_id": record.system_run_id})
        self.tasks[record.system_run_id] = asyncio.create_task(
            self._guard(record, spec, policy), name=f"system-{record.system_run_id}"
        )
        logger.info(
            f"agent_system_run_001: {record.system_run_id} {spec.name} {spec.topology} started"
        )
        return record

    async def run(
        self, spec: AgentSystemSpec, task: str, policy: CallerPolicy, trigger: str
    ) -> SystemRunRecord:
        record = self.start(spec, task, policy, trigger)
        await asyncio.shield(self.tasks[record.system_run_id])
        return record

    async def status(self, system_run_id: str) -> SystemRunRecord | None:
        return self.records.get(system_run_id) or await self.store.get(system_run_id)

    async def cancel(self, system_run_id: str) -> bool:
        task = self.tasks.get(system_run_id)
        if task is None:
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def recent(self, limit: int = 30) -> list[SystemRunRecord]:
        stored = {r.system_run_id: r for r in await self.store.recent(limit)}
        merged = {**stored, **self.records}
        return sorted(merged.values(), key=lambda r: -r.created_at)[:limit]

    @contextmanager
    def observe(self, record: SystemRunRecord) -> Iterator[None]:
        """One Langfuse trace per system run; agent-run observations inside."""
        with get_client().start_as_current_observation(
            as_type="agent",
            name="agent-system",
            input=record.task,
            metadata={"system_run_id": record.system_run_id, "name": record.name},
        ) as observation:
            try:
                yield
            finally:
                observation.update(output=record.summary())

    async def _guard(
        self, record: SystemRunRecord, spec: AgentSystemSpec, policy: CallerPolicy
    ) -> None:
        await self.store.save(record)
        try:
            with self.observe(record):
                async with asyncio.timeout(spec.budget.max_seconds):
                    await self._execute(record, spec, policy)
        except TimeoutError:
            self._fail(record, "timeout", f"over {spec.budget.max_seconds:.0f} s")
        except SystemBudgetExceededError as exc:
            self._fail(record, "budget_exhausted", str(exc))
        except asyncio.CancelledError:
            record.finish("cancelled", "cancelled")
            record.event("system_cancelled")
        except Exception as exc:
            logger.exception(
                f"agent_system_run_error_001: {record.system_run_id}: {exc}"
            )
            self._fail(record, "error", f"{type(exc).__name__}: {exc}")
        finally:
            self.tasks.pop(record.system_run_id, None)
            await self.store.save(record)
            logger.info(
                f"agent_system_run_002: {record.system_run_id} {record.state} "
                f"${record.cost_usd:.4f} {len(record.steps)} steps"
            )

    @staticmethod
    def _fail(record: SystemRunRecord, reason: str, detail: str) -> None:
        record.detail = detail
        record.finish("failed", reason)
        record.event("system_failed", reason=reason, detail=detail)

    async def _execute(
        self, record: SystemRunRecord, spec: AgentSystemSpec, policy: CallerPolicy
    ) -> None:
        policy = node_policy(spec, policy)
        specs = await self._configure(record, spec, policy)
        if specs is None:
            return
        run = _SystemRun(self.agents, record, spec, specs, policy)
        final = await run.execute()
        if record.state in SYSTEM_TERMINAL_STATES:
            return
        record.result = (final.result or {}) if final else run.peer_result()
        if final is not None and final.state == "failed":
            self._fail(record, final.stop_reason or "failed", final.detail)
            return
        state = "waiting_for_approval" if record.proposed_actions else "completed"
        record.finish(state)
        record.event("system_completed", cost_usd=round(record.cost_usd, 5))

    async def _configure(
        self, record: SystemRunRecord, spec: AgentSystemSpec, policy: CallerPolicy
    ) -> dict[str, AgentSpec] | None:
        """Nodes without a spec get one from the configurator (dynamism dynamic)."""
        specs = {n.id: n.spec for n in spec.nodes if n.spec is not None}
        missing = [n for n in spec.nodes if n.spec is None]

        async def configure(node: SystemNode):
            return node, await self.agents.configurator.configure(
                with_task(record.task, node.step),
                f"Role in a {spec.topology} agent system: {node.role}",
                policy,
                skill_names(),
                allow_direct=False,
            )

        for node, result in await asyncio.gather(*(configure(n) for n in missing)):
            record.cost_usd += result.cost_usd
            if result.spec is None:
                self._fail(
                    record,
                    "invalid_spec",
                    f"node {node.id}: {result.reason}; {'; '.join(result.errors)}",
                )
                return None
            specs[node.id] = result.spec
            record.configured[node.id] = result.spec.model_dump(mode="json")
            record.event(
                "node_configured",
                node=node.id,
                name=result.spec.identity.name,
                profile=result.spec.model.profile,
                tools=list(result.spec.tools),
                cost_usd=round(result.cost_usd, 5),
            )
        return specs


class _SystemRun:
    """One execution of a topology; steps go into the system record."""

    def __init__(
        self,
        agents: DynamicAgents,
        record: SystemRunRecord,
        spec: AgentSystemSpec,
        specs: dict[str, AgentSpec],
        policy: CallerPolicy,
    ) -> None:
        self.agents = agents
        self.record = record
        self.spec = spec
        self.specs = specs
        self.policy = policy
        self.peers: dict[str, AgentRunRecord] = {}

    async def execute(self) -> AgentRunRecord | None:
        topology = self.spec.topology
        if topology == "workflow":
            return await self.workflow()
        if topology == "hierarchical":
            return await self.hierarchical()
        if topology == "decentralized":
            return await self.decentralized()
        node = self.spec.nodes[0]
        return await self.step(node.id, "run", with_task(self.record.task, node.step))

    async def step(
        self,
        node_id: str,
        phase: str,
        task: str,
        facts: list[str] | None = None,
        criteria: list[str] | None = None,
        **update: BaseModel,
    ) -> AgentRunRecord:
        """Run one node; its budget is at most what the system has left."""
        left = self.spec.budget.max_cost_usd - self.record.cost_usd
        if left <= 0:
            raise SystemBudgetExceededError(
                f"system budget ${self.spec.budget.max_cost_usd:.2f} is used up before {node_id}"
            )
        base = self.specs[node_id]
        spec = base.model_copy(
            update={
                "objective": Objective(
                    task=task[:2000],
                    completion_criteria=criteria or base.objective.completion_criteria,
                ),
                "context": base.context.model_copy(
                    update={"facts": (base.context.facts + (facts or []))[:20]}
                ),
                "budget": base.budget.model_copy(
                    update={"max_cost_usd": min(base.budget.max_cost_usd or left, left)}
                ),
                **update,
            }
        )
        self.record.event("step_started", node=node_id, phase=phase)
        run = await self.agents.run(
            task,
            spec,
            self.policy,
            context=f"{self.record.system_run_id}:{node_id}:{phase}",
        )
        result = run.result or {}
        self.record.cost_usd += run.cost_usd
        self.record.steps.append(
            SystemStep(
                node_id=node_id,
                phase=phase,
                run_id=run.run_id,
                state=run.state,
                stop_reason=run.stop_reason,
                cost_usd=round(run.cost_usd, 5),
                seconds=run.seconds,
                summary=str(result.get("summary") or run.detail)[:FACT_CHARS],
            )
        )
        self.record.proposed_actions += [
            {**action, "node": node_id}
            for action in result.get("proposed_actions") or []
        ]
        self.record.event(
            "step_finished",
            node=node_id,
            phase=phase,
            run_id=run.run_id,
            state=run.state,
            cost_usd=round(run.cost_usd, 5),
        )
        return run

    async def workflow(self) -> AgentRunRecord | None:
        done: dict[str, AgentRunRecord] = {}
        pending = list(self.spec.nodes)
        while pending:
            ready = [n for n in pending if all(d in done for d in n.depends_on)]
            runs = await asyncio.gather(
                *(
                    self.step(
                        n.id,
                        "step",
                        with_task(self.record.task, n.step),
                        [f for d in n.depends_on for f in result_facts(d, done[d])],
                    )
                    for n in ready
                )
            )
            done.update({n.id: run for n, run in zip(ready, runs, strict=True)})
            pending = [n for n in pending if n.id not in done]
        used = {d for n in self.spec.nodes for d in n.depends_on}
        sinks = [n.id for n in self.spec.nodes if n.id not in used]
        if len(sinks) == 1:
            return done[sinks[0]]
        self.peers = {node_id: done[node_id] for node_id in sinks}
        return None

    async def hierarchical(self) -> AgentRunRecord:
        supervisor = next(n for n in self.spec.nodes if n.role == "supervisor")
        workers = {n.id: n for n in self.spec.nodes if n.role == "worker"}
        roster = [
            f"worker {w.id}: {self.specs[w.id].identity.role}; "
            f"{self.specs[w.id].identity.purpose}; tools: {', '.join(self.specs[w.id].tools) or 'none'}"
            for w in workers.values()
        ]
        base = self.specs[supervisor.id]
        plan = await self.step(
            supervisor.id,
            "plan",
            with_task(self.record.task, f"{supervisor.step}\n{PLAN_TASK}".strip()),
            roster,
            PLAN_CRITERIA,
            output=base.output.model_copy(update={"schema_name": "plan_v1"}),
        )
        assignments: dict[str, str] = {}
        for item in (plan.result or {}).get("assignments") or []:
            if item.get("worker") in workers and item["worker"] not in assignments:
                assignments[item["worker"]] = item.get("task") or ""
        if not assignments:
            # No usable plan: every worker gets its own step of the task
            assignments = {w.id: w.step for w in workers.values()}
        runs = await asyncio.gather(
            *(
                self.step(worker, "work", with_task(self.record.task, subtask))
                for worker, subtask in assignments.items()
            )
        )
        facts = [
            f
            for worker, run in zip(assignments, runs, strict=True)
            for f in result_facts(worker, run)
        ]
        return await self.step(
            supervisor.id,
            "synthesis",
            with_task(self.record.task, SYNTHESIS_TASK),
            facts,
        )

    async def decentralized(self) -> AgentRunRecord | None:
        peers = [n for n in self.spec.nodes if n.role == "peer"]
        aggregator = next((n for n in self.spec.nodes if n.role == "aggregator"), None)
        coordination = self.spec.coordination
        runs = await asyncio.gather(
            *(
                self.step(p.id, "round_0", with_task(self.record.task, p.step))
                for p in peers
            )
        )
        latest = {p.id: run for p, run in zip(peers, runs, strict=True)}
        rounds = coordination.debate_rounds if coordination.protocol == "debate" else 0
        for number in range(1, rounds + 1):
            runs = await asyncio.gather(
                *(
                    self.step(
                        p.id,
                        f"round_{number}",
                        with_task(self.record.task, DEBATE_TASK),
                        [
                            f
                            for other, run in latest.items()
                            if other != p.id
                            for f in result_facts(other, run)
                        ],
                    )
                    for p in peers
                )
            )
            latest = {p.id: run for p, run in zip(peers, runs, strict=True)}
        self.peers = latest
        if aggregator is None:
            return None
        return await self.step(
            aggregator.id,
            "aggregate",
            with_task(self.record.task, AGGREGATE_TASKS[coordination.protocol]),
            [f for peer, run in latest.items() for f in result_facts(peer, run)],
        )

    def peer_result(self) -> dict[str, JsonValue]:
        """Result without one final agent: every branch as it finished."""
        return {
            "summary": " ".join(
                f"{peer}: {(run.result or {}).get('summary') or run.state}"
                for peer, run in self.peers.items()
            ),
            "branches": {peer: run.summary() for peer, run in self.peers.items()},
        }
