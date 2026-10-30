"""Dynamic Configured Agent vs Fixed Configured Agent (ARCHIE-326).

Runs every task of dynamic_tasks.json on five baselines, with fixed tool
results (no real tool runs), and writes raw JSONL plus a summary:

    strong          one MAXIMUM agent with every caller tool (baseline 1)
    fixed           a fixed spec per task class; HA diagnosis = swarm preset (baseline 2)
    dynamic         configurator -> spec or direct (baseline 3, H1, H3)
    dynamic_spawn   configurator without direct (H3)
    dynamic_max     the dynamic spec on MAXIMUM (H4: escalation vs expensive profile)

    PYTHONPATH=. poetry run python scripts/benchmark/dynamic_vs_fixed.py --repeats 3 --out <dir>
"""

import argparse
import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

from app.agents.ephemeral.configurator import AgentConfigurator  # noqa: E402
from app.agents.ephemeral.dynamic import DynamicAgents  # noqa: E402
from app.agents.ephemeral.engine import AgentFactory, AgentRuntime  # noqa: E402
from app.agents.ephemeral.models import SpawnRequest  # noqa: E402
from app.agents.ephemeral.runner import EphemeralRunner  # noqa: E402
from app.config import settings  # noqa: E402
from app.platform.llm.decisions import decide  # noqa: E402
from app.platform.runtime.budget import Budget  # noqa: E402
from app.platform.runtime.runs import RunManager  # noqa: E402
from app.platform.runtime.spawn import SpawnContext  # noqa: E402
from app.platform.runtime.spec import CallerPolicy, Execution  # noqa: E402
from app.platform.tools.models import ToolResult  # noqa: E402
from app.tools.catalog import tool_registry  # noqa: E402


TASKS = Path(__file__).with_name("dynamic_tasks.json")
BUDGET = Budget(max_tokens=80000, max_cost_usd=0.15, max_seconds=150, max_tool_rounds=4)
LIMITS = Execution(max_iterations=4, max_tool_calls=8, timeout_seconds=150, retries=1)
JUDGE = "~typesafe/jev-latest"
FIXED_CLASS = {
    "trivial": (
        "FAST",
        "answer_v1",
        ["google_search_tool", "google_places_search_tool", "smarthome_status_tool"],
    ),
    "research": (
        "BALANCED",
        "findings_v1",
        ["google_search_tool", "google_places_search_tool"],
    ),
    "analysis": ("BALANCED", "findings_v1", ["events_tool", "document_search_tool"]),
    "permission": (
        "BALANCED",
        "findings_v1",
        [
            "smarthome_status_tool",
            "events_tool",
            "smarthome_control_tool",
            "notes_tool",
        ],
    ),
    "conflict": ("BALANCED", "findings_v1", ["google_search_tool"]),
}
WRITE_TOOLS = ["smarthome_control_tool", "notes_tool"]


def fixture_runner(task: dict):
    """Fixed tool results: the first fixture key found in the arguments, else default."""

    async def run(calls, _context):
        results = []
        for call in calls:
            options = task["fixtures"].get(call.tool_name)
            if options is None:
                results.append(
                    ToolResult(
                        tool_name=call.tool_name,
                        output={"error": "no data"},
                        success=False,
                    )
                )
                continue
            text = json.dumps([p.value for p in call.arguments], ensure_ascii=False)
            key = next(
                (k for k in options if k != "default" and k.lower() in text.lower()),
                "default",
            )
            results.append(ToolResult(tool_name=call.tool_name, output=options[key]))
        return results

    return run


def policy(caller_tools: list[str], parent: str) -> CallerPolicy:
    return CallerPolicy.from_spawn(
        SpawnContext(source="benchmark", parent_run_id=parent, demo_mode=True),
        caller_tools,
        budget=BUDGET,
        limits=LIMITS,
        max_profile="MAXIMUM",
        may_request_approval=True,
    )


def spec(
    name: str,
    task: str,
    profile: str,
    schema: str,
    tools: list[str],
    effort: str | None,
    criteria: list[str],
) -> dict:
    writes = [t for t in tools if t in WRITE_TOOLS]
    return {
        "identity": {
            "name": name,
            "role": "Assistant agent",
            "purpose": "Solve the task",
        },
        "objective": {"task": task, "completion_criteria": criteria},
        "instructions": "Use the tools to get the facts, then answer the task with evidence.",
        "model": {"profile": profile, "effort": effort},
        "tools": tools,
        "execution": LIMITS.model_dump(),
        "budget": BUDGET.model_dump(),
        "permissions": {
            "mode": "propose_actions" if writes else "read_only",
            "approval_tools": writes,
        },
        "output": {"schema_name": schema},
    }


def text_of(record) -> str:
    result = record.result or {}
    parts = [
        result.get("summary") or "",
        result.get("answer") or "",
        *(result.get("findings") or []),
    ]
    return "\n".join(str(p) for p in parts) + "\n" + record.detail


def check(task: dict, record) -> tuple[bool, list[str]]:
    """Deterministic checks of the final state."""
    problems = []
    proposal = task.get("expect_proposal")
    if proposal:
        tools = [
            a["tool_name"] for a in (record.result or {}).get("proposed_actions", [])
        ]
        if record.state != "waiting_for_approval" or proposal not in tools:
            problems.append(
                f"expected a proposal of {proposal}, got {record.state} {tools}"
            )
    elif record.state not in ("completed",):
        problems.append(f"state {record.state} {record.stop_reason}")
    text = text_of(record).casefold()
    for group in task.get("require", []):
        if not any(item.casefold() in text for item in group):
            problems.append(f"missing {group}")
    for group in task.get("forbid", []):
        if any(item.casefold() in text for item in group):
            problems.append(f"forbidden {group}")
    return not problems, problems


async def judge(task: dict, record) -> bool | None:
    try:
        data = await decide(
            JUDGE,
            {
                "task": task["task"],
                "result": record.summary(),
                "evidence": record.evidence[:6],
            },
            {
                "goal": {
                    "type": "choice",
                    "instructions": "Does the result solve the task with the evidence, and state conflicts or approvals that the task needs?",
                    "criteria": {
                        "achieved": "Solved with evidence",
                        "not_achieved": "Missing, wrong or unsupported",
                    },
                }
            },
        )
        probabilities = data["answers"]["goal"]["probabilities"]
        return max(probabilities, key=probabilities.get) == "achieved"
    except Exception as exc:
        print(f"judge failed: {exc}")
        return None


class Bench:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.caller_tools = data["caller_tools"]

    def agents(self, task: dict) -> DynamicAgents:
        runtime = AgentRuntime(tool_registry, run_tools=fixture_runner(task))
        return DynamicAgents(
            manager=RunManager(),
            runtime=runtime,
            factory=AgentFactory(tool_registry),
            configurator=AgentConfigurator(tool_registry),
        )

    async def strong(self, task: dict, parent: str):
        draft = spec(
            "strong",
            task["task"],
            "MAXIMUM",
            "findings_v1",
            self.caller_tools,
            "high",
            ["The task is solved with evidence"],
        )
        return await self.agents(task).run(
            task["task"], draft, policy(self.caller_tools, parent)
        )

    async def fixed(self, task: dict, parent: str):
        if task["category"] == "diagnosis":
            return await self.ha_preset(task, parent)
        profile, schema, tools = FIXED_CLASS[task["category"]]
        draft = spec(
            f"fixed_{task['category']}",
            task["task"],
            profile,
            schema,
            tools,
            None,
            ["The task is solved with evidence"],
        )
        return await self.agents(task).run(
            task["task"], draft, policy(self.caller_tools, parent)
        )

    async def ha_preset(self, task: dict, parent: str):
        """The swarm's fixed HA diagnosis (ARCHIE-300 preset), mapped to a record shape."""
        runner = EphemeralRunner(
            run_tools=fixture_runner(task), model=settings.ephemeral_model
        )
        started = time.monotonic()
        result = await runner.run(
            SpawnRequest(goal=task["task"], parent_run_id=parent, publish_to="none"),
            SpawnContext(source="benchmark", parent_run_id=parent, demo_mode=True),
        )
        manager = RunManager()
        recorder, _ = manager.create(task["task"], parent, "benchmark")
        record = recorder.record
        record.state = "completed" if result.status == "completed" else "failed"
        record.stop_reason = None if result.status == "completed" else result.status
        record.result = {
            "summary": result.summary,
            "findings": result.findings,
            "proposed_actions": [],
        }
        record.evidence = [e.model_dump(mode="json") for e in result.evidence]
        record.cost_usd = result.usage.total_cost
        record.input_tokens, record.output_tokens = (
            result.usage.input_tokens,
            result.usage.output_tokens,
        )
        record.model = settings.ephemeral_model
        record.finished_at = record.created_at + (time.monotonic() - started)
        return record

    async def dynamic(self, task: dict, parent: str, allow_direct: bool = True):
        agents = self.agents(task)
        if not allow_direct:
            original = agents.configurator.configure

            async def configure(*args, **kwargs):
                return await original(*args, allow_direct=False, **kwargs)

            agents.configurator.configure = configure
        record = await agents.delegate(
            task["task"], "", policy(self.caller_tools, parent)
        )
        if record.stop_reason == "direct":
            # Archie does it itself: one FAST answer with one tool round
            direct = await self.agents(task).run(
                task["task"],
                spec(
                    "direct",
                    task["task"],
                    "FAST",
                    "answer_v1",
                    [t for t in self.caller_tools if t not in WRITE_TOOLS],
                    None,
                    ["The answer gives the requested value"],
                )
                | {"execution": {**LIMITS.model_dump(), "max_iterations": 1}},
                policy(self.caller_tools, parent + "-direct"),
            )
            direct.configuration = record.configuration
            direct.configurator_cost_usd = record.configurator_cost_usd
            direct.cost_usd += record.cost_usd
            direct.stop_reason = "direct"
            direct.finished_at = (direct.finished_at or time.time()) + (
                record.seconds or 0
            )
            return direct
        return record

    async def dynamic_max(self, task: dict, parent: str, configured: dict | None):
        if not configured:
            return None
        draft = dict(configured)
        draft["model"] = {**draft["model"], "profile": "MAXIMUM", "effort": "high"}
        return await self.agents(task).run(
            task["task"], draft, policy(self.caller_tools, parent)
        )


async def one(
    bench: Bench, task: dict, baseline: str, repeat: int, configured: dict | None = None
) -> dict:
    parent = f"bench-{uuid.uuid4().hex[:10]}"
    started = time.monotonic()
    try:
        if baseline == "dynamic_max":
            record = await bench.dynamic_max(task, parent, configured)
            if record is None:
                return {}
        elif baseline == "dynamic_spawn":
            record = await bench.dynamic(task, parent, allow_direct=False)
        else:
            record = await getattr(bench, baseline)(task, parent)
    except Exception as exc:
        return {
            "task": task["id"],
            "category": task["category"],
            "baseline": baseline,
            "repeat": repeat,
            "error": f"{type(exc).__name__}: {exc}",
        }
    passed, problems = check(task, record)
    verdict = await judge(task, record)
    return {
        "task": task["id"],
        "category": task["category"],
        "baseline": baseline,
        "repeat": repeat,
        "passed": passed,
        "problems": problems,
        "judge_achieved": verdict,
        "state": record.state,
        "stop_reason": record.stop_reason,
        "decision": (record.configuration or {}).get("decision"),
        "spec": record.spec,
        "model": record.model,
        "effort": record.effort,
        "escalations": record.escalations,
        "retries": record.retries,
        "policy_violations": record.policy_violations,
        "loop_guards": record.loop_guards,
        "tools_used": record.tools_used,
        "tool_calls": record.tool_calls,
        "cost_usd": round(record.cost_usd, 6),
        "configurator_cost_usd": round(record.configurator_cost_usd, 6),
        "tokens": record.input_tokens + record.output_tokens,
        "seconds": round(time.monotonic() - started, 2),
        "result": record.result,
        "evidence_items": len(record.evidence),
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    for baseline in sorted({r["baseline"] for r in rows}):
        group = [r for r in rows if r["baseline"] == baseline and "passed" in r]
        ok = [r for r in group if r["passed"]]
        latency = sorted(r["seconds"] for r in group)
        out[baseline] = {
            "runs": len(group),
            "errors": sum(
                1 for r in rows if r["baseline"] == baseline and "error" in r
            ),
            "success_rate": round(len(ok) / len(group), 3) if group else None,
            "judge_rate": round(
                sum(1 for r in group if r["judge_achieved"]) / len(group), 3
            )
            if group
            else None,
            "cost_total_usd": round(sum(r["cost_usd"] for r in group), 4),
            "cost_per_success_usd": round(
                sum(r["cost_usd"] for r in group) / len(ok), 5
            )
            if ok
            else None,
            "latency_p50": statistics.median(latency) if latency else None,
            "latency_p95": latency[min(len(latency) - 1, int(0.95 * len(latency)))]
            if latency
            else None,
            "tokens_median": statistics.median(r["tokens"] for r in group)
            if group
            else None,
            "policy_violation_runs": sum(1 for r in group if r["policy_violations"]),
            "loop_guard_runs": sum(1 for r in group if r["loop_guards"]),
            "retries": sum(r["retries"] for r in group),
            "escalated_runs": sum(1 for r in group if r["escalations"]),
            "direct_decisions": sum(1 for r in group if r.get("decision") == "direct"),
            "judge_disagreements": sum(
                1
                for r in group
                if r["judge_achieved"] is not None
                and r["judge_achieved"] != r["passed"]
            ),
            "by_category": {
                c: round(
                    sum(1 for r in group if r["category"] == c and r["passed"])
                    / max(1, sum(1 for r in group if r["category"] == c)),
                    3,
                )
                for c in sorted({r["category"] for r in group})
            },
        }
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--task", nargs="*")
    parser.add_argument(
        "--baseline", nargs="*", default=["strong", "fixed", "dynamic", "dynamic_spawn"]
    )
    args = parser.parse_args()
    data = json.loads(TASKS.read_text())
    tasks = [t for t in data["tasks"] if not args.task or t["id"] in args.task]
    bench = Bench(data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []

    async def guarded(task, baseline, repeat, configured=None):
        async with gate:
            row = await one(bench, task, baseline, repeat, configured)
        if row:
            rows.append(row)
            print(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in (
                            "task",
                            "baseline",
                            "repeat",
                            "passed",
                            "judge_achieved",
                            "state",
                            "decision",
                            "cost_usd",
                            "seconds",
                            "problems",
                            "error",
                        )
                    },
                    ensure_ascii=False,
                )
            )
        return row

    jobs = [
        guarded(t, b, r)
        for r in range(args.repeats)
        for t in tasks
        for b in args.baseline
    ]
    await asyncio.gather(*jobs)
    if "dynamic_max" not in args.baseline:
        paired = [
            guarded(
                next(t for t in tasks if t["id"] == row["task"]),
                "dynamic_max",
                row["repeat"],
                row["spec"],
            )
            for row in list(rows)
            if row.get("baseline") == "dynamic"
            and row.get("spec")
            and row.get("decision") == "spawn"
        ]
        await asyncio.gather(*paired)
    (out / "raw.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )
    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "repeats": args.repeats,
        "tasks": len(tasks),
        "profiles": settings.agent_profiles,
        "configurator_profile": settings.agent_configurator_profile,
        "ha_preset_model": settings.ephemeral_model,
        "budget": BUDGET.model_dump(),
        "limits": LIMITS.model_dump(),
        "baselines": summarize(rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary["baselines"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
