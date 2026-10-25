"""Replays one swarm eval case (ARCHIE-297): one agent turn (swarm_turn) or one
spontaneous check (swarm_watch) with the case's fixed chat, memory and tool
results. Tools never run for real; the case gives their output."""

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from archie_shared.evals.models import (
    EvalAssertionFailure,
    EvalCaseInfo,
    EvalCaseResult,
    SwarmEvalInput,
)
from archie_shared.monitoring.models import MonitorRuleScore, SwarmMonitorRecord
from archie_shared.swarm.models import SwarmActionInfo, SwarmMessage, SwarmUsage
from app.agents.swarm.agents import SWARM_AGENTS
from app.agents.swarm.models import SwarmAgentProfile, SwarmToolCall
from app.agents.swarm.runner import SwarmAgentRunner
from app.config import settings
from app.platform.tools.models import ToolContext, ToolResult
from .judge import judge_record


logger = logging.getLogger(__name__)

StepCallback = Callable[[str], Awaitable[None]]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _failure(key: str, expected: object, actual: object) -> EvalAssertionFailure:
    return EvalAssertionFailure(
        key=key,
        expected=json.dumps(expected, ensure_ascii=False),
        actual=json.dumps(actual, ensure_ascii=False),
    )


def history_messages(lines: list[str]) -> list[SwarmMessage]:
    """'@weather: Дождь после 17:00' / 'user: ...' -> swarm messages, oldest first."""
    messages = []
    for index, line in enumerate(lines):
        sender, _, text = line.partition(":")
        sender = sender.strip().lstrip("@") or "user"
        messages.append(
            SwarmMessage(
                message_id=f"case-{index}",
                created_at=_now(),
                sender=sender,
                kind="user" if sender == "user" else "report",
                text=text.strip() or line,
            )
        )
    return messages


class _CaseRoster:
    """The case model for every agent; the eval user owns the swarm."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def get_model(self, _profile: SwarmAgentProfile) -> str:
        return self.model

    async def get_owner(self) -> str:
        return settings.eval_user_name


class SwarmCaseReplay:
    """Runs swarm cases on SwarmAgentRunner with stubbed tools, memory and clock."""

    def _runner(self, swarm: SwarmEvalInput, model: str, called: list[str]) -> SwarmAgentRunner:
        runner = SwarmAgentRunner(roster=_CaseRoster(model))  # type: ignore[arg-type]

        async def memory_lines(*_args: Any) -> list[str]:
            return list(swarm.memory)

        async def run_calls(
            _profile: SwarmAgentProfile, calls: list[SwarmToolCall], _context: ToolContext
        ) -> list[ToolResult]:
            results = []
            for call in calls:
                called.append(call.tool_name)
                output = swarm.tool_results.get(call.tool_name)
                results.append(
                    ToolResult(
                        tool_name=call.tool_name,
                        output=json.loads(output) if output else {"error": "no data in the case"},
                        success=output is not None,
                        error=None if output else "no data in the case",
                    )
                )
            return results

        runner.prompts.build_memory_lines = memory_lines  # type: ignore[method-assign]
        runner._run_calls = run_calls  # type: ignore[method-assign]
        if swarm.now:
            now = datetime.fromisoformat(swarm.now)
            runner.prompts.datetime_note = lambda _state: (  # type: ignore[method-assign]
                f"Current date/time: {now:%A, %d.%m.%Y, %H:%M}"
            )
        return runner

    async def run(
        self,
        run_id: str,
        case: EvalCaseInfo,
        model: str,
        attempt: int,
        judge_model: str | None,
        on_step: StepCallback,
    ) -> EvalCaseResult:
        started = time.monotonic()
        base = {"run_id": run_id, "case_id": case.id, "model": model, "attempt": attempt}
        try:
            fields = await self._run(case, model, judge_model, on_step)
        except Exception as exc:
            logger.error(
                f"swarm_replay_error_001: \033[36m{case.id}\033[0m/{model}#{attempt}: "
                f"\033[31m{type(exc).__name__}: {exc}\033[0m"
            )
            fields = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        fields.setdefault("total_ms", int((time.monotonic() - started) * 1000))
        return EvalCaseResult(**base, **fields, finished_at=_now())

    async def _run(
        self, case: EvalCaseInfo, model: str, judge_model: str | None, on_step: StepCallback
    ) -> dict:
        swarm = case.swarm
        if swarm is None or swarm.agent_id not in SWARM_AGENTS:
            raise ValueError("a swarm case needs swarm.agent_id of a known agent")
        profile = SWARM_AGENTS[swarm.agent_id]
        called: list[str] = []
        runner = self._runner(swarm, model, called)
        history = history_messages(swarm.history)
        started = time.monotonic()
        await on_step(case.stage)
        trigger: SwarmMessage | None = None
        if case.stage == "swarm_watch":
            if swarm.watch_task is None:
                raise ValueError("a swarm_watch case needs swarm.watch_task")
            task = runner.prompts.render_task(
                "swarm_watch", task=swarm.watch_task, teams=[], language="ru"
            )
            draft = await runner.watch(profile, swarm.watch_task, history, task)
            reply, importance = draft.outcome.reply.strip(), draft.outcome.importance
            actions: list[SwarmActionInfo] = []
            usage: SwarmUsage = draft.usage
            kind = "alert" if reply else "check"
        else:
            trigger = SwarmMessage(
                message_id="case-trigger", created_at=_now(), sender="user", kind="user", text=case.input
            )
            turn = await runner.run(profile, trigger, history)
            reply = turn.text if turn else ""
            importance = None
            actions = turn.actions if turn else []
            usage = turn.usage if turn else SwarmUsage(model=model)
            kind = "action" if actions else "report"
        elapsed = int((time.monotonic() - started) * 1000)
        failures = self._check(case, reply, importance, called)
        message = SwarmMessage(
            message_id=f"case-{uuid.uuid4().hex[:8]}",
            created_at=_now(),
            sender=profile.agent_id,
            kind=kind,
            text=reply or "(silent)",
            importance=importance if reply else None,
            actions=actions,
        )
        if case.expect.rules_followed and case.rules:
            await on_step("judge")
            record = SwarmMonitorRecord(message=message, trigger=trigger, history=history)
            evaluation = await judge_record(
                record, "eval", judge_model or settings.monitor_judge_model, case.rules
            )
            failures += self._check_rules(case, evaluation.rules)
        return {
            "status": "failed" if failures else "passed",
            "failures": failures,
            "stage1_ms": elapsed,
            "total_ms": elapsed,
            "total_tokens": usage.input_tokens + usage.output_tokens,
            "output_json": json.dumps(
                {"reply": reply, "importance": importance, "tools_called": called,
                 "actions": [a.model_dump() for a in actions]},
                ensure_ascii=False,
            ),
        }

    @staticmethod
    def _check(
        case: EvalCaseInfo, reply: str, importance: str | None, called: list[str]
    ) -> list[EvalAssertionFailure]:
        expect = case.expect
        failures = []
        if expect.tools_called is not None and not set(expect.tools_called) <= set(called):
            failures.append(_failure("tools_called", expect.tools_called, called))
        if expect.reply_empty is not None and expect.reply_empty != (not reply):
            failures.append(_failure("reply_empty", expect.reply_empty, reply))
        if expect.alert is not None and expect.alert != bool(reply):
            failures.append(_failure("alert", expect.alert, reply))
        if expect.importance is not None and reply and importance not in expect.importance:
            failures.append(_failure("importance", expect.importance, importance))
        if expect.reply_contains and expect.reply_contains.casefold() not in reply.casefold():
            failures.append(_failure("reply_contains", expect.reply_contains, reply))
        return failures

    @staticmethod
    def _check_rules(
        case: EvalCaseInfo, scores: list[MonitorRuleScore]
    ) -> list[EvalAssertionFailure]:
        """Like chat: a protected rule graded red or not applicable is broken."""
        graded = {f"{s.spec}:{s.rule_id}": s.grade for s in scores}
        broken = {
            key: graded.get(key) or "not_applicable"
            for key in case.rules
            if graded.get(key) in ("red", None)
        }
        return [_failure("rules_followed", "no red rule", broken)] if broken else []
