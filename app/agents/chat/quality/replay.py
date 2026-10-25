"""Replays one attempt of one chat eval case against the chat pipeline (ARCHIE-296)."""

import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from archie_shared.chat.models import PipelineTrace
from archie_shared.evals.models import (
    EvalAssertionFailure,
    EvalCaseInfo,
    EvalCaseResult,
)
from archie_shared.monitoring.models import MonitorMessage, MonitorRuleScore
from app.agents.chat.pipeline import ChatPipeline
from app.agents.chat.tool_access import chat_tool_names
from app.config import MAX_COMMAND_ITERATIONS, settings
from app.models.orchestration_sgr import DecisionResponse
from app.models.output_models import AgentResponse
from app.models.ws_models import StatusUpdate
from app.platform.llm.catalog import get_provider_for_model
from app.platform.llm.context import ConversationContext
from app.platform.tools.executor import execute_tool_calls
from app.platform.tools.models import ToolContext
from .assertions import check_decision, check_response, check_rules, find_ui_answer
from .judge import judge_rules
from .ui_judge import judge_response


async def _judge_case_rules(
    case: EvalCaseInfo, response: AgentResponse
) -> list[MonitorRuleScore]:
    """Grade the case's rules on its response with the monitoring judge (ARCHIE-213),
    so a prompt fix is verified by the same check that found the problem."""
    now = datetime.now(UTC).isoformat()
    messages = [
        MonitorMessage(
            message_id="eval-user",
            role="user",
            content_json=json.dumps({"content_format": "plain", "text": case.input}),
            created_at=now,
        ),
        MonitorMessage(
            message_id="eval-assistant",
            role="assistant",
            content_json=response.content.model_dump_json(),
            created_at=now,
        ),
    ]
    return await judge_rules(
        messages, 1, case.rules, settings.monitor_judge_model, case.response_format
    )


if TYPE_CHECKING:
    from app.platform.tools.models import ToolResult

logger = logging.getLogger(__name__)

StepCallback = Callable[[str], Awaitable[None]]

SKILL_LOADER_TOOL = "skill_loader_tool"


def _sum_tokens(trace: PipelineTrace | None) -> int | None:
    if trace is None:
        return None
    stages = (trace.command_call, trace.tool_execution, trace.create_output)
    return sum(
        stage.llm_trace.total_tokens for stage in stages if stage and stage.llm_trace
    )


def _status(failures: list[EvalAssertionFailure]) -> str:
    return "failed" if failures else "passed"


def _only_loads_skills(decision: DecisionResponse) -> bool:
    calls = decision.sgr.tool_calls
    return (
        decision.sgr.action.type == "function_call"
        and bool(calls)
        and all(call.tool_name == SKILL_LOADER_TOOL for call in calls)
    )


class ChatCaseReplay:
    """Replays chat eval cases in demo mode as a dedicated eval user (ARCHIE-296)."""

    def __init__(self, agent_factory: ChatPipeline | None = None) -> None:
        self.agent_factory = agent_factory or ChatPipeline()

    async def run(
        self,
        run_id: str,
        case: EvalCaseInfo,
        model: str,
        attempt: int,
        judge_model: str | None,
        on_step: StepCallback,
    ) -> EvalCaseResult:
        base = {
            "run_id": run_id,
            "case_id": case.id,
            "model": model,
            "attempt": attempt,
        }
        started = time.monotonic()
        try:
            if case.stage == "stage1":
                fields = await self._run_stage1(case, model, on_step)
            else:
                fields = await self._run_full(case, model, judge_model, on_step)
        except Exception as exc:
            logger.error(
                f"eval_runner_error_001: \033[36m{case.id}\033[0m/{model}#{attempt}: "
                f"\033[31m{type(exc).__name__}: {exc}\033[0m"
            )
            fields = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "total_ms": int((time.monotonic() - started) * 1000),
            }
        return EvalCaseResult(
            **base, **fields, finished_at=datetime.now(UTC).isoformat()
        )

    async def _run_stage1(
        self, case: EvalCaseInfo, model: str, on_step: StepCallback
    ) -> dict:
        await on_step("command")
        provider = get_provider_for_model(model)
        user_state = await self.agent_factory.state_service.get_user_state(
            user_name=settings.eval_user_name, demo_mode=True
        )
        ctx = ConversationContext().for_provider(provider)
        skill_results: list[ToolResult] = []
        earlier_tools: list[str] = []
        total_tokens = 0
        started = time.monotonic()
        # Loading a skill is a preparation step, as in CommandLoop: run it and
        # check the next decision. The other tools are never executed here.
        for _ in range(MAX_COMMAND_ITERATIONS):
            decision, trace = await self.agent_factory._make_command_call(
                user_input=case.input,
                model=model,
                provider=provider,
                user_state=user_state,
                response_format=case.response_format,
                ctx=ctx,
                previous_results=skill_results or None,
            )
            total_tokens += trace.total_tokens if trace else 0
            if not _only_loads_skills(decision):
                break
            await on_step("skill_loader")
            earlier_tools += [call.tool_name for call in decision.sgr.tool_calls]
            skill_results += await execute_tool_calls(
                [call.to_call() for call in decision.sgr.tool_calls],
                self.agent_factory.tool_registry,
                allowed=chat_tool_names(case.response_format),
                context=ToolContext(
                    user_timezone=user_state.user_timezone, demo_mode=True
                ),
            )
        elapsed = int((time.monotonic() - started) * 1000)
        failures = check_decision(case, decision, earlier_tools)
        return {
            "status": _status(failures),
            "failures": failures,
            "stage1_ms": elapsed,
            "total_ms": elapsed,
            "total_tokens": total_tokens or None,
            "output_json": decision.model_dump_json(),
        }

    async def _run_full(
        self,
        case: EvalCaseInfo,
        model: str,
        judge_model: str | None,
        on_step: StepCallback,
    ) -> dict:
        async def on_status(update: StatusUpdate) -> None:
            await on_step(update.step)

        response = await self.agent_factory.arun(
            messages=[{"role": "user", "content": case.input}],
            command_model=model,
            final_output_model=model,
            response_format=case.response_format,
            user_name=settings.eval_user_name,
            on_status=on_status,
            demo_mode=True,
        )
        trace = response.pipeline_trace
        total_tokens = _sum_tokens(trace)
        total_ms = trace.total_ms if trace else None
        failures = check_response(case, response, total_tokens, total_ms)
        if case.expect.rules_followed and response.content:
            await on_step("rules")
            failures += check_rules(case, await _judge_case_rules(case, response))
        judge = None
        if case.judge and judge_model:
            await on_step("judge")
            judge = await judge_response(case, response.content, judge_model)
        ui_answer = find_ui_answer(response)
        return {
            "status": _status(failures),
            "failures": failures,
            "stage1_ms": (
                trace.command_call.duration_ms if trace and trace.command_call else None
            ),
            "stage2_ms": (
                trace.tool_execution.duration_ms
                if trace and trace.tool_execution
                else None
            ),
            "stage3_ms": (
                trace.create_output.duration_ms
                if trace and trace.create_output
                else None
            ),
            "total_ms": total_ms,
            "total_tokens": total_tokens,
            "ui_items_count": len(ui_answer.items) if ui_answer else None,
            "judge": judge,
            "output_json": (
                response.content.model_dump_json() if response.content else None
            ),
        }
