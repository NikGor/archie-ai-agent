"""Stage 1 (command decision) + Stage 2 (tool execution) iteration loop.

Extracted from AgentFactory.arun()'s while-loop body so the orchestrator
stays a thin readable entry point. `make_command_call` is injected rather
than owned here so AgentFactory._make_command_call (and the tests that
patch it) keep working unchanged.
"""

import logging
from collections.abc import Awaitable, Callable
from archie_shared.chat.models import LllmTrace
from ..models.orchestration_sgr import DecisionResponse
from ..models.state_models import UserState
from ..models.tool_models import ToolResult
from ..models.ws_models import StatusCallback, StatusNotifier
from ..tools.tool_factory import ToolFactory
from ..utils.tool_executor import execute_tool_calls
from ..utils.trace_utils import StepTimer
from .context import ConversationContext
from .pipeline_metrics import PipelineMetrics


logger = logging.getLogger(__name__)

MakeCommandCall = Callable[..., Awaitable[tuple[DecisionResponse, LllmTrace | None]]]


def format_command_summary(command_history: list[dict], tool_count: int) -> str:
    parts = [
        f"Iteration {h['iteration']}: {h['action_type']} - {h['reasoning']}"
        for h in command_history
    ]
    return "\n\n".join(parts) + f"\n\nTotal tools executed: {tool_count}"


class CommandLoopResult:
    def __init__(
        self,
        tool_results: list[ToolResult],
        command_summary: str,
        ui_intents: list[str],
    ):
        self.tool_results = tool_results
        self.command_summary = command_summary
        self.ui_intents = ui_intents


class CommandLoop:
    """Runs Stage 1 (decide) -> Stage 2 (execute tools) until a non-function_call
    action or max_iterations is reached."""

    def __init__(
        self, tool_factory: ToolFactory, metrics: PipelineMetrics, max_iterations: int
    ):
        self.tool_factory = tool_factory
        self.metrics = metrics
        self.max_iterations = max_iterations

    async def run(
        self,
        make_command_call: MakeCommandCall,
        user_input: str,
        command_model: str,
        cmd_provider: str,
        user_state: UserState,
        response_format: str,
        ctx: ConversationContext,
        seed_results: list[ToolResult],
        notifier: StatusNotifier,
        on_status: StatusCallback,
    ) -> CommandLoopResult:
        tool_results = list(seed_results)
        command_history: list[dict] = []
        iteration = 0
        decision: DecisionResponse | None = None

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(
                f"agent_factory_003a: Command iteration \033[33m{iteration}\033[0m"
            )
            await notifier.emit(
                "command",
                "started",
                f"Analyzing request (iteration {iteration})",
                detail=(
                    "Analyzing request"
                    if iteration == 1
                    else f"Refining results (iteration {iteration})"
                ),
            )

            # STAGE 1: Command - Analyze request and decide action
            with StepTimer() as s1_timer:
                decision, s1_llm_trace = await make_command_call(
                    user_input=user_input,
                    model=command_model,
                    provider=cmd_provider,
                    user_state=user_state,
                    response_format=response_format,
                    ctx=ctx,
                    previous_results=tool_results if tool_results else None,
                )
            self.metrics.add_stage1(s1_timer.duration_ms, s1_llm_trace)
            tool_names = (
                [tc.tool_name for tc in decision.sgr.tool_calls]
                if decision.sgr.tool_calls
                else []
            )
            detail_msg = (
                ", ".join(tool_names) if tool_names else decision.sgr.action.type
            )
            await notifier.emit(
                "command",
                "completed",
                f"Action: {decision.sgr.action.type}",
                detail=detail_msg,
            )
            logger.info(
                f"agent_factory_004: Action type: \033[36m{decision.sgr.action.type}\033[0m"
            )
            logger.info(
                f"agent_factory_004a: SGR reasoning: \033[33m{decision.sgr.reasoning}\033[0m"
            )
            command_history.append(
                {
                    "iteration": iteration,
                    "action_type": decision.sgr.action.type,
                    "reasoning": decision.sgr.reasoning,
                }
            )
            if decision.sgr.action.type != "function_call":
                logger.info(
                    f"agent_factory_005a: Exiting command loop - action type: \033[36m{decision.sgr.action.type}\033[0m"
                )
                break
            if decision.sgr.tool_calls:
                for tc in decision.sgr.tool_calls:
                    logger.info(
                        f"agent_factory_006_reason: Tool \033[36m{tc.tool_name}\033[0m - reason: \033[33m{tc.reason}\033[0m"
                    )
                logger.info(
                    f"agent_factory_006: Executing \033[33m{len(decision.sgr.tool_calls)}\033[0m tools"
                )

                # STAGE 2: Tool execution
                with StepTimer() as s2_timer:
                    new_results = await execute_tool_calls(
                        tool_calls=decision.sgr.tool_calls,
                        tool_factory=self.tool_factory,
                        on_status=on_status,
                    )
                self.metrics.add_stage2(s2_timer.duration_ms)
                tool_results.extend(new_results)
                logger.info(
                    f"agent_factory_006a: Total tool results: \033[33m{len(tool_results)}\033[0m"
                )
            else:
                logger.warning(
                    "agent_factory_warning_001: function_call type but no tool_calls"
                )
                break

        if iteration >= self.max_iterations:
            logger.warning(
                f"agent_factory_warning_002: Reached max iterations ({self.max_iterations})"
            )

        command_summary = format_command_summary(command_history, len(tool_results))
        ui_intents = [str(i) for i in decision.sgr.intents] if decision else []
        return CommandLoopResult(
            tool_results=tool_results,
            command_summary=command_summary,
            ui_intents=ui_intents,
        )
