import logging
import time

from archie_shared.chat.models import LllmTrace

from ..backend.gemini_client import GeminiClient
from ..backend.openai_client import OpenAIClient
from ..backend.openrouter_client import OpenRouterClient
from ..backend.state_service import StateService
from ..config import MAX_COMMAND_ITERATIONS
from ..models.orchestration_sgr import DecisionResponse
from ..models.output_models import AgentResponse
from ..models.state_models import UserState
from ..models.tool_models import ToolResult
from ..models.ws_models import (
    StatusCallback,
    StatusNotifier,
    StreamCallback,
    StreamEventCallback,
)
from ..tools.create_output_tool import create_output
from ..tools.tool_factory import ToolFactory
from ..utils.llm_parser import parse_llm_response
from ..utils.provider_utils import get_provider_for_model
from ..utils.tool_executor import execute_tool_calls
from ..utils.trace_utils import StepTimer, build_pipeline_trace
from .prompt_builder import PromptBuilder


logger = logging.getLogger(__name__)


def _format_command_summary(
    command_history: list[dict], tool_count: int
) -> str:
    parts = [
        f"Iteration {h['iteration']}: {h['action_type']} - {h['reasoning']}"
        for h in command_history
    ]
    return "\n\n".join(parts) + f"\n\nTotal tools executed: {tool_count}"


class AgentFactory:
    def __init__(
        self,
        prompt_builder: PromptBuilder | None = None,
        tool_factory: ToolFactory | None = None,
        state_service: StateService | None = None,
        demo_mode: bool = False,
        no_image: bool = False,
    ):
        self.openai_client = OpenAIClient()
        self.openrouter_client = OpenRouterClient()
        self.gemini_client = GeminiClient()  # Fallback
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.tool_factory = tool_factory or ToolFactory(
            demo_mode=demo_mode, no_image=no_image
        )
        self.state_service = state_service or StateService()
        self.clients: dict[str, OpenAIClient | OpenRouterClient | GeminiClient] = {
            "openai": self.openai_client,
            "openrouter": self.openrouter_client,
            "gemini": self.gemini_client,  # Fallback
        }
        logger.info(
            f"agent_factory_001: Initialized AgentFactory, demo_mode: \033[35m{demo_mode}\033[0m, no_image: \033[35m{no_image}\033[0m"
        )

    async def _make_command_call(
        self,
        user_input: str,
        model: str,
        provider: str,
        user_state: UserState,
        response_format: str,
        previous_results: list[ToolResult] | None = None,
        previous_response_id: str | None = None,
        chat_history: str | None = None,
    ) -> tuple[DecisionResponse, LllmTrace | None]:
        """
        Stage 1: Analyze request and decide action using cmd_prompt.

        Returns (DecisionResponse, LllmTrace) with routing, action type, tool calls, and token usage.
        Can be called multiple times in a loop with previous_results from prior iterations.
        """
        logger.info("=== Stage 1: Command Decision ===")
        client = self.clients[provider]
        tools = self.tool_factory.get_tool_schemas(model, response_format)
        messages = self.prompt_builder.build_command_messages(
            user_input=user_input,
            state=user_state.model_dump(),
            tools=tools,
            provider=provider,
            previous_results=previous_results,
            chat_history=chat_history,
        )
        logger.info(f"agent_factory_008: Making command call with {provider}")
        raw_response = await client.create_completion(
            messages=messages,
            model=model,
            response_format=DecisionResponse,
            previous_response_id=previous_response_id if provider == "openai" else None,
        )
        parsed = parse_llm_response(
            raw_response=raw_response,
            provider=provider,
            expected_type=DecisionResponse,
        )
        logger.info(
            f"agent_factory_009: Decision made - Action: \033[36m{parsed.parsed_content.sgr.action.type}\033[0m"
        )

        return parsed.parsed_content, parsed.llm_trace

    async def _run_direct_output(
        self,
        user_input: str,
        response_format: str,
        final_output_model: str,
        output_provider: str,
        user_state: UserState,
        arun_start: float,
        previous_response_id: str | None = None,
        chat_history: str | None = None,
        no_image: bool = False,
        on_stream: StreamCallback = None,
        on_stream_event: StreamEventCallback = None,
    ) -> AgentResponse:
        """Dashboard/Widget flow: skip command loop, go directly to Stage 3."""
        logger.info(
            f"agent_factory_003b: {response_format} format - skipping command loop"
        )
        with StepTimer() as stage3_timer:
            final_response = await create_output(
                user_input=user_input,
                command_summary="Dashboard request - direct output",
                tool_results=None,
                response_format=response_format,
                model=final_output_model,
                state=user_state.model_dump(),
                previous_response_id=(
                    previous_response_id if output_provider == "openai" else None
                ),
                chat_history=chat_history if output_provider != "openai" else None,
                no_image=no_image,
                on_stream=on_stream,
                on_stream_event=on_stream_event,
            )
        total_ms = int((time.monotonic() - arun_start) * 1000)
        final_response.pipeline_trace = build_pipeline_trace(
            total_ms=total_ms,
            stage3_duration_ms=stage3_timer.duration_ms,
            stage3_llm_trace=final_response.llm_trace,
            stage3_ttft_ms=final_response.ttft_ms,
        )
        logger.info(
            f"agent_factory_010: Pipeline trace (direct): output=\033[33m{stage3_timer.duration_ms}\033[0mms, "
            f"ttft=\033[33m{final_response.ttft_ms}\033[0mms, "
            f"total=\033[33m{total_ms}\033[0mms"
        )
        logger.info("=== AgentFactory: Response Created ===")
        return final_response

    async def arun(  # noqa: PLR0912
        self,
        messages: list[dict[str, str]],
        command_model: str = "gpt-4.1",
        final_output_model: str = "gpt-4.1",
        response_format: str = "plain",
        previous_response_id: str | None = None,
        chat_history: str | None = None,
        user_name: str | None = None,
        no_image: bool = False,
        on_status: StatusCallback = None,
        on_stream: StreamCallback = None,
        on_stream_event: StreamEventCallback = None,
    ) -> AgentResponse:
        """
        Main entry point: Create an agent response through 3-stage flow.

        Stage 1: Command - Analyze request and decide action (uses command_model)
        Stage 2: Tool execution (if needed)
        Stage 3: Final response generation (uses final_output_model)
        """
        logger.info("=== AgentFactory: Creating Agent Response ===")
        arun_start = time.monotonic()
        cmd_provider = get_provider_for_model(command_model)
        output_provider = get_provider_for_model(final_output_model)
        logger.info(
            f"agent_factory_001b: Command: \033[34m{cmd_provider}\033[0m/\033[36m{command_model}\033[0m | "
            f"Output: \033[34m{output_provider}\033[0m/\033[36m{final_output_model}\033[0m"
        )
        if user_name:
            self.state_service.user_name = user_name
            logger.info(
                f"agent_factory_001c: Set user_name: \033[35m{user_name}\033[0m"
            )
        user_state = await self.state_service.get_user_state()
        persona_key = user_state.persona
        logger.info(f"agent_factory_002: Persona: \033[35m{persona_key}\033[0m")
        logger.info(f"agent_factory_003: Format: \033[36m{response_format}\033[0m")
        notifier = StatusNotifier(on_status)
        await notifier.emit(
            "init",
            "completed",
            f"Persona: {persona_key}, format: {response_format}, model: {command_model}",
        )
        user_input = messages[-1]["content"] if messages else ""

        # Dashboard/Widget formats: skip command loop, go directly to final output
        if response_format in ["dashboard", "widget"]:
            return await self._run_direct_output(
                user_input=user_input,
                response_format=response_format,
                final_output_model=final_output_model,
                output_provider=output_provider,
                user_state=user_state,
                arun_start=arun_start,
                previous_response_id=previous_response_id,
                chat_history=chat_history,
                no_image=no_image,
                on_stream=on_stream,
                on_stream_event=on_stream_event,
            )

        tool_results = []
        command_history = []
        stage1_duration_ms = 0
        stage1_llm_traces: list[LllmTrace] = []
        stage2_duration_ms = 0
        iteration = 0
        decision: DecisionResponse | None = None
        while iteration < MAX_COMMAND_ITERATIONS:
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
                decision, s1_llm_trace = await self._make_command_call(
                    user_input=user_input,
                    model=command_model,
                    provider=cmd_provider,
                    user_state=user_state,
                    response_format=response_format,
                    previous_results=tool_results if tool_results else None,
                    previous_response_id=previous_response_id,
                    chat_history=chat_history,
                )
            stage1_duration_ms += s1_timer.duration_ms
            if s1_llm_trace:
                stage1_llm_traces.append(s1_llm_trace)
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
                stage2_duration_ms += s2_timer.duration_ms
                tool_results.extend(new_results)
                logger.info(
                    f"agent_factory_006a: Total tool results: \033[33m{len(tool_results)}\033[0m"
                )
            else:
                logger.warning(
                    "agent_factory_warning_001: function_call type but no tool_calls"
                )
                break
        if iteration >= MAX_COMMAND_ITERATIONS:
            logger.warning(
                f"agent_factory_warning_002: Reached max iterations ({MAX_COMMAND_ITERATIONS})"
            )
        command_summary = _format_command_summary(command_history, len(tool_results))
        ui_intents: list[str] = (
            [str(i) for i in decision.sgr.intents] if decision else []
        )
        logger.info(
            f"agent_factory_007: Creating final output, intents: \033[35m{ui_intents}\033[0m"
        )
        await notifier.emit(
            "output",
            "started",
            f"Generating {response_format} response with {final_output_model}",
            detail="Generating response",
        )

        # STAGE 3: Final output generation
        with StepTimer() as stage3_timer:
            final_response = await create_output(
                user_input=user_input,
                command_summary=command_summary,
                tool_results=tool_results if tool_results else None,
                response_format=response_format,
                model=final_output_model,
                state=user_state.model_dump(),
                previous_response_id=(
                    previous_response_id if output_provider == "openai" else None
                ),
                chat_history=chat_history if output_provider != "openai" else None,
                intents=ui_intents,
                no_image=no_image,
                on_stream=on_stream,
                on_stream_event=on_stream_event,
            )
        await notifier.emit("output", "completed", "Response ready", detail="Response ready")
        total_ms = int((time.monotonic() - arun_start) * 1000)
        final_response.pipeline_trace = build_pipeline_trace(
            total_ms=total_ms,
            stage3_duration_ms=stage3_timer.duration_ms,
            stage3_llm_trace=final_response.llm_trace,
            stage3_ttft_ms=final_response.ttft_ms,
            stage1_duration_ms=stage1_duration_ms,
            stage1_llm_traces=stage1_llm_traces,
            stage2_duration_ms=stage2_duration_ms,
        )
        logger.info(
            f"agent_factory_010: Pipeline trace: command_call=\033[33m{stage1_duration_ms}\033[0mms, "
            f"tool_execution=\033[33m{stage2_duration_ms}\033[0mms, "
            f"create_output=\033[33m{stage3_timer.duration_ms}\033[0mms, "
            f"ttft=\033[33m{final_response.pipeline_trace.ttft_ms}\033[0mms, "
            f"total=\033[33m{total_ms}\033[0mms"
        )
        logger.info("=== AgentFactory: Response Created ===")
        return final_response
