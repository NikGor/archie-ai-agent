import logging
import time
from archie_shared.chat.models import LllmTrace
from ..backend.llm.registry import get_client
from ..backend.state_service import StateService
from ..backend.tool_result_store import ToolResultStore
from ..config import DEFAULT_MODEL, MAX_COMMAND_ITERATIONS
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
from ..tools.output import create_output
from ..tools.tool_factory import ToolFactory
from ..utils.llm_parser import parse_llm_response
from ..utils.provider_utils import get_provider_for_model
from ..utils.trace_utils import StepTimer
from .command_loop import CommandLoop
from .context import ConversationContext
from .pipeline_metrics import PipelineMetrics
from .prompt_builder import PromptBuilder


logger = logging.getLogger(__name__)


class AgentFactory:
    def __init__(
        self,
        prompt_builder: PromptBuilder | None = None,
        tool_factory: ToolFactory | None = None,
        state_service: StateService | None = None,
        demo_mode: bool = False,
    ):
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.tool_factory = tool_factory or ToolFactory(demo_mode=demo_mode)
        self.state_service = state_service or StateService()
        self.tool_result_store = ToolResultStore()
        self.demo_mode = demo_mode
        logger.info(
            f"agent_factory_001: Initialized AgentFactory, demo_mode: \033[35m{demo_mode}\033[0m"
        )

    async def _make_command_call(
        self,
        user_input: str,
        model: str,
        provider: str,
        user_state: UserState,
        response_format: str,
        ctx: ConversationContext,
        previous_results: list[ToolResult] | None = None,
    ) -> tuple[DecisionResponse, LllmTrace | None]:
        """
        Stage 1: Analyze request and decide action using cmd_prompt.

        Returns (DecisionResponse, LllmTrace) with routing, action type, tool calls, and token usage.
        Can be called multiple times in a loop with previous_results from prior iterations.
        `ctx` must already be gated for `provider` (see ConversationContext.for_provider).
        """
        logger.info("=== Stage 1: Command Decision ===")
        client = get_client(provider)
        tools = self.tool_factory.get_tool_schemas(model, response_format)
        messages = self.prompt_builder.build_command_messages(
            user_input=user_input,
            state=user_state.model_dump(),
            tools=tools,
            provider=provider,
            previous_results=previous_results,
            chat_history=ctx.chat_history,
        )
        logger.info(f"agent_factory_008: Making command call with {provider}")
        raw_response = await client.create_completion(
            messages=messages,
            model=model,
            response_format=DecisionResponse,
            previous_response_id=ctx.previous_response_id,
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
        user_state: UserState,
        arun_start: float,
        ctx: ConversationContext,
        no_image: bool = False,
        on_stream: StreamCallback = None,
        on_stream_event: StreamEventCallback = None,
    ) -> AgentResponse:
        """Dashboard/Widget flow: skip command loop, go directly to Stage 3.

        `ctx` must already be gated for the output provider.
        """
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
                previous_response_id=ctx.previous_response_id,
                chat_history=ctx.chat_history,
                no_image=no_image,
                on_stream=on_stream,
                on_stream_event=on_stream_event,
            )
        total_ms = int((time.monotonic() - arun_start) * 1000)
        final_response.pipeline_trace = PipelineMetrics().build_trace(
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

    async def arun(
        self,
        messages: list[dict[str, str]],
        command_model: str = DEFAULT_MODEL,
        final_output_model: str = DEFAULT_MODEL,
        response_format: str = "plain",
        previous_response_id: str | None = None,
        chat_history: str | None = None,
        user_name: str | None = None,
        conversation_id: str | None = None,
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
        conversation_ctx = ConversationContext(
            previous_response_id=previous_response_id, chat_history=chat_history
        )
        logger.info(
            f"agent_factory_001b: Command: \033[34m{cmd_provider}\033[0m/\033[36m{command_model}\033[0m | "
            f"Output: \033[34m{output_provider}\033[0m/\033[36m{final_output_model}\033[0m"
        )
        if user_name:
            logger.info(f"agent_factory_001c: Using user_name: \033[35m{user_name}\033[0m")
        user_state = await self.state_service.get_user_state(
            user_name=user_name, demo_mode=self.demo_mode
        )
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
                user_state=user_state,
                arun_start=arun_start,
                ctx=conversation_ctx.for_provider(output_provider),
                no_image=no_image,
                on_stream=on_stream,
                on_stream_event=on_stream_event,
            )

        # Seed with results persisted from earlier requests in this conversation
        # so Stage 1, Stage 3 and the summary all see prior tool context.
        persisted_results = await self.tool_result_store.load(
            conversation_id, user_name
        )
        metrics = PipelineMetrics()
        command_loop = CommandLoop(self.tool_factory, metrics, MAX_COMMAND_ITERATIONS)
        loop_result = await command_loop.run(
            make_command_call=self._make_command_call,
            user_input=user_input,
            command_model=command_model,
            cmd_provider=cmd_provider,
            user_state=user_state,
            response_format=response_format,
            ctx=conversation_ctx.for_provider(cmd_provider),
            seed_results=persisted_results,
            notifier=notifier,
            on_status=on_status,
        )
        await self.tool_result_store.save(
            conversation_id, user_name, loop_result.tool_results
        )
        logger.info(
            f"agent_factory_007: Creating final output, intents: \033[35m{loop_result.ui_intents}\033[0m"
        )
        await notifier.emit(
            "output",
            "started",
            f"Generating {response_format} response with {final_output_model}",
            detail="Generating response",
        )
        output_ctx = conversation_ctx.for_provider(output_provider)

        # STAGE 3: Final output generation
        with StepTimer() as stage3_timer:
            final_response = await create_output(
                user_input=user_input,
                command_summary=loop_result.command_summary,
                tool_results=loop_result.tool_results if loop_result.tool_results else None,
                response_format=response_format,
                model=final_output_model,
                state=user_state.model_dump(),
                previous_response_id=output_ctx.previous_response_id,
                chat_history=output_ctx.chat_history,
                intents=loop_result.ui_intents,
                no_image=no_image,
                on_stream=on_stream,
                on_stream_event=on_stream_event,
            )
        await notifier.emit(
            "output", "completed", "Response ready", detail="Response ready"
        )
        total_ms = int((time.monotonic() - arun_start) * 1000)
        final_response.pipeline_trace = metrics.build_trace(
            total_ms=total_ms,
            stage3_duration_ms=stage3_timer.duration_ms,
            stage3_llm_trace=final_response.llm_trace,
            stage3_ttft_ms=final_response.ttft_ms,
        )
        logger.info(
            f"agent_factory_010: Pipeline trace: command_call=\033[33m{metrics.stage1_duration_ms}\033[0mms, "
            f"tool_execution=\033[33m{metrics.stage2_duration_ms}\033[0mms, "
            f"create_output=\033[33m{stage3_timer.duration_ms}\033[0mms, "
            f"ttft=\033[33m{final_response.pipeline_trace.ttft_ms}\033[0mms, "
            f"total=\033[33m{total_ms}\033[0mms"
        )
        logger.info("=== AgentFactory: Response Created ===")
        return final_response
