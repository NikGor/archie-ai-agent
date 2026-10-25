"""One swarm agent turn: decide -> run allowed tools (up to swarm_max_tool_rounds) -> reply.

`ask` is a single structured call without tools, used by council steps.

Reuses the platform: LLM clients, parser, tool registry and
execute_tool_calls. Skips the heavy cmd_prompt / create_output stages to keep
every turn small.
"""

import json
import logging
from typing import Any
from archie_shared.chat.models import LllmTrace
from archie_shared.swarm.models import (
    SwarmActionInfo,
    SwarmMessage,
    SwarmUsage,
    SwarmWatchTask,
)
from pydantic import BaseModel, Field
from app.agents.swarm import (
    tools as _swarm_tools,  # noqa: F401  registers the swarm tool group
)
from app.agents.swarm.agents import SwarmRoster
from app.agents.swarm.memory import SwarmMemory
from app.agents.swarm.messages import SwarmPromptBuilder, format_chat_line
from app.agents.swarm.models import (
    SwarmAgentProfile,
    SwarmReply,
    SwarmToolCall,
    SwarmTurn,
    SwarmWatchReply,
    SwarmWatchTurn,
)
from app.agents.swarm.policy import block_reason, check_tool_call, is_read_only
from app.backend.state_service import StateService
from app.config import settings
from app.models.state_models import UserState
from app.platform.llm.catalog import get_provider_for_model
from app.platform.llm.registry import get_client, parse_llm_response
from app.platform.runtime.budget import Budget, BudgetMeter, daily_spend
from app.platform.runtime.tool_loop import run_tool_loop
from app.platform.tools.executor import execute_tool_calls
from app.platform.tools.models import ToolCall, ToolContext, ToolResult
from app.platform.tools.registry import ToolRegistry
from app.tools.catalog import tool_registry as default_tool_registry


logger = logging.getLogger(__name__)

# Daily LLM spend of every swarm turn and check counts under this kind
DAILY_SPEND_KIND = "swarm"

FINAL_REPLY_TASK = (
    "Write your chat reply now. You must reply: say what you did, "
    "or say that you could not do the task and why."
)
WATCH_FINAL_TASK = (
    "Write the outcome of the check now. Write an alert only if something is "
    "new for the user. Else leave the reply empty."
)


class SwarmDraft(BaseModel):
    """Result of an agent turn, ready to post to the chat."""

    text: str = Field(description="Reply text")
    actions: list[SwarmActionInfo] = Field(description="Side-effect tool calls made")
    usage: SwarmUsage = Field(description="Summed LLM usage of the turn")


class SwarmWatchDraft(BaseModel):
    """Result of a spontaneous check."""

    outcome: SwarmWatchReply = Field(description="Alert, importance and notes")
    usage: SwarmUsage = Field(description="Summed LLM usage of the check")


class SwarmAgentRunner:
    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        state_service: StateService | None = None,
        roster: SwarmRoster | None = None,
        memory: SwarmMemory | None = None,
    ) -> None:
        self.tool_registry = tool_registry or default_tool_registry
        self.state_service = state_service or StateService()
        self.roster = roster or SwarmRoster()
        self.memory = memory or SwarmMemory()
        self.prompts = SwarmPromptBuilder(self.tool_registry, self.memory)

    async def get_user_state(self) -> UserState:
        """State of the swarm owner; settings.swarm_user_name wins if set."""
        user_name = settings.swarm_user_name or await self.roster.get_owner()
        return await self.state_service.get_user_state(user_name=user_name)

    async def ask(
        self,
        profile: SwarmAgentProfile,
        trigger: SwarmMessage,
        history: list[SwarmMessage],
        task: str,
        schema: type[BaseModel],
    ) -> tuple[Any, SwarmUsage]:
        """One structured call without tools, for council steps."""
        model = await self.roster.get_model(profile)
        provider = get_provider_for_model(model)
        messages = await self.prompts.build_messages(
            await self.get_user_state(),
            profile,
            model,
            trigger.message_id,
            history,
            task,
        )
        traces: list[LllmTrace] = []
        content = await self._call(
            get_client(provider), provider, model, messages, schema, traces
        )
        return content, self._sum_usage(model, traces)

    async def run(
        self,
        profile: SwarmAgentProfile,
        trigger: SwarmMessage,
        history: list[SwarmMessage],
    ) -> SwarmDraft | None:
        model = await self.roster.get_model(profile)
        state = await self.get_user_state()
        messages = await self.prompts.build_messages(
            state,
            profile,
            model,
            trigger.message_id,
            history,
            f"React to this message:\n{format_chat_line(trigger)}",
        )
        turn, actions, traces = await self._loop(
            profile,
            model,
            messages,
            ToolContext(user_timezone=state.user_timezone),
            SwarmTurn,
            SwarmReply,
        )
        reply = turn.reply.strip()
        if not reply and not actions:
            logger.info(
                f"swarm_runner_002: \033[36m{profile.agent_id}\033[0m stays silent"
            )
            return None
        if not reply:
            reply = "; ".join(self._describe_action(a) for a in actions)
        return SwarmDraft(
            text=reply, actions=actions, usage=self._sum_usage(model, traces)
        )

    async def watch(
        self,
        profile: SwarmAgentProfile,
        watch_task: SwarmWatchTask,
        history: list[SwarmMessage],
        task: str,
    ) -> SwarmWatchDraft:
        """One spontaneous check: read fresh data, compare with today's notes,
        return an alert or an empty reply (ARCHIE-238)."""
        model = await self.roster.get_model(profile)
        state = await self.get_user_state()
        messages = await self.prompts.build_messages(
            state, profile, model, None, history, task, watch_task
        )
        turn, _, traces = await self._loop(
            profile,
            model,
            messages,
            ToolContext(user_timezone=state.user_timezone),
            SwarmWatchTurn,
            SwarmWatchReply,
            WATCH_FINAL_TASK,
        )
        return SwarmWatchDraft(
            outcome=SwarmWatchReply.model_validate(turn.model_dump()),
            usage=self._sum_usage(model, traces),
        )

    def _budget(self) -> Budget:
        return Budget(
            max_tool_rounds=settings.swarm_max_tool_rounds,
            max_cost_usd=settings.swarm_turn_max_cost_usd,
            max_seconds=settings.swarm_turn_max_seconds,
        )

    async def _loop(
        self,
        profile: SwarmAgentProfile,
        model: str,
        messages: list[dict[str, Any]],
        context: ToolContext,
        turn_schema: type[SwarmTurn],
        reply_schema: type[SwarmReply],
        final_task: str = FINAL_REPLY_TASK,
    ) -> tuple[SwarmTurn | SwarmReply, list[SwarmActionInfo], list[LllmTrace]]:
        """Tool rounds until the agent replies, inside the turn budget (ARCHIE-298).
        Several rounds let an agent read the state, then change it."""
        provider = get_provider_for_model(model)
        client = get_client(provider)
        traces: list[LllmTrace] = []
        actions: list[SwarmActionInfo] = []
        meter = BudgetMeter(self._budget())

        async def call(schema: type[BaseModel]) -> BaseModel:
            content = await self._call(client, provider, model, messages, schema, traces)
            trace = traces[-1] if traces else None
            daily_spend.add(DAILY_SPEND_KIND, trace.total_cost if trace else 0.0)
            meter.add(trace)
            return content

        async def run_tools(turn: BaseModel) -> tuple[str, bool]:
            assert isinstance(turn, SwarmTurn)
            logger.info(
                f"swarm_runner_001: \033[36m{profile.agent_id}\033[0m round {meter.rounds} tools="
                f"{[(tc.tool_name, {p.name: p.value for p in tc.arguments}) for tc in turn.tool_calls]} "
                f"reasoning: \033[33m{turn.reasoning}\033[0m"
            )
            results, new_actions = await self._execute(profile, turn.tool_calls, context)
            actions.extend(new_actions)
            text = self._format_results(
                results,
                new_actions,
                profile.tool_result_max_chars or settings.swarm_tool_result_max_chars,
            )
            # After a successful side effect only the reply is left: no repeated actions.
            return text, any(a.status == "success" for a in new_actions)

        turn = await run_tool_loop(
            call,
            lambda turn: turn.tool_calls if isinstance(turn, SwarmTurn) else [],
            run_tools,
            messages,
            turn_schema,
            reply_schema,
            final_task,
            meter,
        )
        return turn, actions, traces  # type: ignore[return-value]

    async def _call(
        self,
        client: Any,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        traces: list[LllmTrace],
    ) -> Any:
        raw = await client.create_completion(
            messages=messages, model=model, response_format=schema
        )
        parsed = parse_llm_response(
            raw_response=raw, provider=provider, expected_type=schema
        )
        traces.append(parsed.llm_trace)
        return parsed.parsed_content

    async def _execute(
        self,
        profile: SwarmAgentProfile,
        tool_calls: list[SwarmToolCall],
        context: ToolContext,
    ) -> tuple[list[ToolResult], list[SwarmActionInfo]]:
        allowed: list[SwarmToolCall] = []
        actions: list[SwarmActionInfo] = []
        for call in tool_calls:
            if check_tool_call(profile, call) == "allow":
                allowed.append(call)
            else:
                actions.append(
                    self._action_info(call, "blocked", block_reason(profile, call))
                )
        reads = [c for c in allowed if is_read_only(c)]
        writes = [c for c in allowed if not is_read_only(c)]
        # Dry run: calls with side effects run in demo mode, reads stay real
        write_context = context.model_copy(update={"demo_mode": settings.swarm_dry_run})
        results = await self._run_calls(
            profile, reads, context
        ) + await self._run_calls(profile, writes, write_context)
        allowed = reads + writes
        for call, result in zip(allowed, results, strict=True):
            if not is_read_only(call):
                actions.append(
                    self._action_info(
                        call, "success" if result.success else "error", result.error
                    )
                )
        return results, actions

    async def _run_calls(
        self,
        profile: SwarmAgentProfile,
        calls: list[SwarmToolCall],
        context: ToolContext,
    ) -> list[ToolResult]:
        if not calls:
            return []
        return await execute_tool_calls(
            tool_calls=[
                ToolCall(
                    tool_name=call.tool_name,
                    arguments=call.arguments,
                    reason=f"swarm agent {profile.agent_id}",
                )
                for call in calls
            ],
            registry=self.tool_registry,
            allowed=profile.tools,
            context=context,
        )

    @staticmethod
    def _action_info(
        call: SwarmToolCall, status: str, summary: str | None
    ) -> SwarmActionInfo:
        return SwarmActionInfo(
            tool_name=call.tool_name,
            arguments_json=json.dumps(
                {p.name: p.value for p in call.arguments}, ensure_ascii=False
            ),
            status=status,
            summary=summary,
        )

    @staticmethod
    def _describe_action(action: SwarmActionInfo) -> str:
        """Fallback reply line: the action and its argument values (e.g. spoken text)."""
        values = ", ".join(str(v) for v in json.loads(action.arguments_json).values())
        return f"{action.tool_name} ({action.status}): {values}"

    @staticmethod
    def _format_results(
        results: list[ToolResult], actions: list[SwarmActionInfo], limit: int
    ) -> str:
        lines = ["Tool results:"]
        for result in results:
            output = json.dumps(result.output, ensure_ascii=False, default=str)[:limit]
            status = "ok" if result.success else f"error: {result.error}"
            lines.append(f"- {result.tool_name} ({status}): {output}")
        lines.extend(
            f"- {a.tool_name}: BLOCKED, {a.summary}. Fix the arguments, "
            "or ask the right agent or the user."
            for a in actions
            if a.status == "blocked"
        )
        return "\n".join(lines)

    @staticmethod
    def _sum_usage(model: str, traces: list[LllmTrace]) -> SwarmUsage:
        return SwarmUsage(
            model=model,
            input_tokens=sum(t.input_tokens for t in traces),
            output_tokens=sum(t.output_tokens for t in traces),
            total_cost=sum(t.total_cost for t in traces),
        )
