"""One swarm agent turn: decide -> run allowed tools (up to swarm_max_tool_rounds) -> reply.

`ask` is a single structured call without tools, used by council steps.

Reuses the main pipeline's low level: LLM clients, parser, ToolFactory and
execute_tool_calls. Skips the heavy cmd_prompt / create_output stages to keep
every turn small.
"""

import json
import logging
from datetime import timedelta
from typing import Any
from archie_shared.chat.models import LllmTrace
from archie_shared.swarm.models import (
    SwarmActionInfo,
    SwarmMessage,
    SwarmUsage,
    SwarmWatchTask,
)
from pydantic import BaseModel, Field
from ..agent.prompt_builder import PromptBuilder
from ..agent.prompt_compiler import PromptCompiler
from ..backend.llm.registry import get_client
from ..backend.state_service import StateService
from ..config import settings
from ..models.orchestration_sgr import ToolCallRequest
from ..models.state_models import UserState
from ..models.tool_models import ToolContext, ToolResult
from ..tools.tool_factory import ToolFactory
from ..utils.general_utils import local_now
from ..utils.llm_parser import parse_llm_response
from ..utils.provider_utils import get_provider_for_model
from ..utils.skill_utils import load_skill, load_skill_references
from ..utils.tool_executor import execute_tool_calls
from .agents import SWARM_AGENTS, SWARM_GROUPS, SwarmRoster
from .memory import DayNote, SwarmMemory
from .models import (
    SwarmAgentProfile,
    SwarmReply,
    SwarmToolCall,
    SwarmTurn,
    SwarmWatchReply,
    SwarmWatchTurn,
)
from .policy import block_reason, check_tool_call, is_read_only


logger = logging.getLogger(__name__)

# Before this hour, "today" in the user's words means the evening that just ended.
LATE_NIGHT_END_HOUR = 5

# Day memory lines in a prompt: notes of the own check, or alerts of all checks.
DAY_NOTES_IN_PROMPT = 10

FINAL_REPLY_TASK = (
    "Write your chat reply now. You must reply: say what you did, "
    "or say that you could not do the task and why."
)
WATCH_FINAL_TASK = (
    "Write the outcome of the check now. Write an alert only if something is "
    "new for the user. Else leave the reply empty."
)


def format_chat_line(message: SwarmMessage) -> str:
    sender = "user" if message.sender == "user" else f"@{message.sender}"
    reply = f" (reply to [{message.reply_to}])" if message.reply_to else ""
    # Alerts address agents without "@" in the text (ARCHIE-238).
    hidden = [m for m in message.mentions if f"@{m}" not in message.text]
    to = f" (to {', '.join(f'@{m}' for m in hidden)})" if hidden else ""
    tag = f" [{message.importance} alert]" if message.importance else ""
    return f"[{message.message_id}] {sender}{reply}{to}{tag}: {message.text}"


class SwarmDraft(BaseModel):
    """Result of an agent turn, ready to post to the chat."""

    text: str = Field(description="Reply text")
    actions: list[SwarmActionInfo] = Field(description="Side-effect tool calls made")
    usage: SwarmUsage = Field(description="Summed LLM usage of the turn")


class SwarmWatchDraft(BaseModel):
    """Result of a spontaneous check."""

    outcome: SwarmWatchReply = Field(description="Alert, importance and notes")
    usage: SwarmUsage = Field(description="Summed LLM usage of the check")


def format_day_note(note: DayNote) -> str:
    if note.reported:
        return f"{note.time} {note.task}, reported: {note.reported}"
    return f"{note.time} {note.task}: {note.note}"


class SwarmAgentRunner:
    def __init__(
        self,
        tool_factory: ToolFactory | None = None,
        state_service: StateService | None = None,
        prompt_builder: PromptBuilder | None = None,
        roster: SwarmRoster | None = None,
        memory: SwarmMemory | None = None,
    ) -> None:
        self.tool_factory = tool_factory or ToolFactory()
        # Dry run: tools with side effects run in demo mode, reads stay real.
        # Both factories share one tool registry (filled by get_tool_schemas).
        self.action_factory = ToolFactory(demo_mode=settings.swarm_dry_run)
        self.action_factory.tools = self.tool_factory.tools
        self.state_service = state_service or StateService()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.roster = roster or SwarmRoster()
        self.memory = memory or SwarmMemory()
        # Swarm prompts are rule specs (specs/swarm, rules/swarm.json; ARCHIE-285)
        self.compiler = PromptCompiler()

    async def get_user_state(self) -> UserState:
        """State of the swarm owner; settings.swarm_user_name wins if set."""
        user_name = settings.swarm_user_name or await self.roster.get_owner()
        return await self.state_service.get_user_state(user_name=user_name)

    def build_system_prompt(self, profile: SwarmAgentProfile, model: str) -> str:
        """Static, cacheable part: rules, roster, tools and skill of the agent."""
        tools = self.tool_factory.get_tool_schemas(
            model, "plain", allowed_tools=set(profile.tools)
        )
        return self.compiler.compile(
            "swarm_base",
            agent=profile,
            agents=list(SWARM_AGENTS.values()),
            groups=SWARM_GROUPS,
            tools=[
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": json.dumps(
                        tool.get("parameters", {}), ensure_ascii=False
                    ),
                }
                for tool in tools
            ],
            skill=load_skill(profile.skill) if profile.skill else None,
            references={
                name: body
                for skill_name, reference_names in profile.references.items()
                for name, body in load_skill_references(
                    skill_name, reference_names
                ).items()
            },
        )

    def build_context(
        self,
        state: UserState,
        history: list[SwarmMessage],
        trigger_id: str | None,
        memory_lines: list[str],
    ) -> str:
        """Volatile tail: time, compact user context, memory and recent chat lines."""
        state_dict = state.model_dump()
        now = local_now(state.user_timezone)
        late_night = (
            [
                'It is after midnight: "today" or "tonight" in the user\'s words '
                f"usually means {(now - timedelta(days=1)):%d.%m.%Y}, "
                f'"tomorrow" means {now:%d.%m.%Y}.'
            ]
            if now.hour < LATE_NIGHT_END_HOUR
            else []
        )
        lines = [
            self.prompt_builder.build_datetime_note(state_dict),
            *late_night,
            f"User city: {state.default_city}, user language: {state.language}",
            *memory_lines,
            "Recent chat:",
            *(format_chat_line(m) for m in history if m.message_id != trigger_id),
        ]
        return "\n".join(lines)

    async def build_memory_lines(
        self, state: UserState, watch_task: SwarmWatchTask | None
    ) -> list[str]:
        """Long-term facts, then today's notes: a check sees its own notes,
        every other agent sees today's alerts (ARCHIE-238)."""
        facts = await self.memory.list_facts()
        today = local_now(state.user_timezone).date()
        today_facts = await self.memory.get_today_facts(today)
        notes = await self.memory.get_day_notes(today)
        if watch_task:
            notes = [n for n in notes if n.task == watch_task]
        else:
            notes = [n for n in notes if n.reported]
        lines: list[str] = []
        if facts:
            lines += ["Long-term memory:", *(f"- {f.text}" for f in facts)]
        if today_facts:
            lines += ["Today's memory:", *(f"- {fact}" for fact in today_facts)]
        if notes:
            title = "Your notes today:" if watch_task else "Alerts today:"
            lines += [
                title,
                *(f"- {format_day_note(n)}" for n in notes[-DAY_NOTES_IN_PROMPT:]),
            ]
        return lines

    async def build_messages(
        self,
        state: UserState,
        profile: SwarmAgentProfile,
        model: str,
        trigger_id: str | None,
        history: list[SwarmMessage],
        task: str,
        watch_task: SwarmWatchTask | None = None,
    ) -> list[dict[str, Any]]:
        memory_lines = await self.build_memory_lines(state, watch_task)
        return [
            {
                "role": "system",
                "content": self.build_system_prompt(profile, model),
                "cacheable": True,
            },
            {
                "role": "system",
                "content": self.build_context(state, history, trigger_id, memory_lines),
            },
            {"role": "user", "content": task},
        ]

    def render_task(self, spec: str, **context: Any) -> str:
        """Task prompt of a turn from a swarm spec (swarm_watch, swarm_consul, ...)."""
        return self.compiler.compile(spec, **context)

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
        messages = await self.build_messages(
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
        messages = await self.build_messages(
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
        messages = await self.build_messages(
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
        """Tool rounds until the agent replies. Several rounds let an agent read
        the state, then change it."""
        provider = get_provider_for_model(model)
        client = get_client(provider)
        traces: list[LllmTrace] = []
        actions: list[SwarmActionInfo] = []
        turn: SwarmTurn | SwarmReply = await self._call(
            client, provider, model, messages, turn_schema, traces
        )
        rounds = 0
        while isinstance(turn, SwarmTurn) and turn.tool_calls:
            rounds += 1
            logger.info(
                f"swarm_runner_001: \033[36m{profile.agent_id}\033[0m round {rounds} tools="
                f"{[(tc.tool_name, {p.name: p.value for p in tc.arguments}) for tc in turn.tool_calls]} "
                f"reasoning: \033[33m{turn.reasoning}\033[0m"
            )
            results, new_actions = await self._execute(
                profile, turn.tool_calls, context
            )
            actions.extend(new_actions)
            messages.append(
                {
                    "role": "system",
                    "content": self._format_results(
                        results,
                        new_actions,
                        profile.tool_result_max_chars
                        or settings.swarm_tool_result_max_chars,
                    ),
                }
            )
            # After a successful side effect only the reply is left: no repeated actions.
            last_round = rounds >= settings.swarm_max_tool_rounds or any(
                a.status == "success" for a in new_actions
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        final_task
                        if last_round
                        else f"Call more tools if you still need them. Else: {final_task}"
                    ),
                }
            )
            turn = await self._call(
                client,
                provider,
                model,
                messages,
                reply_schema if last_round else turn_schema,
                traces,
            )
        return turn, actions, traces

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
        results = await self._run_calls(
            profile, reads, self.tool_factory, context
        ) + await self._run_calls(profile, writes, self.action_factory, context)
        allowed = reads + writes
        for call, result in zip(allowed, results, strict=True):
            if not is_read_only(call):
                actions.append(
                    self._action_info(
                        call, "success" if result.success else "error", result.error
                    )
                )
        return results, actions

    @staticmethod
    async def _run_calls(
        profile: SwarmAgentProfile,
        calls: list[SwarmToolCall],
        factory: ToolFactory,
        context: ToolContext,
    ) -> list[ToolResult]:
        if not calls:
            return []
        return await execute_tool_calls(
            tool_calls=[
                ToolCallRequest(
                    tool_name=call.tool_name,
                    arguments=call.arguments,
                    missing_parameters=[],
                    is_confirmed=True,
                    reason=f"swarm agent {profile.agent_id}",
                )
                for call in calls
            ],
            tool_factory=factory,
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
