"""One swarm agent turn: decide -> run allowed tools (up to swarm_max_tool_rounds) -> reply.

`ask` is a single structured call without tools, used by council steps.

Reuses the main pipeline's low level: LLM clients, parser, ToolFactory and
execute_tool_calls. Skips the heavy cmd_prompt / create_output stages to keep
every turn small.
"""

import json
import logging
import os
from typing import Any
from archie_shared.chat.models import LllmTrace
from archie_shared.swarm.models import SwarmActionInfo, SwarmMessage, SwarmUsage
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field
from ..agent.prompt_builder import PromptBuilder
from ..backend.llm.registry import get_client
from ..backend.state_service import StateService
from ..config import settings
from ..models.orchestration_sgr import ToolCallRequest
from ..models.state_models import UserState
from ..models.tool_models import ToolResult
from ..tools.tool_factory import ToolFactory
from ..utils.llm_parser import parse_llm_response
from ..utils.provider_utils import get_provider_for_model
from ..utils.skill_utils import load_skill, load_skill_references
from ..utils.tool_executor import execute_tool_calls
from .agents import SWARM_AGENTS, SWARM_GROUPS, SwarmRoster
from .models import SwarmAgentProfile, SwarmReply, SwarmToolCall, SwarmTurn
from .policy import READ_ONLY_TOOLS, check_tool_call


logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "agent", "prompts")
SWARM_PROMPTS_DIR = os.path.join(PROMPTS_DIR, "swarm")


FINAL_REPLY_TASK = (
    "Write your chat reply now. You must reply: say what you did, "
    "or say that you could not do the task and why."
)


def create_swarm_prompt_env() -> Environment:
    # No autoescape: prompts are plain text and JSON parameters must stay readable.
    # PROMPTS_DIR is on the path so swarm prompts can include persona_prompts/.
    return Environment(
        loader=FileSystemLoader([SWARM_PROMPTS_DIR, PROMPTS_DIR]), autoescape=False
    )


def format_chat_line(message: SwarmMessage) -> str:
    sender = "user" if message.sender == "user" else f"@{message.sender}"
    reply = f" (reply to [{message.reply_to}])" if message.reply_to else ""
    return f"[{message.message_id}] {sender}{reply}: {message.text}"


class SwarmDraft(BaseModel):
    """Result of an agent turn, ready to post to the chat."""

    text: str = Field(description="Reply text")
    actions: list[SwarmActionInfo] = Field(description="Side-effect tool calls made")
    usage: SwarmUsage = Field(description="Summed LLM usage of the turn")


class SwarmAgentRunner:
    def __init__(
        self,
        tool_factory: ToolFactory | None = None,
        state_service: StateService | None = None,
        prompt_builder: PromptBuilder | None = None,
        roster: SwarmRoster | None = None,
    ) -> None:
        self.tool_factory = tool_factory or ToolFactory()
        # Dry run: tools with side effects run in demo mode, reads stay real.
        # Both factories share one tool registry (filled by get_tool_schemas).
        self.action_factory = ToolFactory(demo_mode=settings.swarm_dry_run)
        self.action_factory.tools = self.tool_factory.tools
        self.state_service = state_service or StateService()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.roster = roster or SwarmRoster()
        self.env = create_swarm_prompt_env()

    async def get_user_state(self) -> UserState:
        """State of the swarm owner; settings.swarm_user_name wins if set."""
        user_name = settings.swarm_user_name or await self.roster.get_owner()
        return await self.state_service.get_user_state(user_name=user_name)

    def build_system_prompt(self, profile: SwarmAgentProfile, model: str) -> str:
        """Static, cacheable part: rules, roster, tools and skill of the agent."""
        tools = self.tool_factory.get_tool_schemas(
            model, "plain", allowed_tools=set(profile.tools)
        )
        return self.env.get_template("swarm_base.jinja2").render(
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
        self, state: UserState, history: list[SwarmMessage], trigger: SwarmMessage
    ) -> str:
        """Volatile tail: time, compact user context and recent chat lines."""
        state_dict = state.model_dump()
        lines = [
            self.prompt_builder.build_datetime_note(state_dict),
            f"User city: {state.default_city}, user language: {state.language}",
            "Recent chat:",
            *(
                format_chat_line(m)
                for m in history
                if m.message_id != trigger.message_id
            ),
        ]
        return "\n".join(lines)

    async def build_messages(
        self,
        profile: SwarmAgentProfile,
        model: str,
        trigger: SwarmMessage,
        history: list[SwarmMessage],
        task: str,
    ) -> list[dict[str, Any]]:
        state = await self.get_user_state()
        return [
            {
                "role": "system",
                "content": self.build_system_prompt(profile, model),
                "cacheable": True,
            },
            {"role": "system", "content": self.build_context(state, history, trigger)},
            {"role": "user", "content": task},
        ]

    def render_task(self, template: str, **context: Any) -> str:
        return self.env.get_template(template).render(**context)

    async def ask(
        self,
        profile: SwarmAgentProfile,
        trigger: SwarmMessage,
        history: list[SwarmMessage],
        task: str,
        schema: type[BaseModel],
    ) -> tuple[Any, SwarmUsage]:
        """One structured call without tools, for council steps."""
        model = profile.model or settings.swarm_model
        provider = get_provider_for_model(model)
        messages = await self.build_messages(profile, model, trigger, history, task)
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
        model = profile.model or settings.swarm_model
        provider = get_provider_for_model(model)
        client = get_client(provider)
        messages = await self.build_messages(
            profile,
            model,
            trigger,
            history,
            f"React to this message:\n{format_chat_line(trigger)}",
        )
        traces: list[LllmTrace] = []
        actions: list[SwarmActionInfo] = []
        turn: SwarmTurn | SwarmReply = await self._call(
            client, provider, model, messages, SwarmTurn, traces
        )
        rounds = 0
        # Several tool rounds let an agent read the state, then change it.
        while isinstance(turn, SwarmTurn) and turn.tool_calls:
            rounds += 1
            logger.info(
                f"swarm_runner_001: \033[36m{profile.agent_id}\033[0m round {rounds} tools="
                f"{[tc.tool_name for tc in turn.tool_calls]} reasoning: \033[33m{turn.reasoning}\033[0m"
            )
            results, new_actions = await self._execute(profile, turn.tool_calls)
            actions.extend(new_actions)
            messages.append(
                {
                    "role": "system",
                    "content": self._format_results(results, new_actions),
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
                        FINAL_REPLY_TASK
                        if last_round
                        else f"Call more tools if you still need them. Else: {FINAL_REPLY_TASK}"
                    ),
                }
            )
            turn = await self._call(
                client,
                provider,
                model,
                messages,
                SwarmReply if last_round else SwarmTurn,
                traces,
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
        self, profile: SwarmAgentProfile, tool_calls: list[SwarmToolCall]
    ) -> tuple[list[ToolResult], list[SwarmActionInfo]]:
        allowed: list[SwarmToolCall] = []
        actions: list[SwarmActionInfo] = []
        for call in tool_calls:
            if check_tool_call(profile, call) == "allow":
                allowed.append(call)
            else:
                actions.append(
                    self._action_info(call, "blocked", "Not allowed for this agent")
                )
        reads = [c for c in allowed if c.tool_name in READ_ONLY_TOOLS]
        writes = [c for c in allowed if c.tool_name not in READ_ONLY_TOOLS]
        results = await self._run_calls(
            profile, reads, self.tool_factory
        ) + await self._run_calls(profile, writes, self.action_factory)
        allowed = reads + writes
        for call, result in zip(allowed, results, strict=True):
            if call.tool_name not in READ_ONLY_TOOLS:
                actions.append(
                    self._action_info(
                        call, "success" if result.success else "error", result.error
                    )
                )
        return results, actions

    @staticmethod
    async def _run_calls(
        profile: SwarmAgentProfile, calls: list[SwarmToolCall], factory: ToolFactory
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
        results: list[ToolResult], actions: list[SwarmActionInfo]
    ) -> str:
        limit = settings.swarm_tool_result_max_chars
        lines = ["Tool results:"]
        for result in results:
            output = json.dumps(result.output, ensure_ascii=False, default=str)[:limit]
            status = "ok" if result.success else f"error: {result.error}"
            lines.append(f"- {result.tool_name} ({status}): {output}")
        lines.extend(
            f"- {a.tool_name}: BLOCKED. You may not do this. Ask the right agent or the user."
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
