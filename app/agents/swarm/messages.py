"""Messages of a swarm agent turn (ARCHIE-295): the cacheable system prompt
(rules, roster, tools, skill), the volatile context (time, user, memory, chat)
and the task of the turn. Swarm prompts are rule specs (specs/swarm,
rules/swarm.json; ARCHIE-285)."""

import json
from datetime import timedelta
from typing import Any
from archie_shared.swarm.models import SwarmMessage, SwarmWatchTask
from app.agents.swarm.agents import SWARM_AGENTS, SWARM_GROUPS
from app.agents.swarm.memory import DayNote, SwarmMemory
from app.agents.swarm.models import SwarmAgentProfile
from app.models.state_models import UserState
from app.platform.prompts.compiler import PromptCompiler
from app.platform.prompts.context import datetime_note
from app.platform.tools.registry import ToolRegistry
from app.utils.general_utils import local_now
from app.utils.skill_utils import load_skill, load_skill_references


# Before this hour, "today" in the user's words means the evening that just ended.
LATE_NIGHT_END_HOUR = 5

# Day memory lines in a prompt: notes of the own check, or alerts of all checks.
DAY_NOTES_IN_PROMPT = 10


def format_chat_line(message: SwarmMessage) -> str:
    sender = "user" if message.sender == "user" else f"@{message.sender}"
    reply = f" (reply to [{message.reply_to}])" if message.reply_to else ""
    # Alerts address agents without "@" in the text (ARCHIE-238).
    hidden = [m for m in message.mentions if f"@{m}" not in message.text]
    to = f" (to {', '.join(f'@{m}' for m in hidden)})" if hidden else ""
    tag = f" [{message.importance} alert]" if message.importance else ""
    return f"[{message.message_id}] {sender}{reply}{to}{tag}: {message.text}"


def format_day_note(note: DayNote) -> str:
    if note.reported:
        return f"{note.time} {note.task}, reported: {note.reported}"
    return f"{note.time} {note.task}: {note.note}"


class SwarmPromptBuilder:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        memory: SwarmMemory,
        compiler: PromptCompiler | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.memory = memory
        self.compiler = compiler or PromptCompiler()
        # Current date/time line of the prompt; tests replace it to fix the clock
        self.datetime_note = datetime_note

    def build_system_prompt(self, profile: SwarmAgentProfile, model: str) -> str:
        """Static, cacheable part: rules, roster, tools and skill of the agent."""
        tools = self.tool_registry.schemas(model, profile.tools)
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
            self.datetime_note(state_dict),
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
