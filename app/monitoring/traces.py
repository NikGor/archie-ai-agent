"""Tool and skill calls of stored chat turns, read back from Langfuse (ARCHIE-213).

Each chat turn is one `handle-chat` trace in the conversation's session (see
app/utils/tracing.py). Django does not store tool calls, so the judge gets them
from here; turns older than tracing have no trace.
"""

import itertools
import json
import logging
from datetime import datetime, timedelta
from archie_shared.monitoring.models import MonitorToolCall
from langfuse import get_client
from pydantic import BaseModel, Field
from app.agent.prompt_builder import SKILLS_HEADER
from app.utils.tracing import CHAT_TRACE_NAME, COMMAND_TRACE_NAME


logger = logging.getLogger(__name__)

SKILL_LOADER_TOOL = "skill_loader_tool"
_PAGE_SIZE = 100
_FIELDS = "core,basic,io,metadata"
_SPEC_VERSIONS_PREFIX = "spec_versions."


class TraceTurn(BaseModel):
    """One chat turn as recorded in Langfuse"""

    trace_id: str = Field(description="Langfuse trace ID")
    started_at: datetime = Field(description="Start of the handle-chat observation")
    message_id: str | None = Field(default=None, description="ChatMessage.message_id")
    tool_calls: list[MonitorToolCall] = Field(default_factory=list)
    spec_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Spec -> fingerprint of the prompt version that produced the turn",
    )
    skills_offered: list[str] | None = Field(
        default=None,
        description="'- name: description' lines of the skills that the turn's Stage 1 "
        "prompt listed; None when the turn has no Stage 1 call",
    )

    @property
    def skills_loaded(self) -> list[str]:
        skills = []
        for call in self.tool_calls:
            if call.name != SKILL_LOADER_TOOL:
                continue
            name = json.loads(call.arguments or "{}").get("skill_name")
            if name and name not in skills:
                skills.append(name)
        return skills


def offered_skills(generation_input: object) -> list[str]:
    """Skill lines of the system prompt of a Stage 1 call, as Langfuse stored it.

    The judge needs the skills of the turn, not the current ones: a skill added
    later made older turns break load_matching_skill (ARCHIE-256).
    """
    messages = (
        json.loads(generation_input)
        if isinstance(generation_input, str)
        else generation_input
    )
    system_prompt = next(m["content"] for m in messages if m.get("role") == "system")
    _, found, rest = system_prompt.partition(SKILLS_HEADER)
    if not found:
        return []
    lines = rest.lstrip("\n").splitlines()
    return list(itertools.takewhile(lambda line: line.startswith("- "), lines))


def _spec_versions(metadata: dict) -> dict[str, str]:
    """Langfuse returns nested metadata flattened: spec_versions.<spec> -> version."""
    return {
        key.removeprefix(_SPEC_VERSIONS_PREFIX): str(value)
        for key, value in metadata.items()
        if key.startswith(_SPEC_VERSIONS_PREFIX)
    }


def _tool_success(output: str | None) -> bool | None:
    try:
        success = json.loads(output or "").get("success")
    except (ValueError, AttributeError):
        return None
    return success if isinstance(success, bool) else None


async def fetch_session_turns(session_id: str) -> list[TraceTurn]:
    """All chat turns of a conversation, oldest first. Raises on API errors."""
    observations = []
    cursor = None
    while True:
        page = await get_client().async_api.observations.get_many(
            session_id=session_id, fields=_FIELDS, limit=_PAGE_SIZE, cursor=cursor
        )
        observations.extend(page.data)
        cursor = page.meta.cursor if page.meta else None
        if not cursor or len(page.data) < _PAGE_SIZE:
            break

    turns: dict[str, TraceTurn] = {}
    for observation in observations:
        if observation.type == "AGENT" and observation.name == CHAT_TRACE_NAME:
            metadata = (
                observation.metadata if isinstance(observation.metadata, dict) else {}
            )
            turns[observation.trace_id] = TraceTurn(
                trace_id=observation.trace_id,
                started_at=observation.start_time,
                message_id=metadata.get("message_id"),
                spec_versions=_spec_versions(metadata),
            )
    for observation in sorted(observations, key=lambda o: o.start_time):
        turn = turns.get(observation.trace_id)
        if (
            turn is not None
            and observation.name == COMMAND_TRACE_NAME
            and turn.skills_offered is None
        ):
            try:
                turn.skills_offered = offered_skills(observation.input)
            except (
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                StopIteration,
            ) as e:
                logger.warning(
                    f"monitor_traces_warning_001: No skill list in Stage 1 prompt of "
                    f"trace {observation.trace_id}: {e!r}"
                )
        if observation.type == "TOOL" and observation.trace_id in turns:
            turns[observation.trace_id].tool_calls.append(
                MonitorToolCall(
                    name=observation.name or "",
                    arguments=str(observation.input or "{}"),
                    success=_tool_success(observation.output),
                )
            )
    logger.info(
        f"monitor_traces_001: Session \033[36m{session_id}\033[0m: "
        f"\033[33m{len(turns)}\033[0m traced turns"
    )
    return sorted(turns.values(), key=lambda turn: turn.started_at)


def match_turn(
    turns: list[TraceTurn],
    message_id: str,
    created_at: datetime,
    used: set[str],
    max_gap: timedelta,
) -> TraceTurn | None:
    """The trace of a stored message: same message_id, else the latest unused
    turn that started before the message was stored, within max_gap.

    The fallback covers turns where Django stored its own message ID."""
    for turn in turns:
        if turn.message_id == message_id:
            return turn
    candidates = [
        turn
        for turn in turns
        if turn.trace_id not in used
        and timedelta(0) <= created_at - turn.started_at <= max_gap
    ]
    return candidates[-1] if candidates else None
