"""Langfuse tracing for chat turns (ARCHIE-211) and swarm runs (ARCHIE-281).

The Langfuse SDK is OpenTelemetry-based: observations nest through the active
OTel context, and the OpenAI integration (`langfuse.openai`) attaches its
generations to whatever observation is current.

Only chat turns and swarm runs are recorded. Every trace must start at the
`handle-chat` root or at a swarm root (`swarm-check`, `swarm-council`,
`swarm-agent-turn`): any other root span (eval LLM calls, Django title
generation, dashboard/widget requests) is dropped by the sampler, and its
children inherit that decision via ParentBased sampling. Swarm traces of one
local day share the session `swarm-<YYYY-MM-DD>`.
"""

import contextlib
import functools
import logging
import re
from collections.abc import Awaitable, Callable, Iterator, Sequence
from datetime import date
from typing import Any, Concatenate, Literal, ParamSpec
from archie_shared.chat.models import ChatMessage, ChatRequest, Content
from langfuse import Langfuse, LangfuseAgent, get_client, propagate_attributes
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import (
    Decision,
    ParentBased,
    Sampler,
    SamplingResult,
)
from opentelemetry.trace import Link, SpanKind, TraceState
from opentelemetry.util.types import Attributes
from app.monitoring.spec_rules import spec_versions


logger = logging.getLogger(__name__)

CHAT_TRACE_NAME = "handle-chat"
# Stage 1 LLM call of a chat turn; chat monitoring reads its prompt (ARCHIE-256)
COMMAND_TRACE_NAME = "decide-action"
# Django sends conversation title generation as user "system"
SYSTEM_USER_NAME = "system"
UNTRACED_FORMATS = frozenset({"dashboard", "widget"})
SwarmTraceName = Literal["swarm-check", "swarm-council", "swarm-agent-turn"]
SWARM_TRACE_NAMES: frozenset[str] = frozenset(
    {"swarm-check", "swarm-council", "swarm-agent-turn"}
)
TRACED_ROOT_NAMES = SWARM_TRACE_NAMES | {CHAT_TRACE_NAME}
SWARM_SESSION_PREFIX = "swarm-"

_SECRET_PATTERN = re.compile(
    r"sk-[A-Za-z0-9_-]{16,}|pk-lf-[A-Za-z0-9-]+|AIza[0-9A-Za-z_-]{30,}|Bearer\s+\S+"
)

P = ParamSpec("P")


class _ChatTurnSampler(Sampler):
    """Root sampler: keep a trace only if its root span is a chat turn or a swarm run."""

    def should_sample(
        self,
        parent_context: Context | None,  # noqa: ARG002
        trace_id: int,  # noqa: ARG002
        name: str,
        kind: SpanKind | None = None,  # noqa: ARG002
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,  # noqa: ARG002
        trace_state: TraceState | None = None,
    ) -> SamplingResult:
        if name in TRACED_ROOT_NAMES:
            return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes, trace_state)
        return SamplingResult(Decision.DROP, None, trace_state)

    def get_description(self) -> str:
        return "ChatTurnSampler"


def _mask_secrets(*, data: Any, **_kwargs: Any) -> Any:
    """Redact API-key-like strings from observation input/output/metadata."""
    if isinstance(data, str):
        return _SECRET_PATTERN.sub("[REDACTED]", data)
    if isinstance(data, dict):
        return {k: _mask_secrets(data=v) for k, v in data.items()}
    if isinstance(data, list):
        return [_mask_secrets(data=v) for v in data]
    return data


def init_tracing() -> None:
    """Register the Langfuse client singleton used by get_client() and langfuse.openai.

    Must run before the first LLM call. Without LANGFUSE_* keys (or with
    LANGFUSE_TRACING_ENABLED=false) the client is a no-op.
    """
    Langfuse(
        tracer_provider=TracerProvider(
            resource=Resource.create({"service.name": "archie-ai-agent"}),
            sampler=ParentBased(root=_ChatTurnSampler()),
        ),
        mask=_mask_secrets,
    )
    logger.info("tracing_001: Langfuse tracing initialized")


def shutdown_tracing() -> None:
    """Flush buffered observations before the process exits."""
    get_client().shutdown()


def content_output(content: Content | None) -> dict[str, Any] | None:
    """Observation output for a response: only the filled Content fields."""
    return content.model_dump(mode="json", exclude_none=True) if content else None


def is_traced_request(request: ChatRequest) -> bool:
    return (
        request.user_name != SYSTEM_USER_NAME
        and request.response_format not in UNTRACED_FORMATS
    )


def trace_chat_turn(
    handler: Callable[Concatenate[ChatRequest, P], Awaitable[ChatMessage]],
) -> Callable[Concatenate[ChatRequest, P], Awaitable[ChatMessage]]:
    """Wrap a chat handler in the root observation of the turn.

    One trace per turn, grouped into a session per conversation. Trace input is
    the user's text, output is the Content the frontend renders.
    """

    @functools.wraps(handler)
    async def wrapper(
        request: ChatRequest, *args: P.args, **kwargs: P.kwargs
    ) -> ChatMessage:
        if not is_traced_request(request):
            return await handler(request, *args, **kwargs)
        with (
            get_client().start_as_current_observation(
                as_type="agent", name=CHAT_TRACE_NAME, input=request.input
            ) as root,
            propagate_attributes(
                session_id=request.conversation_id,
                user_id=request.user_name,
                tags=[request.response_format],
            ),
        ):
            message = await handler(request, *args, **kwargs)
            root.update(
                output=content_output(message.content),
                metadata={
                    "model": message.model,
                    "message_id": message.message_id,
                    "demo_mode": bool(request.demo_mode),
                    # Prompt spec versions that produced the turn: chat monitoring
                    # and prompt fixes compare traces before / after a fix (ARCHIE-213)
                    "spec_versions": spec_versions(
                        message.response_format or request.response_format,
                        trace_found=True,
                    ),
                },
            )
            return message

    return wrapper


def swarm_session_id(day: date) -> str:
    """Langfuse session of all swarm traces of one local day."""
    return f"{SWARM_SESSION_PREFIX}{day.isoformat()}"


@contextlib.contextmanager
def trace_swarm_run(
    name: SwarmTraceName,
    run_id: str,
    agent_id: str,
    day: date,
    trace_input: str,
    **metadata: str | int | None,
) -> Iterator[LangfuseAgent]:
    """Root observation of one swarm step (ARCHIE-281): a check, a council
    discussion or one agent turn. `run_id` links the traces of one run; `day`
    is the swarm owner's local day: the session of the trace."""
    with (
        get_client().start_as_current_observation(
            as_type="agent",
            name=name,
            input=trace_input,
            metadata={"swarm_run_id": run_id, "agent_id": agent_id, **metadata},
        ) as root,
        propagate_attributes(
            session_id=swarm_session_id(day),
            tags=["swarm", agent_id],
            metadata={"swarm_run_id": run_id},
        ),
    ):
        yield root
