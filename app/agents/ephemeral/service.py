"""On-demand AgentKind with trusted spawn policy and terminal publication."""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from langfuse import get_client
from app.agents.ephemeral.models import READ_TOOLS, EphemeralResult, SpawnRequest
from app.agents.ephemeral.runner import (
    CompletionCall,
    EphemeralCancelledError,
    EphemeralRunner,
    ToolCallRunner,
)
from app.platform.policy.spawn import SpawnDeniedError, SpawnGate
from app.platform.runtime.agent import RunContext
from app.platform.runtime.budget import Budget, Trigger
from app.platform.runtime.spawn import SpawnContext, spawn_scope


logger = logging.getLogger(__name__)
Publisher = Callable[[SpawnRequest, EphemeralResult], Awaitable[None]]
_PUBLISH_TIMEOUT_SECONDS = 5.0


class EphemeralService:
    name = "ephemeral"
    triggers = ("spawn",)

    def __init__(
        self,
        publisher: Publisher | None = None,
        call: CompletionCall | None = None,
        run_tools: ToolCallRunner | None = None,
        model: str | None = None,
        max_concurrent: int = 2,
        max_depth: int = 1,
        allowed_sources: tuple[str, ...] = ("home", "cognitive"),
        budget: Budget | None = None,
    ) -> None:
        self.publisher = publisher
        self.runner = EphemeralRunner(
            call=call, run_tools=run_tools, model=model, budget=budget
        )
        self.gate = SpawnGate(
            frozenset(allowed_sources), READ_TOOLS, max_concurrent, max_depth
        )

    async def spawn(
        self, goal: str, context: str, caller: SpawnContext, publish_to: str = "swarm"
    ) -> EphemeralResult:
        request = SpawnRequest(
            goal=goal,
            context=context,
            parent_run_id=caller.parent_run_id,
            publish_to=publish_to,
        )
        return await self.run_request(request, caller)

    async def run(self, trigger: Trigger, context: RunContext) -> EphemeralResult:
        if trigger.kind not in self.triggers:
            raise ValueError(f"Unsupported ephemeral trigger: {trigger.kind}")
        request = SpawnRequest.model_validate_json(context.payload)
        caller = SpawnContext(
            source=trigger.source,
            parent_run_id=context.parent_run_id or context.run_id,
            depth=context.depth,
            user_timezone=context.user_timezone,
            demo_mode=context.demo_mode,
        )
        return await self.run_request(request, caller, context.run_id)

    async def run_request(
        self, request: SpawnRequest, caller: SpawnContext, run_id: str | None = None
    ) -> EphemeralResult:
        identifier = run_id or uuid.uuid4().hex[:16]
        # ARCHIE-300: the wire request cannot choose a different parent identity.
        request = request.model_copy(update={"parent_run_id": caller.parent_run_id})
        result: EphemeralResult | None = None
        cancellation: asyncio.CancelledError | None = None
        try:
            self.gate.acquire(caller, request.tools)
        except SpawnDeniedError as exc:
            result = EphemeralResult(
                run_id=identifier,
                parent_run_id=caller.parent_run_id,
                status="blocked",
                summary=str(exc),
            )
        else:
            try:
                child = SpawnContext(
                    source=self.name,
                    parent_run_id=identifier,
                    depth=caller.depth + 1,
                    user_timezone=caller.user_timezone,
                    demo_mode=caller.demo_mode,
                )
                with (
                    spawn_scope(child),
                    get_client().start_as_current_observation(
                        as_type="agent",
                        name="ephemeral-run",
                        input=request.goal,
                        metadata={
                            "run_id": identifier,
                            "parent_run_id": caller.parent_run_id,
                            "source": caller.source,
                            "depth": caller.depth + 1,
                        },
                    ) as root,
                ):
                    try:
                        result = await self.runner.run(request, caller, identifier)
                    except EphemeralCancelledError as exc:
                        # Preserve completed evidence in the trace and publication.
                        result = exc.result
                        cancellation = exc
                    root.update(output=result.model_dump(mode="json"))
            except asyncio.CancelledError as exc:
                cancellation = exc
                result = result or EphemeralResult(
                    run_id=identifier,
                    parent_run_id=caller.parent_run_id,
                    status="partial",
                    summary="Diagnosis cancelled by the runtime.",
                )
            except Exception as exc:
                logger.exception(
                    f"ephemeral_service_error_002: Run {identifier} failed: {exc}"
                )
                # A tracing exit/update error must not discard completed work.
                result = result or EphemeralResult(
                    run_id=identifier,
                    parent_run_id=caller.parent_run_id,
                    status="error",
                    summary=f"Diagnosis failed: {type(exc).__name__}: {exc}",
                )
            finally:
                self.gate.release()
        assert result is not None
        try:
            await self._publish(request, result)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
        if cancellation is not None:
            raise cancellation
        return result

    async def _publish(self, request: SpawnRequest, result: EphemeralResult) -> None:
        """Finish a bounded terminal publication before propagating cancellation."""
        if self.publisher is None or request.publish_to == "none":
            return

        async def send() -> None:
            try:
                async with asyncio.timeout(_PUBLISH_TIMEOUT_SECONDS):
                    await self.publisher(request, result)
            except Exception:
                logger.exception(
                    f"ephemeral_service_error_001: Could not publish result {result.run_id}"
                )

        publication = asyncio.create_task(send(), name=f"publish-{result.run_id}")
        cancellation: asyncio.CancelledError | None = None
        while not publication.done():
            try:
                await asyncio.shield(publication)
            except asyncio.CancelledError as exc:
                cancellation = exc
        if publication.cancelled():
            cancellation = cancellation or asyncio.CancelledError()
        else:
            publication.result()
        if cancellation is not None:
            raise cancellation
