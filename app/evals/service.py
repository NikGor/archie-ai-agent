"""Eval run orchestration: one run at a time, live progress events, publishing.

Run state lives in memory: a run does not survive an agent restart, and Django
keeps the durable history (see EvalPublisher).
"""

import asyncio
import contextlib
import logging
import os
import subprocess
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from archie_shared.evals.models import (
    EvalAttemptFinishedEvent,
    EvalAttemptRef,
    EvalAttemptStartedEvent,
    EvalAttemptStepEvent,
    EvalCaseInfo,
    EvalCaseResult,
    EvalRunFinishedEvent,
    EvalRunInfo,
    EvalRunRequest,
    EvalServerEvent,
    EvalSnapshotEvent,
)
from app.config import settings
from .catalog import CatalogError, compute_prompt_hashes, get_catalog, select_cases
from .publisher import EvalPublisher, eval_publisher
from .runner import EvalAttemptRunner


logger = logging.getLogger(__name__)

_KEEP_RUNS = 20

AttemptKey = tuple[str, str, int]


class EvalBusyError(RuntimeError):
    """Another eval run is in progress."""


class EvalRunNotFoundError(KeyError):
    """No run with this ID in memory."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _git_sha() -> str | None:
    if sha := os.environ.get("GIT_SHA"):
        return sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


@dataclass
class _RunState:
    info: EvalRunInfo
    cases: list[EvalCaseInfo]
    results: list[EvalCaseResult] = field(default_factory=list)
    running: dict[AttemptKey, EvalAttemptRef] = field(default_factory=dict)
    subscribers: set[asyncio.Queue[EvalServerEvent]] = field(default_factory=set)
    task: asyncio.Task | None = None

    def emit(self, event: EvalServerEvent) -> None:
        for queue in self.subscribers:
            queue.put_nowait(event)

    def snapshot(self) -> EvalSnapshotEvent:
        return EvalSnapshotEvent(
            run=self.info,
            cases=self.cases,
            results=self.results,
            running=list(self.running.values()),
        )


class EvalService:
    def __init__(
        self,
        runner: EvalAttemptRunner | None = None,
        publisher: EvalPublisher = eval_publisher,
    ) -> None:
        self._runner = runner
        self.publisher = publisher
        self._runs: OrderedDict[str, _RunState] = OrderedDict()

    @property
    def runner(self) -> EvalAttemptRunner:
        if self._runner is None:
            self._runner = EvalAttemptRunner()
        return self._runner

    def active_run(self) -> EvalRunInfo | None:
        for state in self._runs.values():
            if state.info.status in ("queued", "running"):
                return state.info
        return None

    async def start(self, request: EvalRunRequest) -> EvalRunInfo:
        """Start a run in the background; raises EvalBusyError or CatalogError."""
        if active := self.active_run():
            raise EvalBusyError(f"Run {active.run_id} is in progress")
        cases = select_cases(get_catalog(), request)
        if not cases:
            raise CatalogError("No cases match the request")
        judge_model = request.judge_model or settings.eval_default_model
        request = request.model_copy(
            update={"judge_model": judge_model if request.judge else None}
        )
        info = EvalRunInfo(
            run_id=uuid.uuid4().hex[:12],
            status="queued",
            request=request,
            case_ids=[case.id for case in cases],
            git_sha=_git_sha(),
            prompt_hashes=compute_prompt_hashes(),
            total_attempts=len(cases) * len(request.models) * request.repeats,
            created_at=_now(),
        )
        state = _RunState(info=info, cases=cases)
        self._runs[info.run_id] = state
        while len(self._runs) > _KEEP_RUNS:
            self._runs.popitem(last=False)
        state.task = asyncio.create_task(self._execute(state))
        logger.info(
            f"eval_service_001: Run \033[36m{info.run_id}\033[0m started: "
            f"\033[33m{info.total_attempts}\033[0m attempts, reason: {request.reason}"
        )
        return info

    def get(self, run_id: str) -> _RunState:
        if run_id not in self._runs:
            raise EvalRunNotFoundError(run_id)
        return self._runs[run_id]

    def list_runs(self) -> list[EvalRunInfo]:
        return [state.info for state in reversed(self._runs.values())]

    async def cancel(self, run_id: str) -> EvalRunInfo:
        state = self.get(run_id)
        if state.task and not state.task.done():
            state.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await state.task
        return state.info

    def subscribe(self, run_id: str) -> asyncio.Queue[EvalServerEvent]:
        """Subscribe to a run; the queue starts with a snapshot."""
        state = self.get(run_id)
        queue: asyncio.Queue[EvalServerEvent] = asyncio.Queue()
        queue.put_nowait(state.snapshot())
        state.subscribers.add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[EvalServerEvent]) -> None:
        if run_id in self._runs:
            self._runs[run_id].subscribers.discard(queue)

    def _update(self, state: _RunState, **changes: object) -> None:
        state.info = state.info.model_copy(update=changes)

    async def _execute(self, state: _RunState) -> None:
        request = state.info.request
        self._update(state, status="running")
        self.publisher.publish(state.snapshot())
        semaphore = asyncio.Semaphore(settings.eval_concurrency)
        jobs = [
            self._run_attempt(state, semaphore, case, model, attempt)
            for attempt in range(1, request.repeats + 1)
            for model in request.models
            for case in state.cases
        ]
        status, error = "completed", None
        try:
            await asyncio.gather(*jobs)
        except asyncio.CancelledError:
            status = "cancelled"
        except Exception as exc:
            logger.error(
                f"eval_service_error_001: Run \033[36m{state.info.run_id}\033[0m failed: {exc}"
            )
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            state.running.clear()
            self._update(state, status=status, error=error, finished_at=_now())
            state.emit(EvalRunFinishedEvent(run=state.info))
            self.publisher.publish(state.info)
            logger.info(
                f"eval_service_002: Run \033[36m{state.info.run_id}\033[0m {status}: "
                f"passed \033[32m{state.info.passed}\033[0m, failed \033[31m{state.info.failed}\033[0m, "
                f"errors \033[31m{state.info.errors}\033[0m"
            )

    async def _run_attempt(
        self,
        state: _RunState,
        semaphore: asyncio.Semaphore,
        case: EvalCaseInfo,
        model: str,
        attempt: int,
    ) -> None:
        key: AttemptKey = (case.id, model, attempt)
        async with semaphore:
            ref = EvalAttemptRef(case_id=case.id, model=model, attempt=attempt)
            state.running[key] = ref
            state.emit(EvalAttemptStartedEvent(attempt=ref))

            async def on_step(step: str) -> None:
                current = state.running.get(key)
                if current is None or current.step == step:
                    return
                state.running[key] = current.model_copy(update={"step": step})
                state.emit(EvalAttemptStepEvent(attempt=state.running[key]))

            result = await self.runner.run(
                state.info.run_id,
                case,
                model,
                attempt,
                state.info.request.judge_model,
                on_step,
            )
            state.running.pop(key, None)
            state.results.append(result)
            info = state.info
            self._update(
                state,
                done_attempts=info.done_attempts + 1,
                passed=info.passed + (result.status == "passed"),
                failed=info.failed + (result.status == "failed"),
                errors=info.errors + (result.status == "error"),
                total_tokens=info.total_tokens + (result.total_tokens or 0),
            )
            state.emit(EvalAttemptFinishedEvent(result=result, run=state.info))
            self.publisher.publish(result)


eval_service = EvalService()
