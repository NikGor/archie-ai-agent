"""Chat monitoring runs (ARCHIE-213): judge pending turns, report to Django and Langfuse.

One run at a time. Django gives the pending turns and keeps the results, so a run
lost on agent restart only leaves its turns pending for the next run.
"""

import asyncio
import logging
import uuid
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
import httpx
from archie_shared.monitoring.models import (
    MonitorConversation,
    MonitorPendingBatch,
    MonitorRunInfo,
    MonitorRunRequest,
    MonitorTurnEvaluation,
)
from langfuse import get_client
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential
from app.config import settings
from .judge import judge_turn, worst_grade
from .traces import TraceTurn, fetch_session_turns, match_turn


logger = logging.getLogger(__name__)

_KEEP_RUNS = 20
_HTTP_ATTEMPTS = 4
LANGFUSE_SCORE_NAME = "monitor_grade"
RULE_SCORE_PREFIX = "rule:"


class MonitorBusyError(RuntimeError):
    """Another monitor run is in progress."""


class MonitorConfigError(RuntimeError):
    """Django URL or token is not set, or Django is not reachable."""


class MonitorRunNotFoundError(KeyError):
    """No run with this ID in agent memory."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DjangoMonitorApi:
    """Django chat-monitoring API, authenticated with EVAL_API_TOKEN."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        if not (settings.monitor_django_url and settings.eval_api_token):
            raise MonitorConfigError(
                "MONITOR_DJANGO_URL and EVAL_API_TOKEN are not set"
            )
        self.base = settings.monitor_django_url.rstrip("/")
        self.client = client
        self.headers = {"Authorization": f"Bearer {settings.eval_api_token}"}

    async def get_pending(self, request: MonitorRunRequest) -> MonitorPendingBatch:
        params: dict[str, int | list[str]] = {
            "limit": request.max_conversations,
            "conversation_id": request.conversation_ids,
        }
        try:
            response = await self.client.get(
                f"{self.base}/pending/", params=params, headers=self.headers
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MonitorConfigError(f"Pending turns are not available: {exc}") from exc
        return MonitorPendingBatch.model_validate_json(response.content)

    async def post(
        self, path: str, update: MonitorRunInfo | MonitorTurnEvaluation
    ) -> None:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(_HTTP_ATTEMPTS),
            wait=wait_exponential(multiplier=1, max=10),
            reraise=True,
        ):
            with attempt:
                response = await self.client.post(
                    f"{self.base}/ingest/{path}/",
                    content=update.model_dump_json(),
                    headers={**self.headers, "Content-Type": "application/json"},
                )
                response.raise_for_status()


class MonitorRun:
    """One run: judges every pending turn of the batch."""

    def __init__(self, info: MonitorRunInfo, batch: MonitorPendingBatch) -> None:
        self.info = info
        self.batch = batch
        self.semaphore = asyncio.Semaphore(settings.monitor_concurrency)
        self.lock = asyncio.Lock()
        self.task: asyncio.Task | None = None

    async def execute(self, api: DjangoMonitorApi) -> None:
        try:
            await asyncio.gather(
                *(self._judge_conversation(api, c) for c in self.batch.conversations)
            )
            self.info.status = "completed"
        except Exception as exc:
            logger.error(
                f"monitor_service_error_001: Run {self.info.run_id} failed: {exc}"
            )
            self.info.status = "failed"
            self.info.error = str(exc)
        self.info.finished_at = _now()
        await self.publish_state(api)
        get_client().flush()
        logger.info(
            f"monitor_service_002: Run \033[36m{self.info.run_id}\033[0m {self.info.status}: "
            f"red {self.info.red}, orange {self.info.orange}, green {self.info.green}, "
            f"errors {self.info.errors}, cost ${self.info.total_cost:.4f}"
        )

    async def _judge_conversation(
        self, api: DjangoMonitorApi, conversation: MonitorConversation
    ) -> None:
        try:
            traces = await fetch_session_turns(conversation.conversation_id)
        except Exception as exc:
            logger.warning(
                f"monitor_service_warning_001: No Langfuse turns for "
                f"{conversation.conversation_id}: {exc}"
            )
            traces = []
        used: set[str] = set()
        pending = set(conversation.pending_message_ids)
        jobs = []
        for index, message in enumerate(conversation.messages):
            if message.role != "assistant":
                continue
            trace = match_turn(
                traces,
                message.message_id,
                datetime.fromisoformat(message.created_at),
                used,
                timedelta(seconds=settings.monitor_trace_match_seconds),
            )
            if trace:
                used.add(trace.trace_id)
            if message.message_id in pending:
                jobs.append(self._judge_message(api, conversation, index, trace))
        await asyncio.gather(*jobs)

    async def _judge_message(
        self,
        api: DjangoMonitorApi,
        conversation: MonitorConversation,
        index: int,
        trace: TraceTurn | None,
    ) -> None:
        message = conversation.messages[index]
        try:
            evaluation = await self._evaluate(api, conversation, index, trace)
        except Exception as exc:
            # One bad turn (odd stored content, judge or Django failure) must not
            # stop the run; it stays pending in Django and is retried next run
            logger.exception(f"monitor_service_error_002: {message.message_id}: {exc}")
            await self._count(None, None, api)
            return
        await self._count(evaluation.grade, evaluation.judge_cost, api)

    async def _evaluate(
        self,
        api: DjangoMonitorApi,
        conversation: MonitorConversation,
        index: int,
        trace: TraceTurn | None,
    ) -> MonitorTurnEvaluation:
        message = conversation.messages[index]
        async with self.semaphore:
            result = await judge_turn(
                conversation.messages, index, trace, self.info.judge_model
            )
        rules = result.rules
        evaluation = MonitorTurnEvaluation(
            run_id=self.info.run_id,
            conversation_id=conversation.conversation_id,
            message_id=message.message_id,
            grade=worst_grade(rules),
            rules=rules,
            spec_versions=result.spec_versions,
            trace_found=trace is not None,
            trace_id=trace.trace_id if trace else None,
            tool_calls=trace.tool_calls if trace else [],
            skills_loaded=trace.skills_loaded if trace else [],
            judge_model=result.model,
            judge_cost=result.cost,
            evaluated_at=_now(),
        )
        await api.post("evaluations", evaluation)
        if trace:
            self._score_trace(trace.trace_id, evaluation)
        return evaluation

    @staticmethod
    def _score_trace(trace_id: str, evaluation: MonitorTurnEvaluation) -> None:
        """Turn grade plus one score per broken rule, so Langfuse can filter
        traces by rule ("rule:format_ui_answer:ui_map")."""
        rules = evaluation.rules
        client = get_client()
        for rule in rules:
            if rule.grade not in ("red", "orange"):
                continue
            name = f"{RULE_SCORE_PREFIX}{rule.spec}:{rule.rule_id}"
            client.create_score(
                name=name,
                score_id=f"{name}-{trace_id}",
                value=rule.grade,
                trace_id=trace_id,
                data_type="CATEGORICAL",
                comment=rule.reason,
            )
        client.create_score(
            name=LANGFUSE_SCORE_NAME,
            # Fixed ID: a re-judged turn updates its score instead of adding one
            score_id=f"{LANGFUSE_SCORE_NAME}-{trace_id}",
            value=evaluation.grade,
            trace_id=trace_id,
            data_type="CATEGORICAL",
            # Broken rules, e.g. "format_ui_answer:ui_charts=red"
            comment=", ".join(
                f"{r.spec}:{r.rule_id}={r.grade}"
                for r in rules
                if r.grade in ("red", "orange")
            ),
        )

    async def _count(
        self, grade: str | None, cost: float | None, api: DjangoMonitorApi
    ) -> None:
        async with self.lock:
            self.info.done_turns += 1
            if grade is None:
                self.info.errors += 1
            else:
                setattr(self.info, grade, getattr(self.info, grade) + 1)
            self.info.total_cost += cost or 0.0
            await self.publish_state(api)

    async def publish_state(self, api: DjangoMonitorApi) -> None:
        try:
            await api.post("runs", self.info)
        except httpx.HTTPError as exc:
            logger.error(f"monitor_service_error_003: Run state not stored: {exc}")


class MonitorService:
    """Starts runs in the background and keeps the last ones in memory."""

    def __init__(self) -> None:
        self.runs: OrderedDict[str, MonitorRun] = OrderedDict()

    def active(self) -> MonitorRun | None:
        return next(
            (run for run in self.runs.values() if run.info.status == "running"), None
        )

    async def start(self, request: MonitorRunRequest) -> MonitorRunInfo:
        if self.active():
            raise MonitorBusyError("A monitor run is already in progress")
        client = httpx.AsyncClient(timeout=30)
        try:
            api = DjangoMonitorApi(client)
            batch = await api.get_pending(request)
        except MonitorConfigError:
            await client.aclose()
            raise
        info = MonitorRunInfo(
            run_id=uuid.uuid4().hex[:12],
            status="running",
            request=request,
            judge_model=request.judge_model or settings.monitor_judge_model,
            conversations=len(batch.conversations),
            total_turns=sum(len(c.pending_message_ids) for c in batch.conversations),
            created_at=_now(),
        )
        run = MonitorRun(info, batch)
        await run.publish_state(api)
        run.task = asyncio.create_task(self._execute(run, api, client))
        self.runs[info.run_id] = run
        while len(self.runs) > _KEEP_RUNS:
            self.runs.popitem(last=False)
        logger.info(
            f"monitor_service_001: Run \033[36m{info.run_id}\033[0m started: "
            f"{info.total_turns} turns in {info.conversations} conversations, "
            f"judge {info.judge_model}"
        )
        return info

    async def _execute(
        self, run: MonitorRun, api: DjangoMonitorApi, client: httpx.AsyncClient
    ) -> None:
        # The client is already open (start() fetched the pending turns with it)
        try:
            await run.execute(api)
        finally:
            await client.aclose()

    def list_runs(self) -> list[MonitorRunInfo]:
        return [run.info for run in reversed(self.runs.values())]

    def get(self, run_id: str) -> MonitorRunInfo:
        if run_id not in self.runs:
            raise MonitorRunNotFoundError(run_id)
        return self.runs[run_id].info


monitor_service = MonitorService()
