"""Persistent Redis-backed scheduler for deferred Archie tool calls."""

import asyncio
import contextlib
import importlib
import inspect
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from croniter import CroniterBadCronError, croniter  # type: ignore[import-untyped]
from redis.exceptions import RedisError
from app.backend.redis_factory import get_async_redis
from app.config import TOOLS_CONFIG, settings


logger = logging.getLogger(__name__)

_JOBS_KEY = "archie:cron:jobs"
_DUE_KEY = "archie:cron:due"
_PROCESSING_KEY = "archie:cron:processing"
_DISALLOWED_TARGETS = {"cron_tool"}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _configured_tools() -> dict[str, str]:
    return {
        name: module_path
        for tools in TOOLS_CONFIG.values()
        for name, module_path in tools.items()
    }


def _timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown IANA timezone: {name}") from exc


def _parse_run_at(value: str, timezone_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("run_at must be an ISO 8601 date and time") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_timezone(timezone_name))
    return parsed.astimezone(UTC)


def _next_cron_run(expression: str, timezone_name: str, after: datetime) -> datetime:
    if len(expression.split()) != 5:
        raise ValueError("Cron expression must contain exactly five fields")
    local_after = after.astimezone(_timezone(timezone_name))
    try:
        next_local = croniter(expression, local_after).get_next(datetime)
    except (CroniterBadCronError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid cron expression: {expression}") from exc
    if next_local.tzinfo is None:
        next_local = next_local.replace(tzinfo=_timezone(timezone_name))
    return next_local.astimezone(UTC)


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key not in {"last_result"}}


class CronScheduler:
    """Store schedules in Redis and execute registered Archie tools when due."""

    def __init__(self) -> None:
        self.redis = get_async_redis()
        self._worker: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def create_job(
        self,
        *,
        name: str,
        tool_name: str,
        arguments: dict[str, Any],
        timezone: str,
        run_at: str | None = None,
        cron_expression: str | None = None,
    ) -> dict[str, Any]:
        """Validate and persist a one-shot or recurring job."""
        tools = _configured_tools()
        if tool_name not in tools or tool_name in _DISALLOWED_TARGETS:
            allowed = sorted(set(tools) - _DISALLOWED_TARGETS)
            raise ValueError(
                f"Tool cannot be scheduled: {tool_name}. Available: {', '.join(allowed)}"
            )
        if bool(run_at) == bool(cron_expression):
            raise ValueError("Provide exactly one of run_at or cron_expression")
        _timezone(timezone)

        now = _utc_now()
        if run_at:
            next_run = _parse_run_at(run_at, timezone)
            if next_run <= now:
                raise ValueError("run_at must be in the future")
            schedule_type = "once"
        else:
            next_run = _next_cron_run(cron_expression or "", timezone, now)
            schedule_type = "cron"

        self._validate_tool_arguments(tool_name, tools[tool_name], arguments)
        job_id = uuid.uuid4().hex[:12]
        job: dict[str, Any] = {
            "id": job_id,
            "name": name.strip() or f"Scheduled {tool_name}",
            "tool_name": tool_name,
            "arguments": arguments,
            "schedule_type": schedule_type,
            "run_at": run_at,
            "cron_expression": cron_expression,
            "timezone": timezone,
            "enabled": True,
            "created_at": _iso(now),
            "next_run_at": _iso(next_run),
            "last_run_at": None,
            "last_status": None,
            "last_error": None,
        }
        await self._save(job)
        await self.redis.zadd(_DUE_KEY, {job_id: next_run.timestamp()})
        return _public_job(job)

    async def list_jobs(self) -> list[dict[str, Any]]:
        values = await self.redis.hvals(_JOBS_KEY)
        jobs = [json.loads(value) for value in values]
        jobs.sort(key=lambda job: (job.get("next_run_at") or "~", job["name"]))
        return [_public_job(job) for job in jobs]

    async def get_job(self, job_id: str) -> dict[str, Any]:
        job = await self._load(job_id)
        if job is None:
            raise ValueError(f"Scheduled job not found: {job_id}")
        return _public_job(job)

    async def delete_job(self, job_id: str) -> dict[str, Any]:
        job = await self._load(job_id)
        if job is None:
            raise ValueError(f"Scheduled job not found: {job_id}")
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hdel(_JOBS_KEY, job_id)
            pipe.zrem(_DUE_KEY, job_id)
            pipe.zrem(_PROCESSING_KEY, job_id)
            await pipe.execute()
        return {"id": job_id, "name": job["name"], "deleted": True}

    async def set_enabled(self, job_id: str, enabled: bool) -> dict[str, Any]:
        job = await self._load(job_id)
        if job is None:
            raise ValueError(f"Scheduled job not found: {job_id}")
        job["enabled"] = enabled
        if enabled:
            now = _utc_now()
            if job["schedule_type"] == "once":
                next_run = _parse_run_at(job["run_at"], job["timezone"])
                if next_run <= now:
                    raise ValueError("This one-time job is already in the past")
            else:
                next_run = _next_cron_run(job["cron_expression"], job["timezone"], now)
            job["next_run_at"] = _iso(next_run)
            await self.redis.zadd(_DUE_KEY, {job_id: next_run.timestamp()})
        else:
            await self.redis.zrem(_DUE_KEY, job_id)
        await self._save(job)
        return _public_job(job)

    async def run_job_now(self, job_id: str) -> dict[str, Any]:
        job = await self._load(job_id)
        if job is None:
            raise ValueError(f"Scheduled job not found: {job_id}")
        result = await self._call_tool(job)
        return {"job": _public_job(job), "result": result}

    async def run_due_once(self) -> int:
        """Claim and execute one batch of due jobs; exposed for tests and worker."""
        now = _utc_now()
        job_ids = await self.redis.zrangebyscore(
            _DUE_KEY,
            min="-inf",
            max=now.timestamp(),
            start=0,
            num=settings.cron_max_jobs_per_tick,
        )
        executed = 0
        for job_id in job_ids:
            claimed = await self.redis.zrem(_DUE_KEY, job_id)
            if not claimed:
                continue
            await self.redis.zadd(_PROCESSING_KEY, {job_id: now.timestamp()})
            job = await self._load(job_id)
            if job is None or not job.get("enabled"):
                await self.redis.zrem(_PROCESSING_KEY, job_id)
                continue
            await self._execute_scheduled(job)
            executed += 1
        return executed

    async def start(self) -> None:
        if self._worker and not self._worker.done():
            return
        self._stopping.clear()
        self._worker = asyncio.create_task(self._run_loop(), name="archie-cron")
        logger.info("cron_scheduler_001: Scheduler started")

    async def stop(self) -> None:
        self._stopping.set()
        if self._worker:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        await self.redis.aclose()
        logger.info("cron_scheduler_002: Scheduler stopped")

    async def _run_loop(self) -> None:
        recovered = False
        try:
            while not self._stopping.is_set():
                try:
                    if not recovered:
                        await self._recover_processing()
                        recovered = True
                    await self.run_due_once()
                except RedisError as exc:
                    recovered = False
                    logger.error("cron_scheduler_error_001: Redis error: %s", exc)
                except Exception:
                    logger.exception("cron_scheduler_error_002: Scheduler tick failed")
                await asyncio.sleep(settings.cron_poll_interval)
        except asyncio.CancelledError:
            raise

    async def _recover_processing(self) -> None:
        job_ids = await self.redis.zrange(_PROCESSING_KEY, 0, -1)
        now = _utc_now().timestamp()
        for job_id in job_ids:
            job = await self._load(job_id)
            if job and job.get("enabled"):
                await self.redis.zadd(_DUE_KEY, {job_id: now})
            await self.redis.zrem(_PROCESSING_KEY, job_id)

    async def _execute_scheduled(self, job: dict[str, Any]) -> None:
        now = _utc_now()
        try:
            result = await self._call_tool(job)
            is_error = isinstance(result, dict) and (
                result.get("status") == "error" or "error" in result
            )
            job["last_status"] = "error" if is_error else "success"
            job["last_error"] = str(result) if is_error else None
            job["last_result"] = result
        except Exception as exc:
            logger.exception(
                "cron_scheduler_error_003: Job %s execution failed", job["id"]
            )
            job["last_status"] = "error"
            job["last_error"] = str(exc)
        job["last_run_at"] = _iso(now)

        if job["schedule_type"] == "cron" and job.get("enabled"):
            next_run = _next_cron_run(job["cron_expression"], job["timezone"], now)
            job["next_run_at"] = _iso(next_run)
            await self.redis.zadd(_DUE_KEY, {job["id"]: next_run.timestamp()})
        else:
            job["enabled"] = False
            job["next_run_at"] = None
        await self._save(job)
        await self.redis.zrem(_PROCESSING_KEY, job["id"])

    async def _call_tool(self, job: dict[str, Any]) -> Any:
        tools = _configured_tools()
        module_path = tools[job["tool_name"]]
        module = importlib.import_module(module_path)
        func = getattr(module, job["tool_name"])
        return await func(**job["arguments"])

    @staticmethod
    def _validate_tool_arguments(
        tool_name: str, module_path: str, arguments: dict[str, Any]
    ) -> None:
        module = importlib.import_module(module_path)
        func = getattr(module, tool_name)
        try:
            inspect.signature(func).bind(**arguments)
        except TypeError as exc:
            raise ValueError(f"Invalid arguments for {tool_name}: {exc}") from exc

    async def _load(self, job_id: str) -> dict[str, Any] | None:
        value = await self.redis.hget(_JOBS_KEY, job_id)
        return json.loads(value) if value else None

    async def _save(self, job: dict[str, Any]) -> None:
        await self.redis.hset(_JOBS_KEY, job["id"], json.dumps(job, ensure_ascii=False))


cron_scheduler = CronScheduler()
