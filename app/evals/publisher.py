"""Sends eval runs and results to Django, which stores the history."""

import asyncio
import logging
import httpx
from archie_shared.evals.models import EvalCaseResult, EvalRunInfo
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential
from app.config import settings


logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 4


class EvalPublisher:
    """Posts updates in order through one background worker; never blocks the run."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[EvalRunInfo | EvalCaseResult] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    @property
    def enabled(self) -> bool:
        return bool(settings.eval_publish_url and settings.eval_publish_token)

    def publish(self, update: EvalRunInfo | EvalCaseResult) -> None:
        if not self.enabled:
            return
        self._queue.put_nowait(update)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._drain())

    async def flush(self) -> None:
        """Wait until all queued updates are sent."""
        await self._queue.join()

    async def _drain(self) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            while True:
                update = await self._queue.get()
                try:
                    await self._send(client, update)
                except Exception as exc:
                    logger.error(
                        f"eval_publisher_error_001: Update not stored in Django: {exc}"
                    )
                finally:
                    self._queue.task_done()

    async def _send(
        self, client: httpx.AsyncClient, update: EvalRunInfo | EvalCaseResult
    ) -> None:
        base = str(settings.eval_publish_url).rstrip("/")
        if isinstance(update, EvalRunInfo):
            url = f"{base}/runs/"
        else:
            url = f"{base}/runs/{update.run_id}/results/"
        headers = {
            "Authorization": f"Bearer {settings.eval_publish_token}",
            "Content-Type": "application/json",
        }
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(_MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=1, max=10),
            reraise=True,
        ):
            with attempt:
                response = await client.post(
                    url, content=update.model_dump_json(), headers=headers
                )
                response.raise_for_status()


eval_publisher = EvalPublisher()
