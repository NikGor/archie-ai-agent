"""Swarm journal (ARCHIE-281): sends every posted swarm message to Django.

The Redis stream keeps only the last messages and Stop clears it. The daily
swarm audit needs the whole day, so the journal listens on the hub and posts
batches to the Django swarm journal. Django stores each message once per
message_id, so a batch sent again after a failure is safe. When Django is not
reachable the batch stays in the buffer; past the buffer size the oldest
messages are dropped with a warning.
"""

import asyncio
import contextlib
import logging
from collections import deque
import httpx
from archie_shared.swarm.models import (
    SwarmJournalBatch,
    SwarmMessage,
    SwarmMessageEvent,
)
from app.agents.swarm.bus import SwarmHub
from app.config import settings


logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10.0
_BATCH_SIZE = 200


class SwarmJournal:
    def __init__(self, hub: SwarmHub) -> None:
        self.hub = hub
        self.buffer: deque[SwarmMessage] = deque()
        self._worker: asyncio.Task[None] | None = None
        self._flusher: asyncio.Task[None] | None = None

    @staticmethod
    def is_configured() -> bool:
        return settings.swarm_journal_enabled and bool(settings.eval_api_token)

    async def start(self) -> None:
        if not self.is_configured():
            logger.warning(
                "swarm_journal_warning_001: Journal is off: "
                "SWARM_JOURNAL_ENABLED is false or EVAL_API_TOKEN is not set"
            )
            return
        queue = self.hub.subscribe()
        self._worker = asyncio.create_task(self._listen(queue), name="swarm-journal")
        self._flusher = asyncio.create_task(
            self._flush_loop(), name="swarm-journal-flush"
        )
        logger.info("swarm_journal_001: Journal started")

    async def stop(self) -> None:
        for task in [self._worker, self._flusher]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if self.buffer:
            await self.flush()

    def add(self, message: SwarmMessage) -> None:
        self.buffer.append(message)
        overflow = len(self.buffer) - settings.swarm_journal_buffer_size
        for _ in range(max(overflow, 0)):
            dropped = self.buffer.popleft()
            logger.warning(
                f"swarm_journal_warning_002: Buffer full, dropped {dropped.message_id}"
            )

    async def _listen(self, queue: asyncio.Queue) -> None:
        while True:
            event = await queue.get()
            if isinstance(event, SwarmMessageEvent):
                self.add(event.message)

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(settings.swarm_journal_flush_seconds)
            if self.buffer:
                await self.flush()

    async def flush(self) -> None:
        """Send the buffer in batches; keep what failed for the next flush."""
        while self.buffer:
            batch = list(self.buffer)[:_BATCH_SIZE]
            try:
                await self._send(batch)
            except httpx.HTTPError as exc:
                logger.error(
                    f"swarm_journal_error_001: {len(batch)} message(s) not sent, "
                    f"retry later: \033[31m{exc}\033[0m"
                )
                return
            # A full buffer may drop messages while the batch is on its way
            sent = {m.message_id for m in batch}
            while self.buffer and self.buffer[0].message_id in sent:
                self.buffer.popleft()

    async def _send(self, messages: list[SwarmMessage]) -> None:
        base = settings.web_django_url.rstrip("/")
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{base}/swarm-journal/api/ingest/",
                content=SwarmJournalBatch(messages=messages).model_dump_json(),
                headers={
                    "Authorization": f"Bearer {settings.eval_api_token}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
        logger.info(f"swarm_journal_002: Sent {len(messages)} message(s)")
