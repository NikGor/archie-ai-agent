"""Supervisor of the process's long-lived services (ARCHIE-298): starts and
stops them in order, watches their health and restarts a dead one with
backoff, and reports their state."""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from typing import Literal, Protocol
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

WATCH_SECONDS = 30
_BACKOFF_SECONDS = (5, 15, 60, 300)


class Service(Protocol):
    name: str

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def healthy(self) -> bool: ...


class ServiceStatus(BaseModel):
    name: str = Field(description="Service name")
    state: Literal["running", "stopped", "restarting", "failed"] = Field(
        description="Current state"
    )
    restarts: int = Field(default=0, description="Restarts since the process started")
    last_error: str | None = Field(default=None, description="Last start or health error")
    started_at: str | None = Field(default=None, description="ISO 8601 UTC")


class _Entry:
    def __init__(self, service: Service) -> None:
        self.service = service
        self.status = ServiceStatus(name=service.name, state="stopped")
        self.next_restart = 0.0


class Supervisor:
    def __init__(self, watch_seconds: float = WATCH_SECONDS) -> None:
        self.entries: list[_Entry] = []
        self.watch_seconds = watch_seconds
        self._watch: asyncio.Task[None] | None = None

    def register(self, service: Service) -> None:
        self.entries.append(_Entry(service))

    async def start(self) -> None:
        for entry in self.entries:
            await self._start(entry)
        self._watch = asyncio.create_task(self._watch_loop(), name="runtime-supervisor")
        logger.info(
            f"runtime_supervisor_001: Started {[e.status.name for e in self.entries]}"
        )

    async def stop(self) -> None:
        if self._watch:
            self._watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch
        for entry in reversed(self.entries):
            with contextlib.suppress(Exception):
                await entry.service.stop()
            entry.status.state = "stopped"
        logger.info("runtime_supervisor_002: Stopped")

    def status(self) -> list[ServiceStatus]:
        return [entry.status.model_copy() for entry in self.entries]

    async def _start(self, entry: _Entry) -> None:
        try:
            await entry.service.start()
        except Exception as exc:
            entry.status.state = "failed"
            entry.status.last_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                f"runtime_supervisor_error_001: {entry.status.name} did not start: {exc}"
            )
            return
        entry.status.state = "running"
        entry.status.started_at = datetime.now(UTC).isoformat()

    async def check(self) -> None:
        """One health round: restart dead services whose backoff has passed."""
        loop_time = asyncio.get_running_loop().time()
        for entry in self.entries:
            if entry.status.state == "running" and entry.service.healthy():
                continue
            if loop_time < entry.next_restart:
                continue
            entry.status.state = "restarting"
            entry.status.last_error = entry.status.last_error or "unhealthy"
            logger.warning(
                f"runtime_supervisor_warning_001: Restarting \033[33m{entry.status.name}\033[0m "
                f"(restart {entry.status.restarts + 1})"
            )
            with contextlib.suppress(Exception):
                await entry.service.stop()
            await self._start(entry)
            entry.status.restarts += 1
            delay = _BACKOFF_SECONDS[min(entry.status.restarts - 1, len(_BACKOFF_SECONDS) - 1)]
            entry.next_restart = loop_time + delay

    async def _watch_loop(self) -> None:
        while True:
            await asyncio.sleep(self.watch_seconds)
            try:
                await self.check()
            except Exception as exc:
                logger.error(f"runtime_supervisor_error_002: Health round failed: {exc}")
