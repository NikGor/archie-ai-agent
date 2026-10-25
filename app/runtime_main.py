"""Runtime process (ARCHIE-299): swarm and cron under the supervisor, without
HTTP. The API process (RUNTIME_MODE=api) serves chat and /ws_swarm; swarm
events and ticker commands travel between the two through Redis.

Run: RUNTIME_MODE=runtime python -m app.runtime_main
"""

import asyncio
import logging
import signal
from dotenv import load_dotenv


load_dotenv()

from app.agents.registry import load_agent_kinds  # noqa: E402
from app.agents.swarm.bridge import SwarmHubBridge  # noqa: E402
from app.agents.swarm.control import apply_command  # noqa: E402
from app.agents.swarm.service import swarm_service  # noqa: E402
from app.backend.cron_scheduler import cron_scheduler  # noqa: E402
from app.config import settings  # noqa: E402
from app.platform.runtime.budget import daily_spend  # noqa: E402
from app.platform.runtime.supervisor import Supervisor  # noqa: E402
from app.runtime_endpoints import RuntimeStatus, publish_status  # noqa: E402
from app.utils.tracing import init_tracing, shutdown_tracing  # noqa: E402


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


async def _report(supervisor: Supervisor, stopping: asyncio.Event) -> None:
    """Status of this process for GET /runtime/status of the API process."""
    while not stopping.is_set():
        try:
            await publish_status(
                RuntimeStatus(
                    services=supervisor.status(),
                    daily_spend_usd={k: round(v, 4) for k, v in daily_spend.by_kind.items()},
                )
            )
        except Exception as exc:
            logger.error(f"runtime_main_error_001: Status not published: {exc}")
        await asyncio.sleep(30)


async def main() -> None:
    if settings.runtime_mode != "runtime":
        raise SystemExit("runtime_main needs RUNTIME_MODE=runtime")
    init_tracing()
    load_agent_kinds()
    supervisor = Supervisor()
    swarm_service.bridge = SwarmHubBridge(
        swarm_service.hub, role="runtime", on_command=apply_command
    )
    # The bridge starts first, so it relays the first swarm events too
    supervisor.register(swarm_service.bridge)
    supervisor.register(cron_scheduler)
    if settings.swarm_enabled:
        supervisor.register(swarm_service)
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopping.set)
    await supervisor.start()
    logger.info("runtime_main_001: Runtime process ready")
    reporter = asyncio.create_task(_report(supervisor, stopping))
    await stopping.wait()
    reporter.cancel()
    await supervisor.stop()
    await swarm_service.close()
    await cron_scheduler.close()
    shutdown_tracing()
    logger.info("runtime_main_002: Runtime process stopped")


if __name__ == "__main__":
    asyncio.run(main())
