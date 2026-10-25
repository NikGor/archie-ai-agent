"""Swarm commands and control state, local or through the runtime process
(ARCHIE-299). In RUNTIME_MODE=api the ticker runs in the runtime process: its
commands go there through the bridge, and the control state comes from Redis."""

import logging
from archie_shared.swarm.models import (
    SwarmAgentModelCommand,
    SwarmAgentsEvent,
    SwarmClientCommand,
    SwarmControlEvent,
    SwarmForgetCommand,
    SwarmModeCommand,
    SwarmProposalDecisionCommand,
    SwarmSettingsCommand,
    SwarmToggleAgentCommand,
    SwarmUserMessageCommand,
    SwarmWatchTask,
)
from app.agents.swarm.bridge import RunCheckCommand, RuntimeCommand
from app.agents.swarm.service import swarm_service
from app.config import settings


logger = logging.getLogger(__name__)

# Commands that need the ticker, which lives in the runtime process in api mode
_TICKER_COMMANDS = (SwarmModeCommand, SwarmSettingsCommand, SwarmForgetCommand, RunCheckCommand)


def _remote() -> bool:
    return settings.runtime_mode == "api" and swarm_service.bridge is not None


async def apply_command(command: RuntimeCommand) -> None:
    """Run a command in this process (embedded mode, or the runtime process)."""
    ticker = swarm_service.ticker
    if isinstance(command, SwarmModeCommand):
        await ticker.set_mode(command.mode)
    elif isinstance(command, SwarmSettingsCommand):
        await ticker.set_settings(command.settings)
    elif isinstance(command, SwarmForgetCommand):
        await ticker.forget(command.fact_id)
    elif isinstance(command, RunCheckCommand):
        await ticker.run_check(command.task)
    else:
        await _apply_shared(command)


async def _apply_shared(command: SwarmClientCommand) -> None:
    """Commands on shared Redis state: any process can run them."""
    if isinstance(command, SwarmUserMessageCommand):
        if await swarm_service.memory.get_mode() == "stopped":
            raise ValueError("The swarm is stopped. Press Play or Pause first.")
        await swarm_service.bus.post(sender="user", kind="user", text=command.text)
    elif isinstance(command, SwarmToggleAgentCommand):
        await swarm_service.roster.set_enabled(command.agent_id, command.enabled)
        swarm_service.hub.publish(
            SwarmAgentsEvent(agents=await swarm_service.roster.list_agents())
        )
    elif isinstance(command, SwarmAgentModelCommand):
        await swarm_service.roster.set_model(command.agent_id, command.model)
        swarm_service.hub.publish(
            SwarmAgentsEvent(agents=await swarm_service.roster.list_agents())
        )
    elif isinstance(command, SwarmProposalDecisionCommand):
        logger.warning("swarm_ws_warning_001: Proposals are not supported yet")


async def handle_command(command: RuntimeCommand) -> None:
    """A command from a client (WebSocket, REST)."""
    if isinstance(command, _TICKER_COMMANDS) and _remote():
        assert swarm_service.bridge is not None
        await swarm_service.bridge.send_command(command)
        return
    await apply_command(command)


async def control_event() -> SwarmControlEvent:
    if not _remote():
        return await swarm_service.ticker.build_control_event()
    assert swarm_service.bridge is not None
    last = await swarm_service.bridge.last_control()
    if last is not None:
        return last
    # The runtime has not published yet: shared state without the next run time
    memory = swarm_service.memory
    return SwarmControlEvent(
        mode=await memory.get_mode(),
        settings=await memory.get_settings(),
        next_run_at=None,
        last_run=await memory.get_last_run(),
        memory=await memory.list_facts(),
        today=[],
    )


async def run_check(task: SwarmWatchTask) -> SwarmControlEvent:
    await handle_command(RunCheckCommand(task=task))
    return await control_event()
