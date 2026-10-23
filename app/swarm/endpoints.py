"""Swarm chat endpoints.

/ws_swarm is a long-lived connection (unlike the one-shot /ws_chat): it sends
the history on connect, then streams live events and accepts user commands.
"""

import asyncio
import contextlib
import logging
from archie_shared.swarm.models import (
    SwarmAgentInfo,
    SwarmAgentModelCommand,
    SwarmAgentsEvent,
    SwarmClientCommand,
    SwarmControlEvent,
    SwarmForgetCommand,
    SwarmHistoryEvent,
    SwarmMessage,
    SwarmModeCommand,
    SwarmProposalDecisionCommand,
    SwarmSettingsCommand,
    SwarmToggleAgentCommand,
    SwarmUserMessageCommand,
    SwarmWatchTask,
)
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError
from .agents import SELECTABLE_MODELS
from .service import swarm_service


logger = logging.getLogger(__name__)
router = APIRouter()

# The feed also shows check lines, about one per 10 minutes
_HISTORY_SIZE = 100
_command_adapter: TypeAdapter[SwarmClientCommand] = TypeAdapter(SwarmClientCommand)


@router.get("/swarm/agents")
async def list_swarm_agents() -> list[SwarmAgentInfo]:
    return await swarm_service.roster.list_agents()


@router.post("/swarm/messages")
async def post_swarm_message(command: SwarmUserMessageCommand) -> SwarmMessage:
    """Post a user message to the swarm chat (for Django, HA or scripts)."""
    if await swarm_service.memory.get_mode() == "stopped":
        raise HTTPException(status_code=409, detail="The swarm is stopped")
    return await swarm_service.bus.post(sender="user", kind="user", text=command.text)


@router.get("/swarm/control")
async def get_swarm_control() -> SwarmControlEvent:
    return await swarm_service.ticker.build_control_event()


@router.post("/swarm/checks/{task}")
async def run_swarm_check(task: SwarmWatchTask) -> SwarmControlEvent:
    """Run one spontaneous check now, outside the schedule (for tests and scripts)."""
    await swarm_service.ticker.run_check(task)
    return await swarm_service.ticker.build_control_event()


async def _handle_command(command: SwarmClientCommand) -> None:
    if isinstance(command, SwarmUserMessageCommand):
        if await swarm_service.memory.get_mode() == "stopped":
            raise ValueError("The swarm is stopped. Press Play or Pause first.")
        await swarm_service.bus.post(sender="user", kind="user", text=command.text)
    elif isinstance(command, SwarmModeCommand):
        await swarm_service.ticker.set_mode(command.mode)
    elif isinstance(command, SwarmSettingsCommand):
        await swarm_service.ticker.set_settings(command.settings)
    elif isinstance(command, SwarmForgetCommand):
        await swarm_service.ticker.forget(command.fact_id)
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


async def _send_events(websocket: WebSocket, queue: asyncio.Queue) -> None:
    while True:
        event = await queue.get()
        await websocket.send_json(event.model_dump(mode="json"))


@router.websocket("/ws_swarm")
async def ws_swarm(websocket: WebSocket, user_name: str | None = None) -> None:
    """`user_name` (the logged-in user) becomes the swarm owner: agents use
    that user's UserState, also for runs nobody watches (cron, HA events)."""
    await websocket.accept()
    if user_name:
        await swarm_service.roster.set_owner(user_name)
    queue = swarm_service.hub.subscribe()
    sender = asyncio.create_task(_send_events(websocket, queue))
    logger.info("swarm_ws_001: Client connected")
    try:
        history = SwarmHistoryEvent(
            messages=await swarm_service.bus.get_history(
                _HISTORY_SIZE, with_checks=True
            ),
            agents=await swarm_service.roster.list_agents(),
            models=SELECTABLE_MODELS,
        )
        await websocket.send_json(history.model_dump(mode="json"))
        control = await swarm_service.ticker.build_control_event()
        await websocket.send_json(control.model_dump(mode="json"))
        while True:
            data = await websocket.receive_json()
            try:
                await _handle_command(_command_adapter.validate_python(data))
            except (ValidationError, ValueError) as exc:
                logger.warning(f"swarm_ws_warning_002: Bad command: {exc}")
                await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        logger.info("swarm_ws_002: Client disconnected")
    finally:
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender
        swarm_service.hub.unsubscribe(queue)
