"""Swarm chat endpoints.

/ws_swarm is a long-lived connection (unlike the one-shot /ws_chat): it sends
the history on connect, then streams live events and accepts user commands.
"""

import asyncio
import contextlib
import logging
from archie_shared.swarm.models import (
    SwarmAgentInfo,
    SwarmAgentsEvent,
    SwarmClientCommand,
    SwarmHistoryEvent,
    SwarmMessage,
    SwarmProposalDecisionCommand,
    SwarmToggleAgentCommand,
    SwarmUserMessageCommand,
)
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError
from .service import swarm_service


logger = logging.getLogger(__name__)
router = APIRouter()

_HISTORY_SIZE = 50
_command_adapter: TypeAdapter[SwarmClientCommand] = TypeAdapter(SwarmClientCommand)


@router.get("/swarm/agents")
async def list_swarm_agents() -> list[SwarmAgentInfo]:
    return await swarm_service.roster.list_agents()


@router.post("/swarm/messages")
async def post_swarm_message(command: SwarmUserMessageCommand) -> SwarmMessage:
    """Post a user message to the swarm chat (for Django, HA or scripts)."""
    return await swarm_service.bus.post(sender="user", kind="user", text=command.text)


async def _handle_command(command: SwarmClientCommand) -> None:
    if isinstance(command, SwarmUserMessageCommand):
        await swarm_service.bus.post(sender="user", kind="user", text=command.text)
    elif isinstance(command, SwarmToggleAgentCommand):
        await swarm_service.roster.set_enabled(command.agent_id, command.enabled)
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
async def ws_swarm(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = swarm_service.hub.subscribe()
    sender = asyncio.create_task(_send_events(websocket, queue))
    logger.info("swarm_ws_001: Client connected")
    try:
        history = SwarmHistoryEvent(
            messages=await swarm_service.bus.get_history(_HISTORY_SIZE),
            agents=await swarm_service.roster.list_agents(),
        )
        await websocket.send_json(history.model_dump(mode="json"))
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
