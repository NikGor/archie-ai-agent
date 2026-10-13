"""Gate: picks agents for a message that has no @mentions.

SwarmGate is the seam for the Jev (TypeSafe) gate planned in stage 2;
LlmGate is the default cheap-LLM implementation.
"""

import logging
from typing import Protocol
from archie_shared.swarm.models import SwarmMessage
from ..backend.llm.registry import get_client
from ..config import settings
from ..utils.llm_parser import parse_llm_response
from ..utils.provider_utils import get_provider_for_model
from .models import SwarmAgentProfile, SwarmRoute
from .runner import create_swarm_prompt_env


logger = logging.getLogger(__name__)


class SwarmGate(Protocol):
    async def route(
        self, message: SwarmMessage, agents: list[SwarmAgentProfile]
    ) -> list[str]: ...


class LlmGate:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.swarm_model
        self.env = create_swarm_prompt_env()

    async def route(
        self, message: SwarmMessage, agents: list[SwarmAgentProfile]
    ) -> list[str]:
        if not agents:
            return []
        provider = get_provider_for_model(self.model)
        system_prompt = self.env.get_template("swarm_gate.jinja2").render(agents=agents)
        raw = await get_client(provider).create_completion(
            messages=[
                {"role": "system", "content": system_prompt, "cacheable": True},
                {"role": "user", "content": message.text},
            ],
            model=self.model,
            response_format=SwarmRoute,
        )
        route: SwarmRoute = parse_llm_response(
            raw_response=raw, provider=provider, expected_type=SwarmRoute
        ).parsed_content
        known = {agent.agent_id for agent in agents}
        selected = [agent_id for agent_id in route.agent_ids if agent_id in known]
        logger.info(
            f"swarm_gate_001: Routed to \033[36m{selected}\033[0m: \033[33m{route.reasoning}\033[0m"
        )
        return selected
