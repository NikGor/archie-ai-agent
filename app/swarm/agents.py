"""Swarm agent registry and the Redis-backed roster of awake/asleep agents."""

import logging
from archie_shared.swarm.models import SwarmAgentInfo
from ..backend.redis_factory import get_async_redis
from ..config import settings
from .models import ActionRule, SwarmAgentProfile


logger = logging.getLogger(__name__)

_DISABLED_KEY = "archie:swarm:disabled"
_OWNER_KEY = "archie:swarm:owner"

SWARM_AGENTS: dict[str, SwarmAgentProfile] = {
    profile.agent_id: profile
    for profile in [
        SwarmAgentProfile(
            agent_id="light",
            display_name="Light",
            role="action",
            description="Lights and scenes in the home",
            tools=["smarthome_status_tool", "smarthome_control_tool"],
            references={"homeassistant-mcp": ["apartment_map"]},
            auto_actions=[
                ActionRule(
                    tool_name="smarthome_control_tool",
                    allowed_values={
                        "action": ["turn_on", "turn_off", "set_light", "activate_scene"]
                    },
                )
            ],
            icon="lightbulb",
            color="#f5b301",
        ),
        SwarmAgentProfile(
            agent_id="climate",
            display_name="Climate",
            role="action",
            description="Heating, thermostats, temperature and humidity",
            tools=["smarthome_status_tool", "smarthome_control_tool"],
            references={"homeassistant-mcp": ["apartment_map"]},
            auto_actions=[
                ActionRule(
                    tool_name="smarthome_control_tool",
                    allowed_values={
                        "action": [
                            "set_temperature",
                            "increase_temperature",
                            "decrease_temperature",
                            "turn_on",
                            "turn_off",
                        ]
                    },
                )
            ],
            icon="thermometer",
            color="#e8590c",
        ),
        SwarmAgentProfile(
            agent_id="football",
            display_name="Football",
            role="state",
            description="Matches, results and news of the user's favourite teams",
            tools=["google_search_tool"],
            skill="football",
            icon="trophy",
            color="#2f9e44",
        ),
        SwarmAgentProfile(
            agent_id="search",
            display_name="Search",
            role="state",
            description="Web search: weather forecast, news, places, opening hours, general facts",
            tools=["google_search_tool", "google_places_search_tool"],
            icon="search",
            color="#15aabf",
        ),
        SwarmAgentProfile(
            agent_id="voice",
            display_name="Voice",
            role="action",
            description="Speaks events aloud on the home speaker; does not control devices or volume",
            tools=["smarthome_status_tool", "announce_tool"],
            auto_actions=[ActionRule(tool_name="announce_tool")],
            references={"homeassistant-mcp": ["apartment_map"]},
            listens_to=["light", "climate", "football"],
            icon="megaphone",
            color="#7950f2",
        ),
        *(
            SwarmAgentProfile(
                agent_id=persona,
                display_name=display_name,
                role="state",
                description=f"Council member with the {persona} persona",
                tools=[],
                persona=persona,
                groups=["council"],
                model=settings.swarm_council_model,
                icon=icon,
                color=color,
            )
            for persona, display_name, icon, color in [
                ("bro", "Bro", "beer", "#20c997"),
                ("business", "Business", "briefcase", "#4dabf7"),
                ("flirty", "Flirty", "heart", "#f06595"),
            ]
        ),
    ]
}

SWARM_GROUPS: dict[str, str] = {
    "council": "Main contact of the user: decides and delegates work to home agents",
}


def get_group_members(group: str) -> list[str]:
    return [
        agent_id
        for agent_id, profile in SWARM_AGENTS.items()
        if group in profile.groups
    ]


class SwarmRoster:
    """Keeps track of which agents are awake (disabled IDs in a Redis set) and
    of the swarm owner: the user whose UserState the agents use."""

    def __init__(self) -> None:
        self.redis = get_async_redis()

    async def get_enabled_ids(self) -> set[str]:
        disabled = await self.redis.smembers(_DISABLED_KEY)
        return set(SWARM_AGENTS) - set(disabled)

    async def list_agents(self) -> list[SwarmAgentInfo]:
        enabled = await self.get_enabled_ids()
        return [
            profile.to_info(agent_id in enabled)
            for agent_id, profile in SWARM_AGENTS.items()
        ]

    async def set_enabled(self, agent_id: str, enabled: bool) -> None:
        if agent_id not in SWARM_AGENTS:
            raise ValueError(f"Unknown swarm agent: {agent_id}")
        if enabled:
            await self.redis.srem(_DISABLED_KEY, agent_id)
        else:
            await self.redis.sadd(_DISABLED_KEY, agent_id)
        logger.info(
            f"swarm_agents_001: Agent \033[36m{agent_id}\033[0m enabled=\033[35m{enabled}\033[0m"
        )

    async def get_owner(self) -> str | None:
        return await self.redis.get(_OWNER_KEY)

    async def set_owner(self, user_name: str) -> None:
        if await self.redis.getset(_OWNER_KEY, user_name) != user_name:
            logger.info(f"swarm_agents_002: Owner is \033[35m{user_name}\033[0m")
