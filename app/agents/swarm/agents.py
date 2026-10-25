"""Swarm agent registry and the Redis-backed roster of awake/asleep agents."""

import logging
from archie_shared.swarm.models import SwarmAgentInfo
from app.agents.swarm.models import ActionRule, SwarmAgentProfile
from app.backend.redis_factory import get_async_redis
from app.config import settings
from app.platform.llm.catalog import MODEL_PROVIDERS


logger = logging.getLogger(__name__)

_DISABLED_KEY = "archie:swarm:disabled"
_OWNER_KEY = "archie:swarm:owner"
_MODELS_KEY = "archie:swarm:models"

# Agents that read the whole home at once need the full state
WHOLE_HOME_MAX_CHARS = 16000

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
            # Fixtures of several teams do not fit the default limit (ARCHIE-250)
            tool_result_max_chars=4000,
            icon="trophy",
            color="#2f9e44",
        ),
        SwarmAgentProfile(
            agent_id="search",
            display_name="Search",
            role="state",
            description="Web search: news, places, opening hours, general facts",
            tools=["google_search_tool", "google_places_search_tool"],
            icon="search",
            color="#15aabf",
        ),
        SwarmAgentProfile(
            agent_id="home",
            display_name="Home",
            role="state",
            description="Watches the whole home: windows, air, temperature, devices left on; does not control devices",
            tools=["smarthome_status_tool"],
            references={"homeassistant-mcp": ["apartment_map"]},
            # The whole home state is about 11k characters (ARCHIE-238)
            tool_result_max_chars=WHOLE_HOME_MAX_CHARS,
            icon="house",
            color="#868e96",
        ),
        SwarmAgentProfile(
            agent_id="mail",
            display_name="Mail",
            role="state",
            description="Reads the user's Gmail inbox; cannot send or delete mail",
            tools=["gmail_tool"],
            icon="mail",
            color="#e64980",
        ),
        SwarmAgentProfile(
            agent_id="calendar",
            display_name="Calendar",
            role="state",
            description="Reads the user's Google Calendar: today's and upcoming events; cannot change them",
            tools=["events_tool"],
            icon="calendar",
            color="#5c7cfa",
        ),
        SwarmAgentProfile(
            agent_id="weather",
            display_name="Weather",
            role="state",
            description="Current weather and forecast for the user's city (Open-Meteo)",
            tools=["weather_tool"],
            tool_result_max_chars=4000,
            icon="cloud-sun",
            color="#4dabf7",
        ),
        SwarmAgentProfile(
            agent_id="voice",
            display_name="Voice",
            role="action",
            description="Speaks events aloud on the home speaker; does not control devices or volume",
            tools=["smarthome_status_tool", "announce_tool"],
            auto_actions=[ActionRule(tool_name="announce_tool")],
            references={"homeassistant-mcp": ["apartment_map"]},
            tool_result_max_chars=WHOLE_HOME_MAX_CHARS,
            icon="megaphone",
            color="#7950f2",
        ),
        SwarmAgentProfile(
            agent_id="display",
            display_name="Display",
            role="action",
            description="Shows short notices on the LED display at home; does not control devices",
            tools=["awtrix_tool"],
            auto_actions=[
                ActionRule(
                    tool_name="awtrix_tool", allowed_values={"action": ["notify"]}
                )
            ],
            references={"awtrix": ["ICONS"]},
            icon="monitor",
            color="#fa5252",
        ),
        SwarmAgentProfile(
            agent_id="chat",
            display_name="Chats",
            role="action",
            description="Opens a new Archie chat about an event; there Archie offers next steps with cards, forms and buttons. For events where the user must choose, answer or confirm",
            tools=["alert_chat_tool"],
            auto_actions=[ActionRule(tool_name="alert_chat_tool")],
            icon="message-square-plus",
            color="#12b886",
        ),
        SwarmAgentProfile(
            agent_id="cron",
            display_name="Cron",
            role="action",
            description="Plans swarm alerts for an exact time (reminders before matches, events, deadlines); at that time the alert comes back to the swarm",
            tools=["cron_tool"],
            auto_actions=[
                # The swarm plans only its own alerts, no direct device actions (ARCHIE-248)
                ActionRule(
                    tool_name="cron_tool",
                    allowed_values={
                        "action": ["create"],
                        "tool_name": ["swarm_alert_tool"],
                    },
                ),
                ActionRule(
                    tool_name="cron_tool",
                    allowed_values={"action": ["pause", "resume", "delete"]},
                ),
            ],
            icon="alarm-clock",
            color="#fab005",
        ),
        SwarmAgentProfile(
            agent_id="butler",
            display_name="Butler",
            role="state",
            description="Consul of the council: leads the discussion, then gives tasks to home agents one by one",
            tools=[],
            persona="butler",
            groups=["council"],
            chair=True,
            model=settings.swarm_consul_model,
            icon="concierge-bell",
            color="#c9a227",
        ),
        # Every member thinks on a model of another provider (ARCHIE-249)
        *(
            SwarmAgentProfile(
                agent_id=persona,
                display_name=display_name,
                role="state",
                description=description,
                tools=[],
                persona=persona,
                groups=["council"],
                model=model,
                icon=icon,
                color=color,
            )
            for persona, display_name, description, model, icon, color in [
                (
                    "business",
                    "Business",
                    "Council member: short, practical, professional view",
                    "gpt-6-luna",
                    "briefcase",
                    "#4dabf7",
                ),
                (
                    "bro",
                    "Bro",
                    "Council member: relaxed, friendly street-smart view",
                    "x-ai/grok-4.7",
                    "hand-metal",
                    "#20c997",
                ),
                (
                    "flirty",
                    "Flirty",
                    "Council member: warm, playful view on comfort and mood",
                    "google/gemini-3.8-flash",
                    "heart",
                    "#f06595",
                ),
                (
                    "marcus",
                    "Marcus",
                    "Council member: Swabian expert on German laws, customs and mentality",
                    "anthropic/claude-sonnet-5.5",
                    "beer",
                    "#e0a526",
                ),
            ]
        ),
    ]
}

# Output agent -> agent that always delivers with it: what the voice says and
# what an alert chat is about is also shown on the display (ARCHIE-247)
PAIRED_OUTPUTS: dict[str, str] = {"voice": "display", "chat": "display"}


def add_paired_outputs(mentions: list[str]) -> list[str]:
    """Mentions plus the paired output of every mentioned output agent."""
    result = list(mentions)
    for agent_id in mentions:
        paired = PAIRED_OUTPUTS.get(agent_id)
        if paired and paired not in result:
            result.insert(result.index(agent_id) + 1, paired)
    return result


SWARM_GROUPS: dict[str, str] = {
    "council": "Main contact of the user: discusses, then the consul gives tasks to home agents",
}

# Every model an agent can use: the user picks one per agent (ARCHIE-249)
SELECTABLE_MODELS: list[str] = list(
    dict.fromkeys(m for models in MODEL_PROVIDERS.values() for m in models)
)


def get_group_chair(group: str) -> str | None:
    return next((a for a in get_group_members(group) if SWARM_AGENTS[a].chair), None)


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
        disabled = await self.redis.smembers(_DISABLED_KEY)  # type: ignore[misc]
        return set(SWARM_AGENTS) - set(disabled)

    async def list_agents(self) -> list[SwarmAgentInfo]:
        enabled = await self.get_enabled_ids()
        models = await self.redis.hgetall(_MODELS_KEY)  # type: ignore[misc]
        return [
            profile.to_info(
                agent_id in enabled, models.get(agent_id) or profile.default_model
            )
            for agent_id, profile in SWARM_AGENTS.items()
        ]

    async def get_model(self, profile: SwarmAgentProfile) -> str:
        """The model the user chose for the agent, else its default model."""
        chosen = await self.redis.hget(_MODELS_KEY, profile.agent_id)  # type: ignore[misc]
        return chosen or profile.default_model

    async def set_model(self, agent_id: str, model: str | None) -> None:
        """Choose the agent's model; None or the default model clears the choice."""
        if agent_id not in SWARM_AGENTS:
            raise ValueError(f"Unknown swarm agent: {agent_id}")
        if model and model not in SELECTABLE_MODELS:
            raise ValueError(f"Unknown model: {model}")
        if not model or model == SWARM_AGENTS[agent_id].default_model:
            await self.redis.hdel(_MODELS_KEY, agent_id)  # type: ignore[misc]
        else:
            await self.redis.hset(_MODELS_KEY, agent_id, model)  # type: ignore[misc]
        logger.info(
            f"swarm_agents_003: Agent \033[36m{agent_id}\033[0m model=\033[35m{model}\033[0m"
        )

    async def set_enabled(self, agent_id: str, enabled: bool) -> None:
        if agent_id not in SWARM_AGENTS:
            raise ValueError(f"Unknown swarm agent: {agent_id}")
        if enabled:
            await self.redis.srem(_DISABLED_KEY, agent_id)  # type: ignore[misc]
        else:
            await self.redis.sadd(_DISABLED_KEY, agent_id)  # type: ignore[misc]
        logger.info(
            f"swarm_agents_001: Agent \033[36m{agent_id}\033[0m enabled=\033[35m{enabled}\033[0m"
        )

    async def get_owner(self) -> str | None:
        return await self.redis.get(_OWNER_KEY)

    async def set_owner(self, user_name: str) -> None:
        if await self.redis.getset(_OWNER_KEY, user_name) != user_name:
            logger.info(f"swarm_agents_002: Owner is \033[35m{user_name}\033[0m")
