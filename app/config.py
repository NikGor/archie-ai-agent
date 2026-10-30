"""Configuration constants and centralized settings for the application."""

from typing import Annotated, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


MAX_COMMAND_ITERATIONS = 3

DEFAULT_MODEL = "gpt-6-luna"

# gpt-5.6-luna supports up to 128000 output tokens; verbose UI answers (e.g. multi-month
# weather breakdowns) can exceed the previous 16000 cap and get truncated mid-JSON.
UI_STREAM_MAX_OUTPUT_TOKENS = 32000


class Settings(BaseSettings):
    """Centralized application settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider API keys
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    gemini_api_key: str | None = None

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Persistent scheduled tool execution
    cron_poll_interval: float = 1.0
    cron_max_jobs_per_tick: int = 20

    # Session-scoped tool result cache (ARCHIE-157)
    tool_result_cache_enabled: bool = False
    tool_result_cache_ttl: int = 3600
    tool_result_cache_max_items: int = 20

    # Agent swarm chat (ARCHIE-199)
    swarm_enabled: bool = True
    # embedded: one process; api: chat and /ws_swarm only; runtime: swarm and
    # cron only, events and commands via Redis (ARCHIE-299)
    runtime_mode: Literal["embedded", "api", "runtime"] = "embedded"
    ephemeral_model: str = "gpt-4.1"
    ephemeral_daily_cost_usd: float = Field(default=1.0, ge=0)
    # Model profiles of AgentSpec runs (ARCHIE-322): profile -> ordered
    # "model:effort" candidates; the first priced model with agent support wins.
    # Models for profiles are chosen by the ARCHIE-307 benchmark, not here.
    agent_profiles: dict[str, list[str]] = {
        "FAST": ["gpt-5.6-luna:low", "gpt-4.1"],
        "BALANCED": ["gpt-5.6-luna:medium", "gpt-4.1"],
        "DEEP": ["gpt-5.4:medium", "gpt-4.1"],
        "MAXIMUM": ["gpt-5.4:high"],
    }
    # Profile of the AgentConfigurator call (ARCHIE-323)
    agent_configurator_profile: Literal["FAST", "BALANCED"] = "BALANCED"
    # Daily LLM spend of delegated chat agents (ARCHIE-324)
    agent_daily_cost_usd: float = Field(default=2.0, ge=0)
    # DeepSeek V4.1 Flash broke tool calls and looped to 131k output tokens
    # (ARCHIE-241); GPT-6 Luna via OpenRouter returned several JSON objects in one
    # answer and made up tool results (ARCHIE-243). GPT-5.6 Luna runs on the
    # OpenAI API with strict structured output.
    swarm_model: str = "gpt-5.6-luna"
    # The consul leads the council discussion on a flagship model (ARCHIE-249)
    swarm_consul_model: str = "anthropic/claude-opus-5.5"
    swarm_council_max_rounds: int = 3
    # How long a plan step waits for the agent's reply before the next step
    swarm_step_timeout_seconds: int = 240
    # The council takes one hop itself: user -> decision -> agent -> agent
    swarm_max_depth: int = 4
    swarm_cooldown_seconds: int = 20
    swarm_history_size: int = 12
    swarm_stream_max_length: int = 500
    swarm_tool_result_max_chars: int = 1000
    swarm_max_tool_rounds: int = 3
    # Budget of one swarm agent turn and the daily LLM spend of the swarm (ARCHIE-298)
    swarm_turn_max_cost_usd: float = 0.10
    swarm_turn_max_seconds: float = 180
    swarm_daily_cost_usd: float | None = 3.0
    # A check that follows a live event (a match) runs again after this pause (ARCHIE-286)
    swarm_follow_up_minutes: int = 5
    swarm_user_name: str | None = None
    # Archie web app (Django) as the agent reaches it: alert chats (ARCHIE-247)
    web_django_url: str = "http://archie-web:8000"
    # Dry run: tools with side effects (lights, climate, voice) run in demo mode;
    # state reads stay real. For safe testing against the real home.
    swarm_dry_run: bool = False
    # Swarm journal (ARCHIE-281): every posted message goes to Django
    # {web_django_url}/swarm-journal/api for the daily swarm audit; auth: EVAL_API_TOKEN
    swarm_journal_enabled: bool = True
    swarm_journal_flush_seconds: float = 5.0
    swarm_journal_buffer_size: int = 2000

    # Prompt eval suite (ARCHIE-208)
    # Prompt autofix (ARCHIE-213): cases run on the model a rule failed on, gpt-4.1
    # and gpt-5.6-luna; a fix must pass on gpt-5.6-luna
    eval_default_model: str = "gpt-4.1"
    eval_concurrency: int = 3
    eval_user_name: str = "eval"
    # Django prompt-evals API (.../prompt-evals/api): stores cases and runs
    eval_django_url: str | None = None
    eval_api_token: str | None = None

    # Chat monitoring: LLM judge grades stored conversations (ARCHIE-213)
    # TypeSafe Jev decision model (OpenRouter Decisions API); the alias tracks the newest Jev
    monitor_judge_model: str = "~typesafe/jev-latest"
    auto_response_model: str = "~typesafe/jev-latest"
    auto_response_timeout: float = 5.0
    monitor_concurrency: int = 4
    # Django chat-monitoring API (.../chat-monitoring/api); auth: EVAL_API_TOKEN
    monitor_django_url: str | None = None
    # A turn without a message_id match takes the latest session trace that
    # started at most this long before the stored message
    monitor_trace_match_seconds: int = 300

    # Google integrations
    # OAuth user token shared by Gmail and Calendar (scripts/google_auth_setup.py)
    gmail_token_file: str = "token_gmail.json"
    google_places_api_key: str | None = None
    google_api_key: str | None = None

    # Spotify integration
    spotify_client_id: str | None = None
    spotify_client_secret: str | None = None
    spotify_refresh_token: str | None = None
    spotify_redirect_uri: str = "http://127.0.0.1:8888/callback"

    # Android TV / IPTV integration
    tv_host: str | None = None
    tv_cert_file: str = "~/.config/archie-tv-remote/cert.pem"
    tv_key_file: str = "~/.config/archie-tv-remote/key.pem"
    tv_playlist_path: str = "Sharavoz.Tv.ott (2) - футбол.m3u"
    tv_remote_name: str = "Archie TV Remote"
    tv_connect_timeout: float = 10.0
    sharavoz_token: str | None = None

    # Scheduled agent tasks (agent_task_tool): more Stage 1 rounds than a chat turn
    agent_task_max_iterations: int = 6

    # AWTRIX NG LED display
    awtrix_host: str | None = None
    awtrix_timeout: float = 15.0

    # OpenAI vector store (document search)
    openai_vector_store_id: str | None = None

    # Home Assistant MCP server (smarthome control)
    homeassistant_mcp_url: str = "http://homeassistant.local:8123/api/mcp"
    homeassistant_mcp_token: str | None = None

    # Default persona / locale state
    default_persona: str = "business"
    default_city: str = "London"
    default_country: str = "UK"
    default_timezone: str = "UTC"
    default_language: str = "en"
    default_currency: str = "USD"
    default_holidays: str = "GB"
    default_transport: Annotated[list[str], NoDecode] = ["car", "public_transport"]
    default_cuisine: Annotated[list[str], NoDecode] = ["italian", "asian", "local"]

    @field_validator("default_transport", "default_cuisine", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        """Parse comma-separated env var strings into lists (matches old .split(","))."""
        if isinstance(value, str):
            return value.split(",")
        return value


settings = Settings()

DEFAULT_STATE_CONFIG = {
    "persona": settings.default_persona,
    "default_city": settings.default_city,
    "default_country": settings.default_country,
    "user_timezone": settings.default_timezone,
    "language": settings.default_language,
    "currency": settings.default_currency,
    "commercial_holidays": settings.default_holidays,
    "transport_preferences": settings.default_transport,
    "cuisine_preferences": settings.default_cuisine,
}
