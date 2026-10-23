"""Configuration constants and centralized settings for the application."""

from typing import Annotated
from pydantic import field_validator
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
    # DeepSeek V4.1 Flash broke tool calls and looped to 131k output tokens
    # (ARCHIE-241); GPT-6 Luna via OpenRouter returned several JSON objects in one
    # answer and made up tool results (ARCHIE-243). GPT-5.6 Luna runs on the
    # OpenAI API with strict structured output.
    swarm_model: str = "gpt-5.6-luna"
    swarm_council_model: str = "gpt-5.6-luna"
    # The council takes one hop itself: user -> decision -> agent -> agent
    swarm_max_depth: int = 4
    swarm_cooldown_seconds: int = 20
    swarm_history_size: int = 12
    swarm_stream_max_length: int = 500
    swarm_tool_result_max_chars: int = 1000
    swarm_max_tool_rounds: int = 3
    swarm_user_name: str | None = None
    # Archie web app (Django) as the agent reaches it: alert chats (ARCHIE-247)
    web_django_url: str = "http://archie-web:8000"
    # Dry run: tools with side effects (lights, climate, voice) run in demo mode;
    # state reads stay real. For safe testing against the real home.
    swarm_dry_run: bool = False

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

MODEL_PROVIDERS = {
    "openai": [
        "gpt-6-luna",
        "gpt-6",
        "gpt-6.1",
        "gpt-5.6-luna",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-5.4",
        "gpt-5.4-pro",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ],
    "openrouter": [
        "google/gemini-3.1-pro-preview",
        "google/gemini-3-flash-preview",
        "google/gemini-3.1-flash-lite-preview",
        "google/gemini-3-pro-image-preview",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-opus-4.5",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-haiku-4.5",
        "openai/gpt-6.1-sol",
        "openai/gpt-6-luna",
        "openai/gpt-6-astra",
        "openai/gpt-5.6-terra",
        "anthropic/claude-haiku-5.5",
        "google/gemini-4-argon",
        "google/gemini-3.8-flash",
        "google/gemini-3.5-flash-lite",
        "x-ai/grok-4.7",
        "deepseek/deepseek-v4.1-flash-1017",
        "deepseek/deepseek-v4-pro-0813",
        "qwen/qwen3.8-max-prime",
        "qwen/qwen3.8-flash",
        "qwen/qwen3.8-27b",
        "z-ai/glm-5.3",
        "z-ai/glm-5.3-flash",
        "moonshotai/kimi-k3",
        "minimax/minimax-m3",
        "mistralai/mistral-large-4-0",
        "meta/muse-spark-open",
        "meta/muse-wonder",
        "stepfun/step-5-preview",
        "reflection/beam-1.0-501b-a23b",
        "elorian/glimpse",
        "inclusionai/ling-3.1-flash",
        "inference-net/schematron-v2-turbo",
        "inference-net/schematron-v2-small",
        "inception/mercury-2.5",
        "nex-agi/nex-n2.5-mini",
        "nex-agi/nex-n2.5-pro",
        "cohere/command-a-plus",
        "upstage/solar-mini4",
        "xiaomi/mimo-v2.6-flash",
        "xiaomi/mimo-v2.6-pro",
        "google/gemma-4-31b-it",
        "google/gemma-4-26b-a4b-it",
        "mistralai/mistral-medium-3-5",
        "tencent/hy4-preview",
        "prism-ml/ternary-bonsai-2-27b",
        "fireworks/ember-1",
        "arcee-ai/trinity-large-thinking",
    ],
}

# Models that accept image input (mirrored in frontend model-catalog.js)
VISION_MODELS: frozenset[str] = frozenset(
    [
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-haiku-5.5",
        "anthropic/claude-opus-4.5",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-sonnet-4.6",
        "cohere/command-a-plus",
        "deepseek/deepseek-v4.1-flash-1017",
        "elorian/glimpse",
        "fireworks/ember-1",
        "google/gemini-3-flash-preview",
        "google/gemini-3.1-flash-lite-preview",
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.5-flash-lite",
        "google/gemini-3.8-flash",
        "google/gemini-4-argon",
        "google/gemma-4-26b-a4b-it",
        "google/gemma-4-31b-it",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.4-pro",
        "gpt-5.6-luna",
        "gpt-6",
        "gpt-6-luna",
        "gpt-6.1",
        "meta/muse-spark-open",
        "meta/muse-wonder",
        "minimax/minimax-m3",
        "mistralai/mistral-large-4-0",
        "mistralai/mistral-medium-3-5",
        "moonshotai/kimi-k3",
        "nex-agi/nex-n2.5-mini",
        "nex-agi/nex-n2.5-pro",
        "openai/gpt-5.6-terra",
        "openai/gpt-6-astra",
        "openai/gpt-6-luna",
        "openai/gpt-6.1-sol",
        "prism-ml/ternary-bonsai-2-27b",
        "qwen/qwen3.8-27b",
        "qwen/qwen3.8-flash",
        "qwen/qwen3.8-max-prime",
        "stepfun/step-5-preview",
        "x-ai/grok-4.7",
        "xiaomi/mimo-v2.6-flash",
        "xiaomi/mimo-v2.6-pro",
        "z-ai/glm-5.3-flash",
    ]
)

# Token prices in USD per 1M tokens: {"input": price, "output": price}
# OpenRouter base rates: https://openrouter.ai/api/v1/models (2026-10-17).
# Native OpenAI rates retain their previous provider-specific values.
# Time-based/provider overrides are preserved in docs/model-catalog; estimates use base rates.
# Models not listed return 0.0 from calculate_token_cost() in llm_parser.py
MODEL_TOKEN_PRICES: dict[str, dict[str, float]] = {
    # OpenAI cut the Luna price by 80% on 30.07.2026
    "gpt-5.6-luna": {"input": 0.2, "output": 1.2},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4.1-nano": {"input": 0.1, "output": 0.4},
    "gpt-5.4": {"input": 2.5, "output": 15.0},
    "gpt-5.4-pro": {"input": 30.0, "output": 180.0},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.5},
    "gpt-5.4-nano": {"input": 0.2, "output": 1.25},
    "google/gemini-3.1-pro-preview": {"input": 2.0, "output": 12.0, "cached": 0.2},
    "google/gemini-3-flash-preview": {"input": 0.5, "output": 3.0, "cached": 0.05},
    "google/gemini-3.1-flash-lite-preview": {
        "input": 0.25,
        "output": 1.5,
        "cached": 0.025,
    },
    "google/gemini-3-pro-image-preview": {"input": 2.0, "output": 12.0, "cached": 0.2},
    "anthropic/claude-opus-4.6": {"input": 5.0, "output": 25.0, "cached": 0.5},
    "anthropic/claude-sonnet-4.6": {"input": 3.0, "output": 15.0, "cached": 0.3},
    "anthropic/claude-opus-4.5": {"input": 5.0, "output": 25.0, "cached": 0.5},
    "anthropic/claude-sonnet-4.5": {"input": 3.0, "output": 15.0, "cached": 0.3},
    "anthropic/claude-haiku-4.5": {"input": 1.0, "output": 5.0, "cached": 0.1},
    "x-ai/grok-4.20-beta": {"input": 2.0, "output": 6.0},
    "x-ai/grok-4.1-fast": {"input": 0.2, "output": 0.5},
    "inference-net/schematron-v2-small": {
        "input": 0.05,
        "output": 0.23,
        "cached": 0.05,
    },
    "meta/muse-spark-open": {"input": 1.25, "output": 4.25, "cached": 0.15},
    "deepseek/deepseek-v4-pro-0813": {"input": 0.66, "output": 1.98, "cached": 0.022},
    "reflection/beam-1.0-501b-a23b": {"input": 1.0, "output": 3.0, "cached": 0.1},
    "qwen/qwen3.8-flash": {"input": 0.15, "output": 0.47, "cached": 0.016},
    "inception/mercury-2.5": {"input": 0.04, "output": 0.15, "cached": 0.004},
    "sakana/fugu-max": {"input": 2.0, "output": 6.0, "cached": 0.25},
    "prism-ml/ternary-bonsai-2-27b": {"input": 0.075, "output": 0.5, "cached": 0.0375},
    "z-ai/glm-5.3": {"input": 0.079, "output": 4.8, "cached": 0.078},
    "nex-agi/nex-n2.5-mini": {"input": 0.025, "output": 0.1, "cached": 0.0025},
    "x-ai/grok-4.7": {"input": 2.0, "output": 6.0, "cached": 0.5},
    "openai/gpt-6-luna": {"input": 0.1, "output": 0.5, "cached": 0.01},
    "anthropic/claude-fable-5.1": {"input": 10.0, "output": 50.0, "cached": 0.25},
    "anthropic/claude-haiku-5.5": {"input": 0.1, "output": 0.5, "cached": 0.01},
    "nvidia/nemotron-3.5-lightning": {"input": 0.07, "output": 0.2, "cached": 0.04},
    "inclusionai/ling-3.1-flash": {"input": 0.075, "output": 0.225, "cached": 0.015},
    "cohere/command-a-plus": {"input": 0.3, "output": 1.5, "cached": 0.15},
    "fireworks/ember-1": {"input": 3.0, "output": 15.0, "cached": 0.3},
    "mistralai/mistral-medium-3-5": {"input": 1.5, "output": 7.5},
    "sakana/fugu-ultra-v2": {"input": 5.0, "output": 30.0, "cached": 0.5},
    "deepseek/deepseek-v4.1-flash-1017": {
        "input": 0.15,
        "output": 0.6,
        "cached": 0.003,
    },
    "qwen/qwen3.8-27b": {"input": 0.24, "output": 2.2, "cached": 0.05},
    "meta/muse-wonder": {"input": 4.0, "output": 20.0, "cached": 0.2},
    "google/gemma-4-31b-it": {"input": 0.2, "output": 0.4},
    "liquid/lfm-2.5-24b-a2b": {"input": 0.05, "output": 0.7, "cached": 0.025},
    "aion-labs/aion-3.5-mini": {"input": 0.7, "output": 1.4, "cached": 0.18},
    "nvidia/nemotron-3.5-super-vl": {"input": 0.1, "output": 0.4, "cached": 0.08},
    "google/gemini-3.8-flash": {"input": 0.825, "output": 4.125, "cached": 0.0825},
    "elorian/glimpse": {"input": 0.6, "output": 3.0, "cached": 0.06},
    "anthropic/claude-sonnet-5.5": {"input": 2.0, "output": 10.0, "cached": 0.1},
    "openai/gpt-5.6-terra": {"input": 2.0, "output": 12.0, "cached": 0.2},
    "upstage/solar-mini4": {"input": 0.05, "output": 0.2, "cached": 0.005},
    "xiaomi/mimo-v2.6-flash": {"input": 0.14, "output": 0.28, "cached": 0.0028},
    "tencent/hy4-preview": {"input": 0.7506, "output": 2.2509, "cached": 0.0378},
    "inference-net/schematron-v2-turbo": {
        "input": 0.03,
        "output": 0.15,
        "cached": 0.03,
    },
    "ibm-granite/granite-4.2-8b": {"input": 0.06, "output": 0.25, "cached": 0.015},
    "arcee-ai/trinity-large-thinking": {"input": 0.25, "output": 0.8, "cached": 0.06},
    "minimax/minimax-m3": {"input": 0.3, "output": 1.2, "cached": 0.06},
    "mistralai/mistral-large-4-0": {"input": 0.68, "output": 2.09, "cached": 0.07},
    "google/gemini-3.5-flash-lite": {"input": 0.3, "output": 2.5, "cached": 0.03},
    "xiaomi/mimo-v2.6-pro": {"input": 0.435, "output": 0.87, "cached": 0.0036},
    "qwen/qwen3.8-max-prime": {"input": 4.0, "output": 12.0, "cached": 0.5},
    "moonshotai/kimi-k3": {"input": 0.304, "output": 11.936, "cached": 0.21},
    "anthropic/claude-opus-5.5": {"input": 4.0, "output": 20.0, "cached": 0.2},
    "nex-agi/nex-n2.5-pro": {"input": 0.075, "output": 0.25, "cached": 0.015},
    "aion-labs/aion-3.5": {"input": 3.0, "output": 6.0, "cached": 0.75},
    "openai/gpt-6.1-sol": {"input": 2.0, "output": 10.0, "cached": 0.1},
    "z-ai/glm-5.3-flash": {"input": 0.15, "output": 0.5, "cached": 0.03},
    "google/gemini-4-argon": {"input": 2.0, "output": 10.0, "cached": 0.1},
    "openai/gpt-6-astra": {"input": 10.0, "output": 50.0, "cached": 1.0},
    "google/gemma-4-26b-a4b-it": {"input": 0.0675, "output": 0.225, "cached": 0.0375},
    "upstage/solar-pro4": {"input": 0.09, "output": 0.36, "cached": 0.018},
    "stepfun/step-5-preview": {"input": 1.0, "output": 2.7, "cached": 0.1},
}

# ARCHIE-227: one smoke per case; see docs/model-catalog/openrouter-2026-10-17.md.
MODEL_STAGE_SUPPORT: dict[str, dict[str, bool]] = {
    "openai/gpt-6.1-sol": {"command": True, "final_plain": True, "final_ui": True},
    "openai/gpt-6-luna": {"command": True, "final_plain": True, "final_ui": True},
    "openai/gpt-6-astra": {"command": True, "final_plain": True, "final_ui": True},
    "openai/gpt-5.6-terra": {"command": True, "final_plain": True, "final_ui": True},
    "anthropic/claude-haiku-5.5": {
        "command": True,
        "final_plain": True,
        "final_ui": False,
    },
    "anthropic/claude-sonnet-5.5": {
        "command": False,
        "final_plain": False,
        "final_ui": False,
    },
    "anthropic/claude-opus-5.5": {
        "command": False,
        "final_plain": False,
        "final_ui": False,
    },
    "anthropic/claude-fable-5.1": {
        "command": False,
        "final_plain": False,
        "final_ui": False,
    },
    "google/gemini-4-argon": {"command": True, "final_plain": True, "final_ui": True},
    "google/gemini-3.8-flash": {
        "command": True,
        "final_plain": True,
        "final_ui": False,
    },
    "google/gemini-3.5-flash-lite": {
        "command": True,
        "final_plain": True,
        "final_ui": False,
    },
    "x-ai/grok-4.7": {"command": True, "final_plain": True, "final_ui": True},
    "deepseek/deepseek-v4.1-flash-1017": {
        "command": True,
        "final_plain": True,
        "final_ui": False,
    },
    "deepseek/deepseek-v4-pro-0813": {
        "command": False,
        "final_plain": True,
        "final_ui": True,
    },
    "qwen/qwen3.8-max-prime": {"command": True, "final_plain": True, "final_ui": True},
    "qwen/qwen3.8-flash": {"command": True, "final_plain": True, "final_ui": False},
    "qwen/qwen3.8-27b": {"command": True, "final_plain": False, "final_ui": False},
    "z-ai/glm-5.3": {"command": True, "final_plain": True, "final_ui": False},
    "z-ai/glm-5.3-flash": {"command": False, "final_plain": True, "final_ui": False},
    "moonshotai/kimi-k3": {"command": True, "final_plain": True, "final_ui": True},
    "minimax/minimax-m3": {"command": True, "final_plain": True, "final_ui": False},
    "mistralai/mistral-large-4-0": {
        "command": True,
        "final_plain": True,
        "final_ui": True,
    },
    "nvidia/nemotron-3.5-super-vl": {
        "command": False,
        "final_plain": False,
        "final_ui": False,
    },
    "nvidia/nemotron-3.5-lightning": {
        "command": False,
        "final_plain": False,
        "final_ui": False,
    },
    "meta/muse-spark-open": {"command": True, "final_plain": True, "final_ui": True},
    "meta/muse-wonder": {"command": False, "final_plain": True, "final_ui": False},
    "stepfun/step-5-preview": {"command": True, "final_plain": True, "final_ui": False},
    "reflection/beam-1.0-501b-a23b": {
        "command": False,
        "final_plain": True,
        "final_ui": False,
    },
    "elorian/glimpse": {"command": False, "final_plain": True, "final_ui": False},
    "liquid/lfm-2.5-24b-a2b": {
        "command": False,
        "final_plain": False,
        "final_ui": False,
    },
    "inclusionai/ling-3.1-flash": {
        "command": False,
        "final_plain": True,
        "final_ui": False,
    },
    "inference-net/schematron-v2-turbo": {
        "command": False,
        "final_plain": True,
        "final_ui": False,
    },
    "inference-net/schematron-v2-small": {
        "command": False,
        "final_plain": True,
        "final_ui": False,
    },
    "inception/mercury-2.5": {"command": True, "final_plain": True, "final_ui": True},
    "nex-agi/nex-n2.5-mini": {"command": False, "final_plain": True, "final_ui": False},
    "nex-agi/nex-n2.5-pro": {"command": False, "final_plain": True, "final_ui": False},
    "cohere/command-a-plus": {"command": False, "final_plain": True, "final_ui": True},
    "upstage/solar-mini4": {"command": False, "final_plain": True, "final_ui": False},
    "xiaomi/mimo-v2.6-flash": {
        "command": False,
        "final_plain": True,
        "final_ui": False,
    },
    "xiaomi/mimo-v2.6-pro": {"command": True, "final_plain": True, "final_ui": False},
    "google/gemma-4-31b-it": {"command": False, "final_plain": True, "final_ui": False},
    "google/gemma-4-26b-a4b-it": {
        "command": False,
        "final_plain": True,
        "final_ui": False,
    },
    "mistralai/mistral-medium-3-5": {
        "command": True,
        "final_plain": True,
        "final_ui": True,
    },
    "upstage/solar-pro4": {"command": False, "final_plain": False, "final_ui": False},
    "ibm-granite/granite-4.2-8b": {
        "command": False,
        "final_plain": False,
        "final_ui": False,
    },
    "sakana/fugu-ultra-v2": {"command": False, "final_plain": False, "final_ui": False},
    "sakana/fugu-max": {"command": False, "final_plain": False, "final_ui": False},
    "tencent/hy4-preview": {"command": False, "final_plain": True, "final_ui": False},
    "prism-ml/ternary-bonsai-2-27b": {
        "command": False,
        "final_plain": True,
        "final_ui": True,
    },
    "fireworks/ember-1": {"command": True, "final_plain": True, "final_ui": True},
    "aion-labs/aion-3.5-mini": {
        "command": False,
        "final_plain": False,
        "final_ui": False,
    },
    "aion-labs/aion-3.5": {"command": False, "final_plain": False, "final_ui": False},
    "arcee-ai/trinity-large-thinking": {
        "command": False,
        "final_plain": True,
        "final_ui": False,
    },
}

TOOLS_CONFIG = {
    "smarthome": {
        "smarthome_control_tool": "app.tools.smarthome_control_tool",
        "smarthome_status_tool": "app.tools.smarthome_status_tool",
        "spotify_tool": "app.tools.spotify_tool",
    },
    "tv": {
        "tv_tool": "app.tools.tv_tool",
    },
    "internet_search": {
        "google_search_tool": "app.tools.google_search_tool",
        "google_places_search_tool": "app.tools.google_places_search_tool",
    },
    "productivity": {
        "task_tool": "app.tools.task_tool",
        "notes_tool": "app.tools.notes_tool",
        "events_tool": "app.tools.events_tool",
        "gmail_tool": "app.tools.gmail_tool",
        "cron_tool": "app.tools.cron_tool",
        "announce_tool": "app.tools.announce_tool",
        "awtrix_tool": "app.tools.awtrix_tool",
        "agent_task_tool": "app.tools.agent_task_tool",
    },
    "knowledge": {
        "document_search_tool": "app.tools.document_search_tool",
    },
    "skills": {
        "skill_loader_tool": "app.tools.skill_loader_tool",
    },
    # Only swarm agents get these tools; the main chat does not (ARCHIE-246)
    "swarm": {
        "weather_tool": "app.tools.weather_tool",
        "alert_chat_tool": "app.tools.alert_chat_tool",
    },
}

# Tool groups the main chat pipeline does not load
SWARM_ONLY_TOOL_GROUPS: frozenset[str] = frozenset({"swarm"})
