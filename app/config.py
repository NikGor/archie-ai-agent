"""Configuration constants for the application."""

import os


MAX_COMMAND_ITERATIONS = 3

DEFAULT_STATE_CONFIG = {
    "persona": os.getenv("DEFAULT_PERSONA", "business"),
    "default_city": os.getenv("DEFAULT_CITY", "London"),
    "default_country": os.getenv("DEFAULT_COUNTRY", "UK"),
    "user_timezone": os.getenv("DEFAULT_TIMEZONE", "UTC"),
    "language": os.getenv("DEFAULT_LANGUAGE", "en"),
    "currency": os.getenv("DEFAULT_CURRENCY", "USD"),
    "commercial_holidays": os.getenv("DEFAULT_HOLIDAYS", "GB"),
    "transport_preferences": os.getenv(
        "DEFAULT_TRANSPORT", "car,public_transport"
    ).split(","),
    "cuisine_preferences": os.getenv("DEFAULT_CUISINE", "italian,asian,local").split(
        ","
    ),
}

MODEL_PROVIDERS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-pro",
        "gpt-5.1",
        "o1",
        "o1-pro",
        "o3",
        "o3-mini",
        "o3-pro",
    ],
    "openrouter": [
        # Google Gemini - schema too complex for AgentResponse (400 Bad Request)
        # Works for DecisionResponse (command stage), but not for final output
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-flash-lite",
        "google/gemini-3-pro-image-preview",
        # Anthropic Claude - doesn't support JSON schema via OpenRouter (returns <tool_call> tags)
        "anthropic/claude-opus-4.5",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-haiku-4.5",
        # xAI Grok - works well with structured outputs
        "x-ai/grok-4",
        "x-ai/grok-4-fast",
        "x-ai/grok-4.1-fast",
        # DeepSeek
        "deepseek/deepseek-v3.2-exp",
        # Meta Llama
        "meta-llama/llama-4-maverick",
        "meta-llama/llama-4-scout",
    ],
    # Fallback - direct Gemini API (commented out)
    # "gemini": [
    #     "gemini-2.5-flash",
    #     "gemini-2.5-pro",
    #     "gemini-2.0-flash",
    #     "gemini-2.0-flash-lite",
    #     "gemini-2.0-pro-exp",
    #     "gemini-2.0-flash-thinking-exp",
    #     "gemini-3-pro-preview",
    # ],
}

# Token prices in USD per 1M tokens: {"input": price, "output": price}
# Sources: official provider pricing pages (2026-02)
# Models not listed return 0.0 from calculate_token_cost() in llm_parser.py
MODEL_TOKEN_PRICES: dict[str, dict[str, float]] = {
    # OpenAI — platform.openai.com/docs/pricing
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-pro": {"input": 15.00, "output": 120.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-pro": {"input": 150.00, "output": 600.00},
    "o3": {"input": 2.00, "output": 8.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o3-pro": {"input": 20.00, "output": 80.00},
    # Google Gemini — ai.google.dev/gemini-api/docs/pricing
    "google/gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "google/gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "google/gemini-3-pro-image-preview": {"input": 2.00, "output": 12.00},
    # Anthropic Claude — platform.claude.com/docs/en/about-claude/pricing
    "anthropic/claude-opus-4.5": {"input": 5.00, "output": 25.00},
    "anthropic/claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "anthropic/claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    # xAI Grok — docs.x.ai/developers/models
    "x-ai/grok-4": {"input": 3.00, "output": 15.00},
    "x-ai/grok-4-fast": {"input": 0.20, "output": 0.50},
    "x-ai/grok-4.1-fast": {"input": 0.20, "output": 0.50},
    # DeepSeek — api-docs.deepseek.com (cache-miss input price)
    "deepseek/deepseek-v3.2-exp": {"input": 0.28, "output": 0.42},
    # Meta Llama — openrouter.ai model pages
    "meta-llama/llama-4-maverick": {"input": 0.15, "output": 0.60},
    "meta-llama/llama-4-scout": {"input": 0.08, "output": 0.30},
}

TOOLS_CONFIG = {
    "smarthome": {
        "light_control_tool": "app.tools.light_control_tool",
        "climate_control_tool": "app.tools.climate_control_tool",
        "spotify_tool": "app.tools.spotify_tool",
    },
    "internet_search": {
        "google_search_tool": "app.tools.google_search_tool",
        "google_places_search_tool": "app.tools.google_places_search_tool",
    },
    "specialized": {
        "football_tool": "app.tools.football_tool",
    },
    "productivity": {
        "task_tool": "app.tools.task_tool",
        "notes_tool": "app.tools.notes_tool",
        "events_tool": "app.tools.events_tool",
    },
    "knowledge": {
        "document_search_tool": "app.tools.document_search_tool",
    },
}
