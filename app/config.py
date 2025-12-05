"""Configuration constants for the application."""

MAX_COMMAND_ITERATIONS = 3

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
        # "google/gemini-2.5-pro",
        # "google/gemini-2.5-flash",
        # "google/gemini-2.5-flash-lite",
        # Anthropic Claude - doesn't support JSON schema via OpenRouter (returns <tool_call> tags)
        # "anthropic/claude-opus-4.5",
        # "anthropic/claude-sonnet-4.5",
        # "anthropic/claude-sonnet-4",
        # "anthropic/claude-haiku-4.5",
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

TOOLS_CONFIG = {
    "smarthome": {
        "light_control_tool": "app.tools.light_control_tool",
        "climate_control_tool": "app.tools.climate_control_tool",
        "spotify_tool": "app.tools.spotify_tool",
    },
    "internet_search": {
        "google_search_tool": "app.tools.google_search_tool",
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
