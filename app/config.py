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
    "gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-pro-exp",
        "gemini-2.0-flash-thinking-exp",
        "gemini-3-pro-preview",
    ],
    "oss": [
        "Qwen/Qwen3-4B-Instruct-2507",
        "llama-3-70b",
        "llama-3-13b",
    ],
}

TOOLS_CONFIG = {
    "smarthome": {
        "light_control_tool": "app.tools.light_control_tool",
        "climate_control_tool": "app.tools.climate_control_tool",
        "youtube_music_tool": "app.tools.youtube_music_tool",
    },
    "internet_search": {
        "google_search_tool": "app.tools.google_search_tool",
    },
    "specialized": {
        "football_tool": "app.tools.football_tool",
        "weather_tool": "app.tools.weather_tool",
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
