"""Weather tool for getting weather information via Google Search."""

import logging
from typing import Any
from app.tools.google_search_tool import google_search_tool

logger = logging.getLogger(__name__)

QUERY_TEMPLATES = {
    "current": "current weather in {city} today temperature conditions",
    "forecast": "weather forecast {city} {days} days",
    "hourly": "hourly weather forecast {city} today",
}


def _build_query(
    action: str,
    city: str,
    days: int | None = None,
) -> str:
    """Build search query from action and parameters."""
    template = QUERY_TEMPLATES.get(action, "weather in {city}")
    query = template.format(
        city=city,
        days=days or 7,
    )
    return " ".join(query.split())


async def get_weather(
    city: str,
    action: str = "current",
    days: int | None = None,
) -> dict[str, Any]:
    """
    Get weather information for a given city using Google Search.

    Args:
        city: City name for weather query
        action: Action to perform (current, forecast, hourly)
        days: Number of days for forecast (default: 7)

    Returns:
        Dict with weather data or error information
    """
    logger.info(
        f"weather_001: Action requested: \033[36m{action}\033[0m for city: \033[36m{city}\033[0m"
    )
    if action not in QUERY_TEMPLATES:
        logger.warning(f"weather_002: Unknown action: \033[33m{action}\033[0m")
        return {
            "success": False,
            "message": f"Unknown action: {action}. Supported: {list(QUERY_TEMPLATES.keys())}",
            "action": action,
        }
    query = _build_query(action, city, days)
    logger.info(f"weather_003: Built query: \033[36m{query}\033[0m")
    search_result = await google_search_tool(query)
    if not search_result.get("success"):
        logger.error(
            f"weather_error_001: Search failed: \033[31m{search_result.get('message')}\033[0m"
        )
        return {
            "success": False,
            "message": search_result.get("message", "Search failed"),
            "action": action,
            "city": city,
            "query": query,
        }
    logger.info(
        f"weather_004: Search completed, grounded: \033[33m{search_result.get('grounded')}\033[0m"
    )
    return {
        "success": True,
        "action": action,
        "city": city,
        "query": query,
        "answer": search_result.get("answer", ""),
        "grounded": search_result.get("grounded", False),
        "sources": search_result.get("sources", []),
        "search_queries": search_result.get("search_queries", []),
    }
