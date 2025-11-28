import logging
from typing import Any
from app.tools.google_search_tool import google_search_tool

logger = logging.getLogger(__name__)

QUERY_TEMPLATES = {
    "live_scores": "live football scores {league} {team} today",
    "fixtures": "football fixtures {team} {league} {date}",
    "standings": "football league standings {league} table current season",
    "team_info": "football team {team} statistics roster recent results",
}


def _build_query(
    action: str,
    team: str | None = None,
    league: str | None = None,
    date: str | None = None,
) -> str:
    """Build search query from action and parameters."""
    template = QUERY_TEMPLATES.get(action, "football {team} {league} {date}")
    query = template.format(
        team=team or "",
        league=league or "",
        date=date or "",
    )
    return " ".join(query.split())


async def football_tool(
    action: str,
    team: str | None = None,
    league: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    """
    Provides football (soccer) information including live scores, fixtures, standings, and team statistics.

    Args:
        action: Action to perform (live_scores, fixtures, standings, team_info)
        team: Team name for team-specific queries
        league: League name or ID for league-specific queries
        date: Date for fixtures (YYYY-MM-DD format)

    Returns:
        Dict with football data or error information
    """
    logger.info(f"football_001: Action requested: \033[36m{action}\033[0m")
    if action not in QUERY_TEMPLATES:
        logger.warning(f"football_002: Unknown action: \033[33m{action}\033[0m")
        return {
            "success": False,
            "message": f"Unknown action: {action}. Supported: {list(QUERY_TEMPLATES.keys())}",
            "action": action,
        }
    query = _build_query(action, team, league, date)
    logger.info(f"football_003: Built query: \033[36m{query}\033[0m")
    search_result = await google_search_tool(query)
    if not search_result.get("success"):
        logger.error(
            f"football_error_001: Search failed: \033[31m{search_result.get('message')}\033[0m"
        )
        return {
            "success": False,
            "message": search_result.get("message", "Search failed"),
            "action": action,
            "query": query,
        }
    logger.info(
        f"football_004: Search completed, grounded: \033[33m{search_result.get('grounded')}\033[0m"
    )
    return {
        "success": True,
        "action": action,
        "query": query,
        "answer": search_result.get("answer", ""),
        "grounded": search_result.get("grounded", False),
        "sources": search_result.get("sources", []),
        "search_queries": search_result.get("search_queries", []),
    }
