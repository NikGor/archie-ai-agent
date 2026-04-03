"""Human-readable status messages for tool calls.

Maps tool names to functions that extract a descriptive detail
from tool call arguments, so StatusUpdate messages are informative
instead of generic.
"""

from typing import Any


def _google_search_detail(args: dict[str, Any]) -> str:
    query = args.get("query", "")
    return f"Google: {query}" if query else ""


def _google_places_detail(args: dict[str, Any]) -> str:
    query = args.get("query", "")
    return f"Google Places: {query}" if query else ""


def _events_detail(args: dict[str, Any]) -> str:  # noqa: PLR0911
    action = args.get("action", "")
    title = args.get("title", "")
    if action == "create" and title:
        return f"Creating event: {title}"
    if action == "today":
        return "Checking today's calendar"
    if action == "upcoming":
        return "Checking upcoming events"
    if action == "list":
        return "Loading events list"
    if action == "delete" and title:
        return f"Deleting event: {title}"
    if action == "update" and title:
        return f"Updating event: {title}"
    return f"Calendar: {action}" if action else ""


def _task_detail(args: dict[str, Any]) -> str:
    action = args.get("action", "")
    title = args.get("title", "")
    if action == "create" and title:
        return f"Creating task: {title}"
    if action == "list":
        return "Loading tasks list"
    if action == "complete" and title:
        return f"Completing task: {title}"
    return f"Tasks: {action}" if action else ""


def _notes_detail(args: dict[str, Any]) -> str:
    action = args.get("action", "")
    title = args.get("title", "")
    search_query = args.get("search_query", "")
    if action == "search" and search_query:
        return f"Searching notes: {search_query}"
    if action == "create" and title:
        return f"Creating note: {title}"
    if action == "list":
        return "Loading notes list"
    return f"Notes: {action}" if action else ""


def _spotify_detail(args: dict[str, Any]) -> str:
    action = args.get("action", "")
    query = args.get("query", "")
    if action == "search" and query:
        return f"Searching Spotify: {query}"
    if action == "play":
        return f"Playing: {query}" if query else "Playing"
    if action == "get_current":
        return "Current track"
    return f"Spotify: {action}" if action else ""


def _light_detail(args: dict[str, Any]) -> str:
    device = args.get("device_name", "")
    return f"Controlling light: {device}" if device else "Controlling light"


def _climate_detail(args: dict[str, Any]) -> str:
    action = args.get("action", "")
    temp = args.get("temperature", "")
    if action == "set_temperature" and temp:
        return f"Setting temperature: {temp}°"
    return f"Climate: {action}" if action else "Controlling climate"


def _football_detail(args: dict[str, Any]) -> str:
    action = args.get("action", "")
    team = args.get("team", "")
    if action == "live_scores":
        return "Checking live scores" + (f": {team}" if team else "")
    if action == "fixtures":
        return "Match schedule" + (f": {team}" if team else "")
    if action == "standings":
        league = args.get("league", "")
        return "Standings" + (f": {league}" if league else "")
    return f"Football: {action}" if action else ""


def _document_search_detail(args: dict[str, Any]) -> str:
    query = args.get("query", "")
    return f"Searching documents: {query}" if query else ""


_TOOL_DETAIL_MAP: dict[str, Any] = {
    "google_search_tool": _google_search_detail,
    "google_places_search_tool": _google_places_detail,
    "events_tool": _events_detail,
    "task_tool": _task_detail,
    "notes_tool": _notes_detail,
    "spotify_tool": _spotify_detail,
    "light_control_tool": _light_detail,
    "climate_control_tool": _climate_detail,
    "football_tool": _football_detail,
    "document_search_tool": _document_search_detail,
}


def get_tool_detail(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Get human-readable detail for a tool call, or None if unavailable."""
    fn = _TOOL_DETAIL_MAP.get(tool_name)
    if fn is None:
        return None
    detail = fn(arguments)
    return detail or None
