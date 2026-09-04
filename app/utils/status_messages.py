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


def _smarthome_control_detail(args: dict[str, Any]) -> str:
    action = args.get("action", "")
    target = args.get("device_name", "") or args.get("area", "")
    temp = args.get("temperature", "")
    if action == "set_temperature" and temp:
        return f"Setting temperature: {temp}°"
    if action and target:
        return f"{action}: {target}"
    return f"Smart home: {action}" if action else "Controlling smart home"


def _smarthome_status_detail(args: dict[str, Any]) -> str:
    device_name = args.get("device_name", "")
    area = args.get("area", "")
    target = device_name or area
    return f"Checking status: {target}" if target else "Checking smart home status"


def _document_search_detail(args: dict[str, Any]) -> str:
    query = args.get("query", "")
    return f"Searching documents: {query}" if query else ""


def _tv_detail(args: dict[str, Any]) -> str:
    action = args.get("action", "")
    query = args.get("query", "")
    remote_detail = {
        "volume_up": "Turning TV volume up",
        "volume_down": "Turning TV volume down",
        "mute": "Toggling TV mute",
    }.get(action)
    if remote_detail:
        return remote_detail
    if action == "play_channel":
        return f"Playing TV channel: {query}" if query else "Playing TV channel"
    if action == "play_youtube":
        return "Opening YouTube video"
    if action == "list_channels":
        return f"Searching TV channels: {query}" if query else "Loading TV channels"
    return "Opening media on TV" if action == "play_url" else "Controlling TV"


def _skill_loader_detail(args: dict[str, Any]) -> str:
    skill_name = args.get("skill_name", "")
    return f"Loading skill: {skill_name}" if skill_name else "Loading skill"


def _cron_detail(args: dict[str, Any]) -> str:
    action = args.get("action", "")
    name = args.get("schedule_name", "") or args.get("name", "")
    if action == "create":
        return f"Scheduling: {name}" if name else "Scheduling a task"
    if action == "list":
        return "Loading scheduled tasks"
    return f"Scheduled tasks: {action}" if action else "Managing scheduled tasks"


_TOOL_DETAIL_MAP: dict[str, Any] = {
    "google_search_tool": _google_search_detail,
    "google_places_search_tool": _google_places_detail,
    "events_tool": _events_detail,
    "task_tool": _task_detail,
    "notes_tool": _notes_detail,
    "spotify_tool": _spotify_detail,
    "smarthome_control_tool": _smarthome_control_detail,
    "smarthome_status_tool": _smarthome_status_detail,
    "document_search_tool": _document_search_detail,
    "tv_tool": _tv_detail,
    "skill_loader_tool": _skill_loader_detail,
    "cron_tool": _cron_detail,
}


def get_tool_detail(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Get human-readable detail for a tool call, or None if unavailable."""
    fn = _TOOL_DETAIL_MAP.get(tool_name)
    if fn is None:
        return None
    detail = fn(arguments)
    return detail or None
