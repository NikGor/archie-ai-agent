"""Google Calendar events management tool."""

import logging
import os
from typing import Any

from app.backend.google_calendar_client import GoogleCalendarClient

logger = logging.getLogger(__name__)

_calendar_client: GoogleCalendarClient | None = None


def _get_calendar_client() -> GoogleCalendarClient:
    """Gets or creates a singleton GoogleCalendarClient instance."""
    global _calendar_client
    if _calendar_client is None:
        _calendar_client = GoogleCalendarClient()
    return _calendar_client


async def events_tool(
    action: str,
    title: str | None = None,
    event_id: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    location: str | None = None,
    description: str | None = None,
    attendees: list[str] | None = None,
    date: str | None = None,
    calendar_id: str = "primary",
    max_results: int = 10,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Manages calendar events and appointments via Google Calendar API.

    Args:
        action: Action to perform (list, create, read, update, delete, today, upcoming)
        title: Event title (required for create)
        event_id: Event ID (required for read, update, delete)
        start_time: Event start time in ISO format (required for create)
        end_time: Event end time in ISO format (required for create)
        location: Event location (optional)
        description: Event description (optional)
        attendees: List of attendee emails (optional)
        date: Specific date for filtering events (YYYY-MM-DD format)
        calendar_id: Calendar ID or 'primary' for default calendar
        max_results: Maximum number of events to return for list/upcoming actions
        demo_mode: If True, return mock data without calling Google API

    Returns:
        Dict with event data or error information
    """
    logger.info(f"events_001: Action requested: \033[36m{action}\033[0m")
    if demo_mode:
        return _demo_response(action, title, event_id, date)
    try:
        client = _get_calendar_client()
    except FileNotFoundError as error:
        logger.error(f"events_error_001: \033[31m{error}\033[0m")
        return {
            "success": False,
            "action": action,
            "error": str(error),
            "message": "Google Calendar credentials not configured",
        }
    action_handlers = {
        "create": lambda: _handle_create(
            client=client,
            calendar_id=calendar_id,
            title=title,
            start_time=start_time,
            end_time=end_time,
            location=location,
            description=description,
            attendees=attendees,
        ),
        "read": lambda: _handle_read(
            client=client,
            calendar_id=calendar_id,
            event_id=event_id,
        ),
        "update": lambda: _handle_update(
            client=client,
            calendar_id=calendar_id,
            event_id=event_id,
            title=title,
            start_time=start_time,
            end_time=end_time,
            location=location,
            description=description,
            attendees=attendees,
        ),
        "delete": lambda: _handle_delete(
            client=client,
            calendar_id=calendar_id,
            event_id=event_id,
        ),
        "today": lambda: client.get_today_events(calendar_id=calendar_id),
        "upcoming": lambda: client.list_events(
            calendar_id=calendar_id,
            max_results=max_results,
        ),
        "list": lambda: _handle_list(
            client=client,
            calendar_id=calendar_id,
            date=date,
            max_results=max_results,
        ),
    }
    handler = action_handlers.get(action)
    if handler is None:
        logger.warning(f"events_002: Unknown action: \033[33m{action}\033[0m")
        return {
            "success": False,
            "action": action,
            "error": f"Unknown action: {action}",
        }
    result = await handler()
    result["action"] = action
    return result


async def _handle_create(
    client: GoogleCalendarClient,
    calendar_id: str,
    title: str | None,
    start_time: str | None,
    end_time: str | None,
    location: str | None,
    description: str | None,
    attendees: list[str] | None,
) -> dict[str, Any]:
    """Handles event creation."""
    if not title:
        return {"success": False, "error": "Title is required for create action"}
    if not start_time:
        return {"success": False, "error": "Start time is required for create action"}
    if not end_time:
        return {"success": False, "error": "End time is required for create action"}
    return await client.create_event(
        summary=title,
        start_time=start_time,
        end_time=end_time,
        calendar_id=calendar_id,
        description=description,
        location=location,
        attendees=attendees,
    )


async def _handle_read(
    client: GoogleCalendarClient,
    calendar_id: str,
    event_id: str | None,
) -> dict[str, Any]:
    """Handles reading a single event."""
    if not event_id:
        return {"success": False, "error": "Event ID is required for read action"}
    return await client.get_event(
        event_id=event_id,
        calendar_id=calendar_id,
    )


async def _handle_update(
    client: GoogleCalendarClient,
    calendar_id: str,
    event_id: str | None,
    title: str | None,
    start_time: str | None,
    end_time: str | None,
    location: str | None,
    description: str | None,
    attendees: list[str] | None,
) -> dict[str, Any]:
    """Handles event update."""
    if not event_id:
        return {"success": False, "error": "Event ID is required for update action"}
    return await client.update_event(
        event_id=event_id,
        calendar_id=calendar_id,
        summary=title,
        start_time=start_time,
        end_time=end_time,
        description=description,
        location=location,
        attendees=attendees,
    )


async def _handle_delete(
    client: GoogleCalendarClient,
    calendar_id: str,
    event_id: str | None,
) -> dict[str, Any]:
    """Handles event deletion."""
    if not event_id:
        return {"success": False, "error": "Event ID is required for delete action"}
    return await client.delete_event(
        event_id=event_id,
        calendar_id=calendar_id,
    )


async def _handle_list(
    client: GoogleCalendarClient,
    calendar_id: str,
    date: str | None,
    max_results: int,
) -> dict[str, Any]:
    """Handles listing events."""
    if date:
        return await client.get_events_for_date(
            date=date,
            calendar_id=calendar_id,
        )
    return await client.list_events(
        calendar_id=calendar_id,
        max_results=max_results,
    )


def _demo_response(
    action: str,
    title: str | None,
    event_id: str | None,
    date: str | None,
) -> dict[str, Any]:
    """Returns mock response for demo mode."""
    result: dict[str, Any] = {"success": True, "action": action, "demo_mode": True}
    if action == "create":
        result["message"] = f"Event '{title}' created successfully (demo)"
        result["event"] = {
            "id": "demo_event_123",
            "title": title,
            "status": "confirmed",
        }
    elif action == "read":
        result["message"] = f"Event {event_id} retrieved (demo)"
        result["event"] = {
            "id": event_id,
            "title": "Demo Event",
            "status": "confirmed",
        }
    elif action == "update":
        result["message"] = f"Event {event_id} updated (demo)"
    elif action == "delete":
        result["message"] = f"Event {event_id} deleted (demo)"
    elif action == "today":
        result["message"] = "Today's events retrieved (demo)"
        result["events"] = []
        result["count"] = 0
    elif action == "upcoming":
        result["message"] = "Upcoming events retrieved (demo)"
        result["events"] = []
        result["count"] = 0
    elif action == "list":
        result["message"] = f"Events for {date or 'all'} retrieved (demo)"
        result["events"] = []
        result["count"] = 0
    else:
        result["message"] = f"Action '{action}' executed (demo)"
    return result
