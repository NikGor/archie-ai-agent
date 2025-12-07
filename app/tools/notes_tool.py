import logging
from typing import Any

logger = logging.getLogger(__name__)


async def notes_tool(
    action: str,
    title: str | None = None,
    note_id: str | None = None,
    content: str | None = None,
    tags: list[str] | None = None,
    search_query: str | None = None,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Manages notes and quick memos with backend integration.

    Args:
        action: Action to perform (list, create, read, update, delete, search)
        title: Note title (required for create)
        note_id: Note ID (required for read, update, delete)
        content: Note content (required for create, optional for update)
        tags: List of tags for categorization (optional)
        search_query: Search query for finding notes (required for search)
        demo_mode: If True, return mock data without calling backend

    Returns:
        Dict with note data or error information
    """
    logger.info(f"notes_001: Action requested: \033[36m{action}\033[0m")
    if demo_mode:
        return _demo_response(
            action=action,
            title=title,
            note_id=note_id,
            content=content,
            tags=tags,
            search_query=search_query,
        )
    return {
        "success": False,
        "action": action,
        "error": "Notes backend not configured. Use demo_mode=True for testing.",
    }


def _demo_response(
    action: str,
    title: str | None,
    note_id: str | None,
    content: str | None,
    tags: list[str] | None,
    search_query: str | None,
) -> dict[str, Any]:
    """Returns mock response for demo mode."""
    result: dict[str, Any] = {"success": True, "action": action, "demo_mode": True}
    if action == "create":
        result["message"] = f"Note '{title}' created successfully (demo)"
        result["note"] = {
            "id": "demo_note_123",
            "title": title,
            "content": content,
            "tags": tags or [],
        }
    elif action == "read":
        result["message"] = f"Note {note_id} retrieved (demo)"
        result["note"] = {
            "id": note_id,
            "title": "Demo Note",
            "content": "This is demo content",
            "tags": [],
        }
    elif action == "update":
        result["message"] = f"Note {note_id} updated (demo)"
        result["note"] = {
            "id": note_id,
            "title": title or "Updated Note",
            "content": content,
            "tags": tags or [],
        }
    elif action == "delete":
        result["message"] = f"Note {note_id} deleted (demo)"
    elif action == "search":
        result["message"] = f"Search completed for: {search_query} (demo)"
        result["notes"] = []
        result["count"] = 0
    elif action == "list":
        result["message"] = "Notes list retrieved (demo)"
        result["notes"] = []
        result["count"] = 0
    else:
        result["message"] = f"Action '{action}' executed (demo)"
    return result
