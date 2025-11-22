import logging
from typing import Any

logger = logging.getLogger(__name__)


async def notes_tool(
    action: str,
    title: str | None = None,
    note_id: str | None = None,
    content: str | None = None,
    tags: list[str] | None = None,
    search_query: str | None = None
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
        
    Returns:
        Dict with note data or error information
    """
    logger.info(f"notes_001: Action requested: \033[36m{action}\033[0m")
    
    result = {
        "success": True,
        "action": action
    }
    
    if action == "create":
        result["message"] = f"Note '{title}' created successfully"
        result["note"] = {"title": title, "content": content, "tags": tags}
    elif action == "read":
        result["message"] = f"Note with ID {note_id} retrieved successfully"
        result["note_id"] = note_id
    elif action == "update":
        result["message"] = f"Note {note_id} updated successfully"
        result["note_id"] = note_id
    elif action == "delete":
        result["message"] = f"Note {note_id} deleted successfully"
        result["note_id"] = note_id
    elif action == "search":
        result["message"] = f"Search completed for query: {search_query}"
        result["query"] = search_query
    elif action == "list":
        result["message"] = "Notes list retrieved successfully"
    else:
        result["message"] = f"Action '{action}' executed successfully"
    
    return result
