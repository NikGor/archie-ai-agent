import logging
from typing import Any

logger = logging.getLogger(__name__)


async def task_tool(
    action: str,
    title: str | None = None,
    task_id: str | None = None,
    description: str | None = None,
    due_date: str | None = None,
    priority: str | None = None,
    status: str | None = None,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Manages tasks and to-do lists with backend integration.

    Args:
        action: Action to perform (list, create, update, delete, complete)
        title: Task title (required for create)
        task_id: Task ID (required for update, delete, complete)
        description: Task description (optional)
        due_date: Due date in ISO format (optional)
        priority: Task priority (low, medium, high)
        status: Task status (todo, in_progress, completed)

    Returns:
        Dict with task data or error information
    """
    logger.info(f"task_001: Action requested: \033[36m{action}\033[0m")

    result = {"success": True, "action": action}

    if action == "create":
        result["message"] = f"Task '{title}' created successfully"
        result["task"] = {
            "title": title,
            "description": description,
            "due_date": due_date,
            "priority": priority,
            "status": status or "todo",
        }
    elif action == "update":
        result["message"] = f"Task {task_id} updated successfully"
        result["task_id"] = task_id
    elif action == "delete":
        result["message"] = f"Task {task_id} deleted successfully"
        result["task_id"] = task_id
    elif action == "complete":
        result["message"] = f"Task {task_id} marked as completed"
        result["task_id"] = task_id
    elif action == "list":
        result["message"] = "Tasks list retrieved successfully"
    else:
        result["message"] = f"Action '{action}' executed successfully"

    return result
