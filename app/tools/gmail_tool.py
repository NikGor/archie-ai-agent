"""Gmail mailbox reading tool (read-only)."""

import logging
from typing import Any
from app.backend.gmail_client import GmailClient


logger = logging.getLogger(__name__)

_gmail_client: GmailClient | None = None


def _get_gmail_client() -> GmailClient:
    """Gets or creates a singleton GmailClient instance."""
    global _gmail_client
    if _gmail_client is None:
        _gmail_client = GmailClient()
    return _gmail_client


async def gmail_tool(
    action: str,
    query: str | None = None,
    message_id: str | None = None,
    max_results: int = 5,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Reads the user's Gmail mailbox (read-only: cannot send, delete or modify emails).
    Use for questions about incoming emails, unread messages, or emails from a person or about a topic.

    Args:
        action: Action to perform: 'search' (list messages matching query; empty query returns latest inbox messages) or 'read' (full text of one message)
        query: Gmail search query, e.g. 'is:unread', 'from:anna@example.com', 'subject:invoice newer_than:7d'
        message_id: Message ID from a previous search result (required for read)
        max_results: Maximum number of messages to return for search (1-20)
        demo_mode: If True, return mock data without calling Gmail API

    Returns:
        Dict with messages (message_id, subject, sender_name, sender_email, snippet, received_at, is_unread, has_attachments, gmail_url) or error information
    """
    logger.info(f"gmail_001: Action requested: \033[36m{action}\033[0m")
    if demo_mode:
        return _demo_response(action)
    try:
        client = _get_gmail_client()
    except FileNotFoundError as error:
        logger.error(f"gmail_error_001: \033[31m{error}\033[0m")
        return {
            "success": False,
            "action": action,
            "error": str(error),
            "message": "Gmail token not configured",
        }
    if action == "search":
        result = await client.search_messages(query=query or "", max_results=max(1, min(max_results, 20)))
    elif action == "read":
        if not message_id:
            return {"success": False, "action": action, "error": "message_id is required for read action"}
        result = await client.get_message(message_id)
    else:
        logger.warning(f"gmail_002: Unknown action: \033[33m{action}\033[0m")
        return {"success": False, "action": action, "error": f"Unknown action: {action}"}
    result["action"] = action
    return result


def _demo_response(action: str) -> dict[str, Any]:
    """Returns mock mailbox data for demo mode."""
    message = {
        "message_id": "demo_001",
        "thread_id": "demo_thread_001",
        "subject": "Welcome to Archie",
        "sender_name": "Archie Team",
        "sender_email": "team@archie.local",
        "snippet": "Your smart home assistant is ready.",
        "received_at": "2026-01-01 09:00",
        "is_unread": True,
        "has_attachments": False,
        "gmail_url": "https://mail.google.com/mail/u/0/#all/demo_001",
    }
    if action == "read":
        return {"success": True, "action": action, "message": {**message, "body": "Your smart home assistant is ready."}}
    return {"success": True, "action": action, "messages": [message], "count": 1}
