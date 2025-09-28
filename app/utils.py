"""
Utility functions for the application
"""

import uuid
from datetime import datetime, timezone


def generate_id_with_timestamp(prefix: str) -> str:
    """
    Generate an ID with the pattern: prefix-YYYYMMDDHHMMSS-uuid

    Args:
        prefix: The prefix for the ID (e.g., 'message', 'conversation')

    Returns:
        Generated ID string
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID for brevity
    return f"{prefix}-{timestamp}-{unique_id}"


def generate_message_id() -> str:
    """Generate a message ID with timestamp."""
    return generate_id_with_timestamp("message")


def generate_conversation_id() -> str:
    """Generate a conversation ID with timestamp."""
    return generate_id_with_timestamp("conversation")
