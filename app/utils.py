"""Utility functions for the application"""

import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def generate_id_with_timestamp(prefix: str) -> str:
    """Generate an ID with the pattern: prefix-YYYYMMDDHHMMSS-uuid"""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    generated_id = f"{prefix}-{timestamp}-{unique_id}"
    logger.info(f"utils_001: generated {prefix} ID: \033[36m{generated_id}\033[0m")
    return generated_id


def generate_message_id() -> str:
    """Generate a message ID with timestamp."""
    return generate_id_with_timestamp("message")


def generate_conversation_id() -> str:
    """Generate a conversation ID with timestamp."""
    return generate_id_with_timestamp("conversation")
