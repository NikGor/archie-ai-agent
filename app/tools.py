import logging

from agents import function_tool

logger = logging.getLogger(__name__)


@function_tool
def greet_user(name: str) -> str:
    """Return a short greeting addressing the provided user name."""
    return f"Hello, {name}!"
