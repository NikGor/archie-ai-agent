"""Loads full instructions for one Archie skill on demand."""

import logging
from typing import Any
from ..utils.skill_utils import load_skill


logger = logging.getLogger(__name__)


async def skill_loader_tool(skill_name: str) -> dict[str, Any]:
    """
    Loads the full purpose and usage instructions for one skill from the "Available Skills"
    list injected into this prompt. Call this before acting on a request that matches a
    skill's one-line description, so you can follow its detailed guidance (defaults, edge
    cases, disambiguation rules) instead of guessing from the description alone.

    Args:
        skill_name: Exact skill name from the "Available Skills" list, e.g. "homeassistant-mcp"

    Returns:
        dict with the skill's description and full instructions, or an error if not found
    """
    logger.info(f"skill_loader_001: Requested skill: \033[36m{skill_name}\033[0m")
    skill = load_skill(skill_name)
    if skill is None:
        logger.warning(f"skill_loader_warning_001: Skill not found: \033[33m{skill_name}\033[0m")
        return {
            "success": False,
            "error": f"Skill not found: {skill_name}",
        }

    logger.info(f"skill_loader_002: Loaded skill \033[36m{skill_name}\033[0m")
    return {
        "success": True,
        "name": skill["name"],
        "description": skill["description"],
        "instructions": skill["content"],
    }
