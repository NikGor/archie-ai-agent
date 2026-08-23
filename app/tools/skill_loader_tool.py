"""Loads full instructions for one Archie skill on demand."""

import logging
from typing import Any
from ..utils.skill_utils import load_skill, load_skill_references


logger = logging.getLogger(__name__)


async def skill_loader_tool(skill_name: str, references: str | None = None) -> dict[str, Any]:
    """
    Loads the full purpose and usage instructions for one skill from the "Available Skills"
    list injected into this prompt. Call this before acting on a request that matches a
    skill's one-line description, so you can follow its detailed guidance (defaults, edge
    cases, disambiguation rules) instead of guessing from the description alone.

    The response includes "available_references" — extra reference documents (name +
    short description) that exist for this skill but are NOT included in "instructions".
    If one of them matches what you need (e.g. a lookup table, an allowed-values list),
    call this tool again with the same skill_name and references set to load their actual
    content.

    Args:
        skill_name: Exact skill name from the "Available Skills" list, e.g. "homeassistant-mcp"
        references: Comma-separated reference document names to load content for, from a
            previous call's "available_references" (e.g. "CHANNELS" or "CHANNELS,OTHER").
            Omit on the first call.

    Returns:
        dict with the skill's description, full instructions, available_references, and
        (if requested) the loaded reference content — or an error if not found
    """
    logger.info(f"skill_loader_001: Requested skill: \033[36m{skill_name}\033[0m")
    skill = load_skill(skill_name)
    if skill is None:
        logger.warning(f"skill_loader_warning_001: Skill not found: \033[33m{skill_name}\033[0m")
        return {
            "success": False,
            "error": f"Skill not found: {skill_name}",
        }

    result: dict[str, Any] = {
        "success": True,
        "name": skill["name"],
        "description": skill["description"],
        "instructions": skill["content"],
        "available_references": skill["references"],
    }

    if references:
        reference_names = [name.strip() for name in references.split(",") if name.strip()]
        loaded_references = load_skill_references(skill_name, reference_names)
        result["references"] = loaded_references
        missing = [name for name in reference_names if name not in loaded_references]
        if missing:
            result["references_not_found"] = missing
            logger.warning(
                f"skill_loader_warning_002: References not found for \033[33m{skill_name}\033[0m: {missing}"
            )

    logger.info(f"skill_loader_002: Loaded skill \033[36m{skill_name}\033[0m")
    return result
