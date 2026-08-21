"""Discovers and loads Archie skills — markdown files with YAML frontmatter under app/skills/.

Mirrors tools_utils.py: skills expose their purpose via frontmatter (name, description)
so the orchestrator can list them cheaply in the prompt, then load the full body on
demand via skill_loader_tool instead of paying the token cost for every skill upfront.
"""

import logging
from pathlib import Path
from typing import Any
import yaml


logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _parse_skill_file(path: Path) -> tuple[dict[str, Any], str] | None:
    """Split a SKILL.md into (frontmatter dict, body). Returns None if malformed."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if not text.startswith("---") or len(parts) < 3:
        logger.warning(f"skill_utils_warning_001: Malformed frontmatter in \033[33m{path}\033[0m")
        return None
    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        logger.error(
            f"skill_utils_error_001: Failed to parse frontmatter in \033[31m{path}\033[0m: {e}"
        )
        return None
    return frontmatter, parts[2].strip()


def list_skills() -> list[dict[str, str]]:
    """List all available skills as {"name", "description"}, sorted by directory name."""
    if not SKILLS_DIR.exists():
        return []

    skills = []
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            continue
        parsed = _parse_skill_file(skill_file)
        if parsed is None:
            continue
        frontmatter, _ = parsed
        name = frontmatter.get("name")
        if not name:
            logger.warning(
                f"skill_utils_warning_002: Missing 'name' in \033[33m{skill_file}\033[0m"
            )
            continue
        description = " ".join(str(frontmatter.get("description", "")).split())
        skills.append({"name": name, "description": description})

    logger.info(f"skill_utils_001: Discovered \033[33m{len(skills)}\033[0m skills")
    return skills


def load_skill(skill_name: str) -> dict[str, str] | None:
    """Load a skill's description and full body content by name. Returns None if not found."""
    skill_file = SKILLS_DIR / skill_name / "SKILL.md"
    if not skill_file.is_file():
        logger.warning(f"skill_utils_warning_003: Skill not found: \033[33m{skill_name}\033[0m")
        return None

    parsed = _parse_skill_file(skill_file)
    if parsed is None:
        return None
    frontmatter, body = parsed

    logger.info(f"skill_utils_002: Loaded skill \033[36m{skill_name}\033[0m")
    return {
        "name": frontmatter.get("name", skill_name),
        "description": " ".join(str(frontmatter.get("description", "")).split()),
        "content": body,
    }
