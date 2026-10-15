"""Eval case catalog: loads and validates YAML cases, selects cases for a run."""

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import get_args
import yaml
from archie_shared.evals.models import (
    EvalCaseInfo,
    EvalExpect,
    EvalRunRequest,
    EvalStage,
)
from archie_shared.ui.models import AdvancedAnswerItem
from pydantic import BaseModel, ConfigDict, Field
from app.agent.prompt_builder import PromptBuilder
from app.models.orchestration_sgr import ToolName


logger = logging.getLogger(__name__)

CASES_DIR = Path(__file__).parent / "cases"
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "agent" / "prompts"

# Formats that skip Stage 1 (see AgentFactory.arun)
_DIRECT_OUTPUT_FORMATS = {"dashboard", "widget"}


class CatalogError(ValueError):
    """The case catalog is invalid."""


class _CaseEntry(BaseModel):
    """One case as written in a YAML file; unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable case ID")
    name: str = Field(description="Short case name")
    description: str = Field(min_length=10, description="What the case checks")
    prompts: list[str] = Field(
        default_factory=list, description="Extra prompt templates the case tests"
    )
    tags: list[str] = Field(default_factory=list, description="Selection tags")
    input: str = Field(description="User message")
    response_format: str = Field(default="plain", description="Response format")
    stage: EvalStage = Field(default="full", description="stage1 or full")
    judge: bool = Field(default=False, description="Score with the LLM judge")
    expect: EvalExpect = Field(description="Assertions")


def prompt_stem(template_path: Path) -> str:
    return template_path.stem


def list_prompt_templates() -> dict[str, Path]:
    """Prompt name (template file stem) -> template path."""
    return {prompt_stem(path): path for path in sorted(PROMPTS_DIR.rglob("*.jinja2"))}


def compute_prompt_hashes() -> dict[str, str]:
    """Prompt name -> sha256 of the template source, to tell prompt versions apart."""
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in list_prompt_templates().items()
    }


def infer_prompts(stage: EvalStage, response_format: str) -> list[str]:
    """Prompt templates a case exercises, from its stage and response format."""
    format_prompt = prompt_stem(
        Path(PromptBuilder.format_template_name(response_format))
    )
    if stage == "stage1":
        return ["cmd_prompt"]
    if response_format in _DIRECT_OUTPUT_FORMATS:
        return [format_prompt]
    return ["cmd_prompt", format_prompt]


def _validate_entry(entry: _CaseEntry, known_prompts: set[str]) -> None:
    tool_names = set(get_args(ToolName))
    item_types = set(get_args(AdvancedAnswerItem.model_fields["type"].annotation))
    unknown_tools = set(entry.expect.tools_called or []) - tool_names
    if unknown_tools:
        raise CatalogError(f"{entry.id}: unknown tools {sorted(unknown_tools)}")
    unknown_items = set(entry.expect.expected_item_types or []) - item_types
    if unknown_items:
        raise CatalogError(f"{entry.id}: unknown ui item types {sorted(unknown_items)}")
    unknown_prompts = set(entry.prompts) - known_prompts
    if unknown_prompts:
        raise CatalogError(f"{entry.id}: unknown prompts {sorted(unknown_prompts)}")


def load_catalog(cases_dir: Path = CASES_DIR) -> list[EvalCaseInfo]:
    """Load all cases; raises CatalogError on duplicates or invalid references."""
    known_prompts = set(list_prompt_templates())
    cases: list[EvalCaseInfo] = []
    seen: set[str] = set()
    for path in sorted(cases_dir.glob("*.yaml")):
        suite = path.stem
        for raw in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            entry = _CaseEntry.model_validate(raw)
            if entry.id in seen:
                raise CatalogError(f"{entry.id}: duplicate case id ({path.name})")
            seen.add(entry.id)
            _validate_entry(entry, known_prompts)
            inferred = infer_prompts(entry.stage, entry.response_format)
            prompts = inferred + [p for p in entry.prompts if p not in inferred]
            cases.append(
                EvalCaseInfo(
                    suite=suite,
                    prompts=prompts,
                    **entry.model_dump(exclude={"prompts"}),
                )
            )
    logger.info(f"eval_catalog_001: Loaded \033[33m{len(cases)}\033[0m eval cases")
    return cases


@lru_cache(maxsize=1)
def get_catalog() -> tuple[EvalCaseInfo, ...]:
    return tuple(load_catalog())


def select_cases(
    catalog: tuple[EvalCaseInfo, ...] | list[EvalCaseInfo], request: EvalRunRequest
) -> list[EvalCaseInfo]:
    """Cases matching any selector of the request; all cases if no selector is set."""
    if not (request.case_ids or request.prompts or request.tags or request.suites):
        return list(catalog)
    unknown_ids = set(request.case_ids) - {case.id for case in catalog}
    if unknown_ids:
        raise CatalogError(f"Unknown case ids: {sorted(unknown_ids)}")
    return [
        case
        for case in catalog
        if case.id in request.case_ids
        or case.suite in request.suites
        or set(case.prompts) & set(request.prompts)
        or set(case.tags) & set(request.tags)
    ]
