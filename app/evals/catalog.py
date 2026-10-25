"""Eval case catalog: validates cases, fetches the catalog from Django, selects cases.

Django stores the cases (ARCHIE-208); the agent owns the rules a case must follow
(known tools, ui item types, prompts) and infers which prompts a case tests.
"""

import logging
from typing import Literal, get_args, get_origin
import httpx
from archie_shared.chat.models import ChatRequest
from archie_shared.evals.models import (
    EvalAgentKind,
    EvalCaseInfo,
    EvalExpect,
    EvalRunRequest,
    EvalStage,
    SwarmEvalInput,
)
from archie_shared.ui.models import AdvancedAnswerItem
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from app.agents.chat.formats import resolve_format
from app.agents.chat.messages import ChatPromptBuilder
from app.agents.chat.quality.rules import ASSISTANT_SPEC, COMMAND_SPEC, rules_for_turn
from app.agents.registry import load_agent_kinds
from app.agents.swarm.agents import SWARM_AGENTS
from app.agents.swarm.quality.rules import case_rules, case_specs
from app.config import settings
from app.platform.prompts.compiler import PromptCompiler
from app.platform.quality.adapter import UnknownAgentKindError, quality_adapter
from app.platform.tools.models import ToolName


logger = logging.getLogger(__name__)

CASE_ID_PATTERN = r"^[a-z0-9_]+(\.[a-z0-9_]+)+$"

# Formats that skip Stage 1 (see ChatPipeline.arun)


class CatalogError(ValueError):
    """A case or the case catalog is invalid, or the catalog is not available."""


class EvalCaseDraft(BaseModel):
    """A case as Django stores it; unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        pattern=CASE_ID_PATTERN, description="Stable case ID: <suite>.<slug>"
    )
    name: str = Field(min_length=1, description="Short case name")
    description: str = Field(min_length=10, description="What the case checks")
    suite: str = Field(pattern=r"^[a-z0-9_]+$", description="Case group")
    extra_prompts: list[str] = Field(
        default_factory=list,
        description="Prompt templates tested beyond the inferred ones",
    )
    tags: list[str] = Field(default_factory=list, description="Selection tags")
    rules: list[str] = Field(
        default_factory=list,
        description="Prompt spec rules the case protects, as 'spec:rule_id'",
    )
    input: str = Field(min_length=1, description="User message")
    response_format: str = Field(default="plain", description="Response format")
    stage: EvalStage = Field(
        default="full", description="stage1, full, swarm_turn, swarm_watch"
    )
    agent_kind: EvalAgentKind = Field(default="chat", description="chat or swarm")
    swarm: SwarmEvalInput | None = Field(default=None, description="Swarm case context")
    judge: bool = Field(default=False, description="Score with the LLM judge")
    expect: EvalExpect = Field(description="Assertions")


class EvalSchema(BaseModel):
    """Allowed values for case fields, for the case editor."""

    tools: list[str] = Field(description="Tool names for expect.tools_called")
    item_types: list[str] = Field(description="ui_answer item types")
    response_formats: list[str] = Field(description="Response formats")
    prompts: list[str] = Field(description="Prompt template names")
    stages: list[str] = Field(description="Case stages")
    action_types: list[str] = Field(description="Stage 1 action types")
    rules: dict[str, list[str]] = Field(
        description="Response format -> judged rules ('spec:rule_id') a case can protect"
    )
    swarm_rules: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Swarm agent -> judged rules of its turn a swarm case can protect",
    )


def _literal_values(annotation: object) -> list[str]:
    """String values of a Literal, also inside Optional/Union."""
    if get_origin(annotation) is Literal:
        return [value for value in get_args(annotation) if isinstance(value, str)]
    return [value for arg in get_args(annotation) for value in _literal_values(arg)]


def compute_prompt_hashes() -> dict[str, str]:
    """Prompt name -> sha256 of its current version, to tell prompt versions apart.

    Every prompt is a spec: the hash covers the spec and the rules it uses
    (ARCHIE-212; swarm prompts since ARCHIE-285).
    """
    compiler = PromptCompiler()
    hashes = {name: compiler.fingerprint(name) for name in compiler.library.specs}
    return dict(sorted(hashes.items()))


def list_prompt_templates() -> list[str]:
    """Prompt names: the spec names."""
    return list(compute_prompt_hashes())


def get_schema() -> EvalSchema:
    return EvalSchema(
        tools=list(get_args(ToolName)),
        item_types=list(get_args(AdvancedAnswerItem.model_fields["type"].annotation)),
        response_formats=_literal_values(
            ChatRequest.model_fields["response_format"].annotation
        ),
        prompts=list_prompt_templates(),
        stages=list(get_args(EvalStage)),
        action_types=_literal_values(EvalExpect.model_fields["action_type"].annotation),
        rules={
            response_format: [
                rule.key for rule in rules_for_turn(response_format, trace_found=True)
            ]
            for response_format in _literal_values(
                ChatRequest.model_fields["response_format"].annotation
            )
        },
        swarm_rules={
            agent_id: [rule.key for rule in case_rules(agent_id, "swarm_turn", None)]
            for agent_id in SWARM_AGENTS
        },
    )


def infer_prompts(stage: EvalStage, response_format: str) -> list[str]:
    """Prompt templates a case exercises, from its stage and response format."""
    format_prompt = ChatPromptBuilder.format_spec_name(response_format)
    if stage == "stage1":
        return ["cmd_prompt"]
    if resolve_format(response_format).skips_command_loop:
        return [format_prompt]
    return ["cmd_prompt", format_prompt]


def _swarm_errors(draft: EvalCaseDraft) -> list[str]:
    """A swarm case names a known agent, a check for swarm_watch, and protects
    only judged rules of its specs (ARCHIE-297)."""
    swarm = draft.swarm
    if not draft.stage.startswith("swarm_"):
        return [
            f"agent_kind swarm needs stage swarm_turn or swarm_watch, not {draft.stage}"
        ]
    if swarm is None or swarm.agent_id not in SWARM_AGENTS:
        return ["swarm.agent_id must name a swarm agent"]
    errors = []
    if draft.stage == "swarm_watch" and swarm.watch_task is None:
        errors.append("swarm_watch needs swarm.watch_task")
    known = {
        rule.key for rule in case_rules(swarm.agent_id, draft.stage, swarm.watch_task)
    }
    errors += [
        f"rule {key!r} is not a judged rule of this swarm case"
        for key in draft.rules
        if key not in known
    ]
    if draft.expect.rules_followed and not draft.rules:
        errors.append("rules_followed needs at least one rule")
    return errors


def build_case(draft: EvalCaseDraft) -> EvalCaseInfo:
    """Check a draft against the agent's tools, item types and prompts; add inferred prompts."""
    if draft.agent_kind == "swarm":
        errors = _swarm_errors(draft)
        unknown_tools = set(draft.expect.tools_called or []) - set(get_schema().tools)
        if unknown_tools:
            errors.append(f"unknown tools {sorted(unknown_tools)}")
        if errors:
            raise CatalogError(f"{draft.id}: " + "; ".join(errors))
        assert draft.swarm is not None
        inferred = case_specs(draft.swarm.agent_id, draft.stage)
        return EvalCaseInfo(
            prompts=inferred + [p for p in draft.extra_prompts if p not in inferred],
            **draft.model_dump(exclude={"extra_prompts"}),
        )
    if draft.agent_kind != "chat":
        load_agent_kinds()
        try:
            adapter = quality_adapter(draft.agent_kind)
        except UnknownAgentKindError as exc:
            raise CatalogError(f"{draft.id}: unregistered agent kind") from exc
        if (
            draft.stage
            != f"{draft.agent_kind}_{'tick' if draft.agent_kind == 'cognitive' else 'run'}"
        ):
            raise CatalogError(f"{draft.id}: stage does not match agent kind")
        try:
            versions = adapter.spec_versions(draft.response_format)
            known_rules = {
                rule.key for rule in adapter.rules_for(draft.response_format)
            }
        except ValueError as exc:
            raise CatalogError(f"{draft.id}: {exc}") from exc
        unknown = set(draft.rules) - known_rules
        if unknown:
            raise CatalogError(f"{draft.id}: unknown rules {sorted(unknown)}")
        if draft.expect.rules_followed and not draft.rules:
            raise CatalogError(f"{draft.id}: rules_followed needs at least one rule")
        unknown_prompts = set(draft.extra_prompts) - set(list_prompt_templates())
        if unknown_prompts:
            raise CatalogError(f"{draft.id}: unknown prompts {sorted(unknown_prompts)}")
        return EvalCaseInfo(
            prompts=list(versions)
            + [name for name in draft.extra_prompts if name not in versions],
            **draft.model_dump(exclude={"extra_prompts"}),
        )
    if draft.stage.startswith("swarm_"):
        raise CatalogError(f"{draft.id}: stage {draft.stage} needs agent_kind swarm")
    schema = get_schema()
    errors = []
    unknown_tools = set(draft.expect.tools_called or []) - set(schema.tools)
    if unknown_tools:
        errors.append(f"unknown tools {sorted(unknown_tools)}")
    unknown_items = set(draft.expect.expected_item_types or []) - set(schema.item_types)
    if unknown_items:
        errors.append(f"unknown ui item types {sorted(unknown_items)}")
    unknown_prompts = set(draft.extra_prompts) - set(schema.prompts)
    if unknown_prompts:
        errors.append(f"unknown prompts {sorted(unknown_prompts)}")
    if draft.response_format not in schema.response_formats:
        errors.append(f"unknown response_format {draft.response_format!r}")
    else:
        errors += _rule_errors(draft)
    if errors:
        raise CatalogError(f"{draft.id}: " + "; ".join(errors))
    inferred = infer_prompts(draft.stage, draft.response_format)
    return EvalCaseInfo(
        prompts=inferred + [p for p in draft.extra_prompts if p not in inferred],
        **draft.model_dump(exclude={"extra_prompts"}),
    )


def _rule_errors(draft: EvalCaseDraft) -> list[str]:
    """A case may protect only judged rules of the prompts it runs; rules_followed
    can check only rules visible in the response (no tool calls in a full run)."""
    errors = []
    prompts = set(infer_prompts(draft.stage, draft.response_format)) | {ASSISTANT_SPEC}
    known = {rule.key: rule for rule in rules_for_turn(draft.response_format, True)}
    for key in draft.rules:
        rule = known.get(key)
        if rule is None or rule.spec not in prompts:
            errors.append(
                f"rule {key!r} is not a judged rule of {sorted(prompts)} "
                f"for {draft.response_format}"
            )
    if draft.expect.rules_followed:
        if draft.stage != "full" or not draft.rules:
            errors.append("rules_followed needs stage full and at least one rule")
        stage1_rules = [k for k in draft.rules if k.startswith(f"{COMMAND_SPEC}:")]
        if stage1_rules:
            errors.append(
                f"rules_followed cannot judge Stage 1 rules {stage1_rules}; "
                "check them with tools_called / action_type"
            )
    return errors


def parse_catalog(raw_cases: list[dict]) -> list[EvalCaseInfo]:
    """Build cases from Django data; raises CatalogError naming every invalid case."""
    cases, errors = [], []
    for raw in raw_cases:
        try:
            cases.append(build_case(EvalCaseDraft.model_validate(raw)))
        except (ValidationError, CatalogError) as exc:
            errors.append(f"{raw.get('id', '?')}: {exc}")
    if errors:
        raise CatalogError("Invalid cases: " + " | ".join(errors))
    return cases


async def fetch_catalog() -> list[EvalCaseInfo]:
    """Enabled cases from Django."""
    if not (settings.eval_django_url and settings.eval_api_token):
        raise CatalogError("EVAL_DJANGO_URL and EVAL_API_TOKEN are not set")
    url = f"{settings.eval_django_url.rstrip('/')}/cases/"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                url,
                params={"format": "draft"},
                headers={"Authorization": f"Bearer {settings.eval_api_token}"},
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise CatalogError(f"Case catalog is not available: {exc}") from exc
    cases = parse_catalog(response.json()["cases"])
    logger.info(
        f"eval_catalog_001: Loaded \033[33m{len(cases)}\033[0m eval cases from Django"
    )
    return cases


def select_cases(
    catalog: list[EvalCaseInfo], request: EvalRunRequest
) -> list[EvalCaseInfo]:
    """Cases matching any selector of the request; all cases if no selector is set."""
    if not (
        request.case_ids
        or request.prompts
        or request.tags
        or request.suites
        or request.rules
    ):
        return list(catalog)
    unknown_ids = set(request.case_ids) - {case.id for case in catalog}
    if unknown_ids:
        raise CatalogError(f"Unknown or disabled case ids: {sorted(unknown_ids)}")
    return [
        case
        for case in catalog
        if case.id in request.case_ids
        or case.suite in request.suites
        or set(case.prompts) & set(request.prompts)
        or set(case.tags) & set(request.tags)
        or set(case.rules) & set(request.rules)
    ]
