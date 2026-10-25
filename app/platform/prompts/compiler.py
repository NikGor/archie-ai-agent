"""Prompt compiler: turns declarative prompt specs into prompt text (ARCHIE-212).

A spec (`prompts/specs/**/<name>.yaml`) says WHAT a prompt contains: ordered
sections that list rule ids. A rule catalog (`prompts/rules/*.json`) says HOW
each rule reads: its text and optional sub-items. The compiler resolves
conditions and includes, drops duplicate and empty rules, renders the
variables and lays out each section in its style.
"""

import functools
import hashlib
import json
import logging
from pathlib import Path
from typing import Literal
import yaml
from jinja2 import Environment, TemplateSyntaxError
from pydantic import BaseModel, Field, ValidationError, field_validator


logger = logging.getLogger(__name__)

# Prompt data stays in app/agent/prompts until ARCHIE-294 moves it to app/prompts
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "agent" / "prompts"

SectionStyle = Literal["bullets", "numbered", "lines", "paragraphs"]
ItemStyle = Literal["bullets", "numbered"]


class PromptSpecError(ValueError):
    """A spec or rule catalog is invalid, or a spec is not found."""


class PromptRule(BaseModel):
    """One rule of a catalog: the text the model reads."""

    text: str = Field(description="Rule text; a Jinja2 template over the context")
    items: list[str] = Field(
        default_factory=list,
        description="Sub-items shown under the text (examples, word lists)",
    )
    item_style: ItemStyle = Field(default="bullets", description="Sub-item layout")


class RuleRef(BaseModel):
    """A rule a spec section uses, with an optional condition."""

    rule: str = Field(description="Rule id from a rule catalog")
    when: str | None = Field(
        default=None, description="Jinja2 expression; the rule is kept only if true"
    )


class SpecSection(BaseModel):
    """An ordered group of rules, optionally under a heading."""

    title: str | None = Field(default=None, description="Section heading")
    level: int = Field(default=1, ge=1, le=4, description="Heading level")
    style: SectionStyle = Field(default="bullets", description="Rule layout")
    when: str | None = Field(
        default=None, description="Jinja2 expression; the section is kept only if true"
    )
    rules: list[RuleRef] = Field(default_factory=list, description="Rules in order")
    include: str | None = Field(
        default=None,
        description="Name of another spec to inline here; a Jinja2 template",
    )
    fallback: list[RuleRef] = Field(
        default_factory=list,
        description="Rules used when the included spec does not exist",
    )

    @field_validator("rules", "fallback", mode="before")
    @classmethod
    def coerce_rule_refs(cls, value: list) -> list:
        return [{"rule": ref} if isinstance(ref, str) else ref for ref in value]


class PromptSpec(BaseModel):
    """A prompt as an ordered list of sections."""

    name: str = Field(description="Spec name (file stem), also the eval prompt name")
    sections: list[SpecSection] = Field(description="Sections in order")


class PromptLibrary(BaseModel):
    """All specs and rules of a prompts directory, checked for consistency."""

    specs: dict[str, PromptSpec] = Field(description="Spec name -> spec")
    rules: dict[str, PromptRule] = Field(description="Rule id -> rule")
    sources: dict[str, str] = Field(description="Spec name -> spec file text")

    def used_rule_ids(self) -> set[str]:
        return {
            ref.rule
            for spec in self.specs.values()
            for section in spec.sections
            for ref in section.rules + section.fallback
        }

    def unused_rule_ids(self) -> set[str]:
        return set(self.rules) - self.used_rule_ids()


def _read_rules(rules_dir: Path) -> dict[str, PromptRule]:
    rules: dict[str, PromptRule] = {}
    for path in sorted(rules_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            catalog = {rule_id: PromptRule(**rule) for rule_id, rule in data.items()}
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise PromptSpecError(f"{path.name}: {exc}") from exc
        duplicates = set(catalog) & set(rules)
        if duplicates:
            raise PromptSpecError(
                f"{path.name}: duplicate rule ids {sorted(duplicates)}"
            )
        rules.update(catalog)
    return rules


def _read_specs(specs_dir: Path) -> tuple[dict[str, PromptSpec], dict[str, str]]:
    specs: dict[str, PromptSpec] = {}
    sources: dict[str, str] = {}
    for path in sorted(specs_dir.rglob("*.yaml")):
        if path.stem in specs:
            raise PromptSpecError(f"{path}: duplicate spec name {path.stem!r}")
        source = path.read_text(encoding="utf-8")
        try:
            specs[path.stem] = PromptSpec(name=path.stem, **yaml.safe_load(source))
        except (yaml.YAMLError, ValidationError, TypeError) as exc:
            raise PromptSpecError(f"{path}: {exc}") from exc
        sources[path.stem] = source
    return specs, sources


def _check_library(library: PromptLibrary, env: Environment) -> None:
    """Fail at load time on unknown rule ids and broken templates."""
    errors = []
    for spec in library.specs.values():
        for section in spec.sections:
            missing = {ref.rule for ref in section.rules + section.fallback} - set(
                library.rules
            )
            if missing:
                errors.append(f"{spec.name}: unknown rules {sorted(missing)}")
            expressions = [section.when] + [
                ref.when for ref in section.rules + section.fallback
            ]
            for expression in filter(None, expressions):
                try:
                    env.compile_expression(expression)
                except TemplateSyntaxError as exc:
                    errors.append(f"{spec.name}: bad condition {expression!r}: {exc}")
    for rule_id, rule in library.rules.items():
        for template in [rule.text, *rule.items]:
            try:
                env.parse(template)
            except TemplateSyntaxError as exc:
                errors.append(f"rule {rule_id}: {exc}")
    if errors:
        raise PromptSpecError("; ".join(errors))


def _create_env() -> Environment:
    # Prompts are plain text: no autoescape, keep newlines as written.
    return Environment(autoescape=False, keep_trailing_newline=False)


@functools.cache
def load_library(prompts_dir: Path = PROMPTS_DIR) -> PromptLibrary:
    """Load and check all specs and rules once per process."""
    specs, sources = _read_specs(prompts_dir / "specs")
    library = PromptLibrary(
        specs=specs, rules=_read_rules(prompts_dir / "rules"), sources=sources
    )
    _check_library(library, _create_env())
    logger.info(
        f"prompt_compiler_001: Loaded \033[33m{len(library.specs)}\033[0m specs, "
        f"\033[33m{len(library.rules)}\033[0m rules"
    )
    return library


def _indent_tail(text: str, prefix: str) -> str:
    """Indent every line but the first, so a multi-line rule stays in its bullet."""
    head, *tail = text.split("\n")
    return "\n".join([head, *(f"{prefix}{line}" if line else line for line in tail)])


class PromptCompiler:
    """Compiles a spec and a context into the prompt text for the model."""

    def __init__(self, library: PromptLibrary | None = None) -> None:
        self.library = library or load_library()
        self.env = _create_env()

    def has_spec(self, name: str) -> bool:
        return name in self.library.specs

    def compile(self, name: str, **context: object) -> str:
        """Render spec `name` with the context (e.g. state=..., response_format=...)."""
        if name not in self.library.specs:
            raise PromptSpecError(f"unknown prompt spec {name!r}")
        blocks = self._compile_spec(name, context, seen_rules=set(), stack=())
        return "\n\n".join(blocks)

    def check(self, expression: str | None, **context: object) -> bool:
        """Value of a spec `when` expression in the context (no expression is true)."""
        return self._check(expression, context)

    def render_rule(self, rule_id: str, **context: object) -> str:
        """Text of one rule with its items, as the model reads it in a prompt."""
        return self._render_rule(self.library.rules[rule_id], context)

    def fingerprint(self, name: str) -> str:
        """sha256 of the spec and every rule it uses, to tell prompt versions apart.

        Includes with a fixed name are followed; templated ones (e.g. the persona
        chosen from the state) are versioned by their own spec.
        """
        digest = hashlib.sha256()
        for spec_name in self._static_closure(name):
            digest.update(self.library.sources[spec_name].encode())
            spec = self.library.specs[spec_name]
            rule_ids = sorted(
                {ref.rule for s in spec.sections for ref in s.rules + s.fallback}
            )
            rules = {
                rule_id: self.library.rules[rule_id].model_dump()
                for rule_id in rule_ids
            }
            digest.update(
                json.dumps(rules, ensure_ascii=False, sort_keys=True).encode()
            )
        return digest.hexdigest()

    def _static_closure(self, name: str) -> list[str]:
        names = [name]
        for section in self.library.specs[name].sections:
            include = section.include
            if include and "{" not in include and include in self.library.specs:
                names += [n for n in self._static_closure(include) if n not in names]
        return names

    def _check(self, expression: str | None, context: dict) -> bool:
        return expression is None or bool(
            self.env.compile_expression(expression)(**context)
        )

    def _compile_spec(
        self, name: str, context: dict, seen_rules: set[str], stack: tuple[str, ...]
    ) -> list[str]:
        if name in stack:
            raise PromptSpecError(f"include cycle: {' -> '.join((*stack, name))}")
        blocks = []
        for section in self.library.specs[name].sections:
            if not self._check(section.when, context):
                continue
            body: list[str] = []
            if section.include:
                included = self.env.from_string(section.include).render(**context)
                if included in self.library.specs:
                    body += self._compile_spec(
                        included, context, seen_rules, (*stack, name)
                    )
                else:
                    body += self._render_rules(
                        section, section.fallback, context, seen_rules
                    )
            body += self._render_rules(section, section.rules, context, seen_rules)
            # A section without rules is a pure heading for the sections below it
            if not body and (section.rules or section.include):
                continue
            heading = (
                f"{'#' * section.level} {section.title}\n" if section.title else ""
            )
            separator = "\n\n" if section.style == "paragraphs" else "\n"
            blocks.append(heading + separator.join(body))
        return blocks

    def _render_rules(
        self,
        section: SpecSection,
        refs: list[RuleRef],
        context: dict,
        seen_rules: set[str],
    ) -> list[str]:
        lines = []
        for ref in refs:
            if ref.rule in seen_rules or not self._check(ref.when, context):
                continue
            seen_rules.add(ref.rule)
            text = self._render_rule(self.library.rules[ref.rule], context)
            if not text:
                continue
            if section.style == "bullets":
                text = "- " + _indent_tail(text, "  ")
            elif section.style == "numbered":
                text = f"{len(lines) + 1}. " + _indent_tail(text, "   ")
            lines.append(text)
        return lines

    def _render_rule(self, rule: PromptRule, context: dict) -> str:
        text = self.env.from_string(rule.text).render(**context).strip()
        items = [
            self.env.from_string(item).render(**context).strip() for item in rule.items
        ]
        # dict.fromkeys drops repeated items and keeps their order
        items = [item for item in dict.fromkeys(items) if item]
        if not items:
            return text
        # Bullet and numbered styles indent the item lines via _indent_tail
        markers = (
            [f"{n}." for n in range(1, len(items) + 1)]
            if rule.item_style == "numbered"
            else ["-"] * len(items)
        )
        return "\n".join(
            [
                text,
                *(
                    f"{marker} {item}"
                    for marker, item in zip(markers, items, strict=True)
                ),
            ]
        )
