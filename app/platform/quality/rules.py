"""Spec rules a judge checks (ARCHIE-213, ARCHIE-296).

An agent kind gives a RulePolicy (rules it never judges, style advice, rules
with a condition); the selection below is the same for every kind.
"""

import functools
from dataclasses import dataclass
from pydantic import BaseModel, Field
from ..prompts.compiler import PromptCompiler, PromptLibrary, load_library


@dataclass(frozen=True)
class SpecRule:
    """One rule the judge checks."""

    spec: str
    rule_id: str
    text: str
    guidance: bool = False
    # The rule concerns only some requests: when false, the judge must answer
    # "not applicable" instead of "violated"
    condition: str | None = None

    @property
    def key(self) -> str:
        return f"{self.spec}:{self.rule_id}"


class RulePolicy(BaseModel):
    """How an agent kind judges its spec rules."""

    excluded: frozenset[str] = Field(
        default_factory=frozenset,
        description="Rule ids never judged: internals, schema instructions, examples",
    )
    guidance: frozenset[str] = Field(
        default_factory=frozenset,
        description="Style advice: breaking one gives orange, never red",
    )
    conditions: dict[str, str] = Field(
        default_factory=dict,
        description="Rule id -> when the rule applies",
    )


@functools.cache
def compiler() -> PromptCompiler:
    return PromptCompiler(load_library())


def library() -> PromptLibrary:
    return compiler().library


def is_templated(rule_id: str) -> bool:
    """State facts like "Timezone: {{ state.user_timezone }}" are not judged."""
    rule = library().rules[rule_id]
    return any("{{" in part or "{%" in part for part in [rule.text, *rule.items])


def applies(expression: str | None, context: dict) -> bool:
    """A `when` condition for this context; conditions on the user state are
    unknown for stored records, so such rules are not judged."""
    if expression is None:
        return True
    if "state" in expression:
        return False
    return compiler().check(expression, **context)


def select_spec_rules(
    spec: str, context: dict, policy: RulePolicy
) -> tuple[SpecRule, ...]:
    """Judged rules of one spec for a context, in spec order, each rule once."""
    rules: list[SpecRule] = []
    seen: set[str] = set()
    for section in library().specs[spec].sections:
        if not applies(section.when, context):
            continue
        for ref in section.rules + section.fallback:
            rule_id = ref.rule
            if (
                rule_id in seen
                or rule_id in policy.excluded
                or is_templated(rule_id)
                or not applies(ref.when, context)
            ):
                continue
            seen.add(rule_id)
            rules.append(
                SpecRule(
                    spec=spec,
                    rule_id=rule_id,
                    text=compiler().render_rule(rule_id),
                    guidance=rule_id in policy.guidance,
                    condition=policy.conditions.get(rule_id),
                )
            )
    return tuple(rules)


def spec_fingerprints(specs: list[str]) -> dict[str, str]:
    """Spec name -> fingerprint of its current version."""
    return {spec: compiler().fingerprint(spec) for spec in specs}
