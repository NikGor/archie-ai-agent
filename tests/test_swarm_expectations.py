"""The swarm expectation catalog loads and points to real prompt rules (ARCHIE-282)."""

import pytest
from app.agent.prompt_compiler import load_library
from app.config import settings
from app.swarm.audit.models import load_catalog


def test_catalog_loads_with_unique_ids():
    catalog = load_catalog()
    assert catalog.rules
    assert len({rule.id for rule in catalog.rules}) == len(catalog.rules)


@pytest.mark.parametrize(
    "ref", sorted({ref for rule in load_catalog().rules for ref in rule.protects})
)
def test_protected_rule_is_used_by_its_spec(ref):
    spec_name, rule_id = ref.split(":")
    spec = load_library().specs[spec_name]
    used = {r.rule for section in spec.sections for r in section.rules}
    assert rule_id in used, f"{rule_id} is not in spec {spec_name}"


def test_plan_step_limit_matches_the_step_timeout():
    rule = next(r for r in load_catalog().rules if r.id == "council_steps_done")
    assert rule.expect.within_minutes * 60 == settings.swarm_step_timeout_seconds
