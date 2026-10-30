"""Model Registry: profiles, effort translation and escalation (ARCHIE-322)."""

import logging
import pytest
from pydantic import ValidationError
from app.config import settings
from app.platform.llm.catalog import MODEL_TOKEN_PRICES, supports_agent
from app.platform.llm.profiles import (
    PROFILES,
    NoModelForProfileError,
    next_profile,
    select_model,
    translate_effort,
)
from app.utils.openai_utils import build_openai_args


@pytest.mark.parametrize("profile", PROFILES)
def test_every_profile_resolves_to_a_priced_agent_model(profile):
    selection = select_model(profile)
    assert supports_agent(selection.model)
    assert any(selection.model.startswith(name) for name in MODEL_TOKEN_PRICES)


def test_unpriced_or_unsupported_candidates_are_skipped(monkeypatch):
    monkeypatch.setitem(
        settings.agent_profiles,
        "FAST",
        ["no-such-model:low", "anthropic/claude-fable-5.1", "gpt-4.1"],
    )
    selection = select_model("FAST")
    assert selection.model == "gpt-4.1"
    assert "skipped" in selection.reason
    monkeypatch.setitem(settings.agent_profiles, "FAST", ["no-such-model"])
    with pytest.raises(NoModelForProfileError):
        select_model("FAST")


def test_logical_effort_maps_to_a_supported_level_and_reaches_the_client():
    assert translate_effort("gpt-5.4-pro", "low") == "medium"
    assert translate_effort("mistralai/mistral-large-4-0", "medium") == "high"
    assert translate_effort("z-ai/glm-5.3", "medium") == "high"
    assert translate_effort("gpt-4.1", "high") is None
    selection = select_model("MAXIMUM")
    args = build_openai_args(selection.model, [], reasoning_effort=selection.effort)
    assert args["reasoning"]["effort"] == "high"
    plain = build_openai_args(
        "gpt-4.1", [], reasoning_effort=translate_effort("gpt-4.1", "high")
    )
    assert "reasoning" not in plain


def test_spec_effort_overrides_profile_default():
    assert select_model("BALANCED", "high").effort == "high"
    assert select_model("BALANCED").effort == "medium"


def test_escalation_is_one_step_and_stops_at_the_ceiling(caplog):
    with caplog.at_level(logging.WARNING):
        step = next_profile("FAST", "invalid_output", "DEEP", "schema error x2")
    assert step.to_profile == "BALANCED" and step.reason == "invalid_output"
    assert "Escalate FAST -> BALANCED" in caplog.text
    assert next_profile("BALANCED", "no_progress", "BALANCED") is None
    assert next_profile("MAXIMUM", "criteria_failed", "MAXIMUM") is None
    with pytest.raises(ValidationError):
        next_profile("FAST", "agent_asked", "DEEP")  # not a checked problem
