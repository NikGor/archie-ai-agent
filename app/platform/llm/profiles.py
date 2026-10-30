"""Model Registry of AgentSpec runs (ARCHIE-322).

A spec names a profile (work complexity), never a model. The registry resolves
the profile to the first candidate model with a known price and structured
agent support, and translates the logical effort to a level the model accepts.
Escalation moves one profile up, only for a problem the runtime can check.
"""

import logging
from typing import Literal
from archie_shared.chat.models import ReasoningEffort
from pydantic import BaseModel, Field
from app.config import settings
from app.platform.llm.catalog import MODEL_EFFORTS, MODEL_TOKEN_PRICES, supports_agent


logger = logging.getLogger(__name__)

ModelProfile = Literal["FAST", "BALANCED", "DEEP", "MAXIMUM"]
PROFILES: tuple[ModelProfile, ...] = ("FAST", "BALANCED", "DEEP", "MAXIMUM")
LogicalEffort = Literal["low", "medium", "high"]
EscalationReason = Literal["invalid_output", "criteria_failed", "no_progress"]
_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class ModelSelection(BaseModel):
    profile: ModelProfile = Field(description="Requested profile")
    model: str = Field(description="Resolved model")
    effort: ReasoningEffort | None = Field(
        None, description="Provider effort level; None when the model has none"
    )
    reason: str = Field(description="Why this model and effort")


class Escalation(BaseModel):
    from_profile: ModelProfile = Field(description="Profile that failed")
    to_profile: ModelProfile = Field(description="Next profile")
    reason: EscalationReason = Field(description="Checked problem")
    detail: str = Field("", description="What the runtime observed")


class NoModelForProfileError(LookupError):
    """No candidate of the profile is priced and supports agent replies."""


def has_price(model: str) -> bool:
    return any(model.startswith(name) for name in MODEL_TOKEN_PRICES)


def translate_effort(
    model: str, effort: LogicalEffort | None
) -> ReasoningEffort | None:
    """The supported level nearest to the logical effort; ties go up."""
    levels = MODEL_EFFORTS.get(model)
    if effort is None or not levels:
        return None
    wanted = _EFFORT_ORDER.index(effort)
    return min(
        levels,
        key=lambda level: (
            abs(_EFFORT_ORDER.index(level) - wanted),
            -_EFFORT_ORDER.index(level),
        ),
    )


def candidates(profile: ModelProfile) -> list[tuple[str, LogicalEffort | None]]:
    result = []
    for item in settings.agent_profiles.get(profile, []):
        model, _, effort = item.partition(":")
        result.append((model, effort or None))
    return result


def select_model(
    profile: ModelProfile, effort: LogicalEffort | None = None, reason: str = ""
) -> ModelSelection:
    """First usable candidate; the spec's effort overrides the profile default."""
    skipped = []
    for model, default_effort in candidates(profile):
        if not has_price(model) or not supports_agent(model):
            skipped.append(model)
            continue
        level = translate_effort(model, effort or default_effort)
        why = f"profile {profile}" + (f" ({reason})" if reason else "")
        if skipped:
            why += f"; skipped without price or agent support: {', '.join(skipped)}"
        logger.info(
            f"model_profiles_001: \033[35m{profile}\033[0m -> \033[36m{model}\033[0m effort {level}"
        )
        return ModelSelection(profile=profile, model=model, effort=level, reason=why)
    raise NoModelForProfileError(f"No usable model for profile {profile}")


def next_profile(
    current: ModelProfile,
    reason: EscalationReason,
    ceiling: ModelProfile,
    detail: str = "",
) -> Escalation | None:
    """One step up for a checked problem; None at the caller's ceiling."""
    index = PROFILES.index(current)
    if index >= PROFILES.index(ceiling) or index + 1 >= len(PROFILES):
        logger.info(
            f"model_profiles_002: No escalation from {current}: ceiling {ceiling}"
        )
        return None
    step = Escalation(
        from_profile=current,
        to_profile=PROFILES[index + 1],
        reason=reason,
        detail=detail,
    )
    logger.warning(
        f"model_profiles_warning_001: Escalate {current} -> {step.to_profile}: {reason} {detail}"
    )
    return step
