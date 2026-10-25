"""Catalog of swarm expectations (ARCHIE-282): what the swarm must do in a day.

The daily swarm audit checks each rule against the swarm journal, the Langfuse
swarm session and the sources (calendar, cron jobs, check results). A rule
names the prompt spec rules it protects, so a broken expectation points to the
prompt rule to fix, like `rules: ["spec:rule_id"]` of an eval case.
"""

from functools import cache
from pathlib import Path
from typing import Literal
import yaml
from archie_shared.swarm.models import SwarmImportance
from pydantic import BaseModel, Field


CATALOG_PATH = Path(__file__).with_name("expectations.yaml")

ExpectationSource = Literal[
    "calendar",
    "cron",
    "mail",
    "football",
    "home",
    "weather",
    "user",
    "council",
    "ticker",
]
OutputChannel = Literal["display", "voice", "chat"]


class NoticeExpectation(BaseModel):
    """The user must get a notice about each matching event."""

    type: Literal["notice"] = Field("notice", description="Expectation type")
    importance: list[SwarmImportance] = Field(
        description="Allowed importance of the notice"
    )
    channels: list[OutputChannel] = Field(
        default_factory=list,
        description="Outputs that must deliver it; voice only inside the voice hours",
    )
    before_minutes: int | None = Field(
        default=None,
        description="Latest notice time, in minutes before the event starts",
    )
    once: bool = Field(default=True, description="No second notice of the same event")


class DelegationExpectation(BaseModel):
    """A request or a plan step must reach an agent and end with its result."""

    type: Literal["delegation"] = Field("delegation", description="Expectation type")
    within_minutes: int = Field(description="Time from the trigger to the result")


class CoverageExpectation(BaseModel):
    """Checks must run often enough inside the active hours."""

    type: Literal["coverage"] = Field("coverage", description="Expectation type")
    max_gap_minutes: int = Field(description="Longest allowed pause of one check")
    skip_quiet: bool = Field(
        default=True, description="A check may wait while its quiet_until is set"
    )


class DeliveryExpectation(BaseModel):
    """Every output action must succeed: voice has a listener, display answers 2xx."""

    type: Literal["delivery"] = Field("delivery", description="Expectation type")
    channels: list[OutputChannel] = Field(description="Outputs to check")


class NoNoticeExpectation(BaseModel):
    """Something the swarm must not do."""

    type: Literal["no_notice"] = Field("no_notice", description="Expectation type")
    reason: Literal["outside_voice_hours", "duplicate", "quiet_until"] = Field(
        description="What makes a notice wrong"
    )


Expectation = (
    NoticeExpectation
    | DelegationExpectation
    | CoverageExpectation
    | DeliveryExpectation
    | NoNoticeExpectation
)


class ExpectationRule(BaseModel):
    """One expectation of the catalog."""

    id: str = Field(pattern=r"^[a-z0-9_]+$", description="Unique rule ID")
    description: str = Field(description="The expectation in one sentence")
    source: ExpectationSource = Field(description="Where the events come from")
    when: str | None = Field(
        default=None, description="Which source events the rule applies to"
    )
    expect: Expectation = Field(discriminator="type", description="What must happen")
    severity: Literal["red", "orange"] = Field(
        description="red: the user missed something; orange: quality issue"
    )
    judge: bool = Field(
        default=False,
        description="An LLM judge decides which events match (important mail, goals)",
    )
    protects: list[str] = Field(
        default_factory=list,
        description="Prompt spec rules behind the expectation, as 'spec:rule_id'",
    )


class ExpectationCatalog(BaseModel):
    """All expectations of the swarm."""

    rules: list[ExpectationRule] = Field(description="Rules in the order of the audit")


@cache
def load_catalog(path: Path = CATALOG_PATH) -> ExpectationCatalog:
    catalog = ExpectationCatalog.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )
    ids = [rule.id for rule in catalog.rules]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate expectation ids: {sorted(duplicates)}")
    return catalog
