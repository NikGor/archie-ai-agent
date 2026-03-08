"""Data types for the eval runner."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class JudgeResult:
    ui_optimality: int
    content_informativeness: int
    prompt_following: int
    reasoning: str

    @property
    def mean_score(self) -> float:
        return (
            self.ui_optimality + self.content_informativeness + self.prompt_following
        ) / 3


@dataclass
class Expect:
    action_type: str | None = None
    tools_called: list[str] | None = None
    content_not_empty: bool | None = None
    max_total_tokens: int | None = None
    max_latency_ms: int | None = None
    text_contains: str | None = None
    # UI quality metrics (ui_answer format)
    min_ui_items: int | None = None
    has_quick_actions: bool | None = None
    has_form: bool | None = None
    expected_item_types: list[str] | None = None


@dataclass
class EvalCase:
    name: str
    input: str
    response_format: str = "plain"
    stage: str = "full"
    judge: bool = False
    expect: Expect = field(default_factory=Expect)


@dataclass
class AssertionFailure:
    key: str
    expected: Any
    actual: Any

    def __str__(self) -> str:
        return f"{self.key}: expected={self.expected!r}, actual={self.actual!r}"


@dataclass
class EvalResult:
    name: str
    passed: bool
    failures: list[AssertionFailure] = field(default_factory=list)
    stage1_ms: int | None = None
    stage2_ms: int | None = None
    stage3_ms: int | None = None
    total_ms: int | None = None
    total_tokens: int | None = None
    ui_items_count: int | None = None
    judge: JudgeResult | None = None
    error: str | None = None


@dataclass
class RunSummary:
    timestamp: str
    model: str
    suite: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    median_total_ms: float | None
    mean_total_tokens: float | None
    stage1_mean_ms: float | None
    stage3_mean_ms: float | None
    mean_ui_items: float | None = None
    mean_judge_ui_optimality: float | None = None
    mean_judge_content_informativeness: float | None = None
    mean_judge_prompt_following: float | None = None
    cases: list[EvalResult] = field(default_factory=list)
