"""Data types for the eval runner."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Expect:
    action_type: str | None = None
    tools_called: list[str] | None = None
    content_not_empty: bool | None = None
    max_total_tokens: int | None = None
    max_latency_ms: int | None = None
    text_contains: str | None = None


@dataclass
class EvalCase:
    name: str
    input: str
    response_format: str = "plain"
    stage: str = "full"
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
    cases: list[EvalResult] = field(default_factory=list)
