"""Budgets of agent runs and the daily spend of background kinds (ARCHIE-298)."""

import logging
import time
from datetime import date
from typing import Literal
from archie_shared.chat.models import LllmTrace
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

TriggerKind = Literal["request", "schedule", "event", "spawn"]


class Trigger(BaseModel):
    """What started an agent run."""

    kind: TriggerKind = Field(description="request | schedule | event | spawn")
    source: str = Field(description="Who started it: user, cron, ticker, an agent id")


class Budget(BaseModel):
    """Limits of one agent run; None = no limit."""

    max_tokens: int | None = Field(default=None, description="Input + output tokens")
    max_cost_usd: float | None = Field(default=None, description="LLM cost in USD")
    max_seconds: float | None = Field(default=None, description="Wall time")
    max_tool_rounds: int | None = Field(default=None, description="Tool rounds")


class BudgetExceededError(RuntimeError):
    """A run used up one of its limits."""

    def __init__(self, limit: str, used: float, allowed: float) -> None:
        super().__init__(f"budget {limit} exceeded: {used:g} > {allowed:g}")
        self.limit = limit
        self.used = used
        self.allowed = allowed


class BudgetMeter:
    """Counts what a run used and stops it at the first exceeded limit."""

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.started = time.monotonic()
        self.tokens = 0
        self.cost = 0.0
        self.rounds = 0

    def add(self, trace: LllmTrace | None) -> None:
        if trace is None:
            return
        self.tokens += trace.input_tokens + trace.output_tokens
        self.cost += trace.total_cost or 0.0
        self.check()

    def add_round(self) -> None:
        self.rounds += 1

    def last_round(self) -> bool:
        """True when the next tool round would pass max_tool_rounds."""
        limit = self.budget.max_tool_rounds
        return limit is not None and self.rounds >= limit

    @property
    def seconds(self) -> float:
        return time.monotonic() - self.started

    def check(self) -> None:
        for limit, used, allowed in (
            ("max_tokens", self.tokens, self.budget.max_tokens),
            ("max_cost_usd", self.cost, self.budget.max_cost_usd),
            ("max_seconds", self.seconds, self.budget.max_seconds),
        ):
            if allowed is not None and used > allowed:
                logger.warning(
                    f"runtime_budget_warning_001: \033[33m{limit}\033[0m {used:g} > {allowed:g}"
                )
                raise BudgetExceededError(limit, used, allowed)


class DailySpend:
    """LLM spend of background kinds per local day, in memory (resets on restart)."""

    def __init__(self) -> None:
        self.day = date.today()
        self.by_kind: dict[str, float] = {}

    def _roll(self) -> None:
        if date.today() != self.day:
            self.day = date.today()
            self.by_kind = {}

    def add(self, kind: str, cost: float) -> None:
        self._roll()
        self.by_kind[kind] = self.by_kind.get(kind, 0.0) + cost

    def spent(self, kind: str) -> float:
        self._roll()
        return self.by_kind.get(kind, 0.0)

    def exceeded(self, kind: str, limit: float | None) -> bool:
        return limit is not None and self.spent(kind) >= limit


daily_spend = DailySpend()
