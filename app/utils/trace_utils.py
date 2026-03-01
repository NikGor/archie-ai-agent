"""Pipeline step timing utilities for agent tracing."""

import time
from archie_shared.chat.models import LllmTrace, StepTrace


class StepTimer:
    """Context manager for measuring elapsed wall-clock time of a pipeline step."""

    def __init__(self):
        self._start: float = 0.0
        self.duration_ms: int = 0

    def __enter__(self) -> "StepTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *args) -> None:
        self.duration_ms = int((time.monotonic() - self._start) * 1000)


def accumulate_llm_traces(traces: list[LllmTrace]) -> LllmTrace | None:
    """Combine multiple LllmTrace records into one by summing tokens and costs.

    Uses the last trace's model name and token detail objects.
    Returns None if the list is empty.
    """
    if not traces:
        return None
    return LllmTrace(
        model=traces[-1].model,
        input_tokens=sum(t.input_tokens for t in traces),
        input_tokens_details=traces[-1].input_tokens_details,
        output_tokens=sum(t.output_tokens for t in traces),
        output_tokens_details=traces[-1].output_tokens_details,
        total_tokens=sum(t.total_tokens for t in traces),
        total_cost=sum(t.total_cost for t in traces),
    )


def make_step_trace(duration_ms: int, llm_trace: LllmTrace | None = None) -> StepTrace:
    """Build a StepTrace from a duration measurement and optional LLM trace."""
    return StepTrace(duration_ms=duration_ms, llm_trace=llm_trace)
