"""Pipeline step timing utilities for agent tracing."""

import time

from archie_shared.chat.models import LllmTrace, PipelineTrace, StepTrace


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


def make_step_trace(
    duration_ms: int,
    llm_trace: LllmTrace | None = None,
    ttft_ms: int | None = None,
) -> StepTrace:
    """Build a StepTrace from a duration measurement and optional LLM trace."""
    return StepTrace(duration_ms=duration_ms, ttft_ms=ttft_ms, llm_trace=llm_trace)


def build_pipeline_trace(
    *,
    total_ms: int,
    stage3_duration_ms: int,
    stage3_llm_trace: LllmTrace | None = None,
    stage3_ttft_ms: int | None = None,
    stage1_duration_ms: int = 0,
    stage1_llm_traces: list[LllmTrace] | None = None,
    stage2_duration_ms: int = 0,
) -> PipelineTrace:
    """Build a complete PipelineTrace for both direct and 3-stage flows."""
    create_output_trace = make_step_trace(
        stage3_duration_ms, stage3_llm_trace, ttft_ms=stage3_ttft_ms
    )

    # Direct flow (dashboard/widget): no Stage 1 or 2
    if not stage1_duration_ms and not stage2_duration_ms:
        return PipelineTrace(
            create_output=create_output_trace,
            ttft_ms=stage3_ttft_ms,
            total_ms=total_ms,
        )

    # Full 3-stage flow
    command_call_trace = make_step_trace(
        stage1_duration_ms, accumulate_llm_traces(stage1_llm_traces or [])
    )
    stage2_trace = make_step_trace(stage2_duration_ms) if stage2_duration_ms else None
    pipeline_ttft_ms = (
        (stage1_duration_ms + stage2_duration_ms + stage3_ttft_ms)
        if stage3_ttft_ms is not None
        else None
    )
    return PipelineTrace(
        command_call=command_call_trace,
        tool_execution=stage2_trace,
        create_output=create_output_trace,
        ttft_ms=pipeline_ttft_ms,
        total_ms=total_ms,
    )
