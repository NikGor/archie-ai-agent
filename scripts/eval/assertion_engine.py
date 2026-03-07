"""Assertion engine — checks eval expectations against agent responses."""

from archie_shared.chat.models import PipelineTrace

from app.models.orchestration_sgr import DecisionResponse
from app.models.output_models import AgentResponse

from .eval_types import AssertionFailure, EvalCase, EvalResult


def _total_tokens_from_trace(pipeline_trace: PipelineTrace | None) -> int | None:
    if pipeline_trace is None:
        return None
    total = 0
    for stage in (pipeline_trace.stage1, pipeline_trace.stage2, pipeline_trace.stage3):
        if stage and stage.llm_trace:
            total += stage.llm_trace.total_tokens
    return total


def check_stage1(case: EvalCase, decision: DecisionResponse) -> EvalResult:
    failures: list[AssertionFailure] = []
    expect = case.expect
    if expect.action_type is not None:
        actual = decision.sgr.action.type
        if actual != expect.action_type:
            failures.append(AssertionFailure("action_type", expect.action_type, actual))
    if expect.tools_called is not None:
        actual_tools = [tc.tool_name for tc in decision.sgr.tool_calls]
        for expected_tool in expect.tools_called:
            if expected_tool not in actual_tools:
                failures.append(
                    AssertionFailure("tools_called", expect.tools_called, actual_tools)
                )
                break
    return EvalResult(
        name=case.name,
        passed=len(failures) == 0,
        failures=failures,
    )


def check_full(case: EvalCase, response: AgentResponse) -> EvalResult:
    failures: list[AssertionFailure] = []
    expect = case.expect
    if expect.content_not_empty is True:
        if response.content is None:
            failures.append(AssertionFailure("content_not_empty", True, None))
        else:
            text = getattr(response.content, "text", None) or str(response.content)
            if not text or not text.strip():
                failures.append(AssertionFailure("content_not_empty", True, repr(text)))
    if expect.text_contains is not None:
        text = ""
        if response.content:
            text = getattr(response.content, "text", None) or str(response.content)
        if expect.text_contains not in text:
            failures.append(
                AssertionFailure("text_contains", expect.text_contains, text[:200])
            )
    pt = response.pipeline_trace
    stage1_ms = pt.stage1.duration_ms if pt and pt.stage1 else None
    stage2_ms = pt.stage2.duration_ms if pt and pt.stage2 else None
    stage3_ms = pt.stage3.duration_ms if pt and pt.stage3 else None
    total_ms = pt.total_ms if pt else None
    total_tokens = _total_tokens_from_trace(pt)
    if expect.max_total_tokens is not None and total_tokens is not None:
        if total_tokens > expect.max_total_tokens:
            failures.append(
                AssertionFailure(
                    "max_total_tokens", expect.max_total_tokens, total_tokens
                )
            )
    if expect.max_latency_ms is not None and total_ms is not None:
        if total_ms > expect.max_latency_ms:
            failures.append(
                AssertionFailure("max_latency_ms", expect.max_latency_ms, total_ms)
            )
    if expect.action_type is not None and pt and pt.stage1:
        pass
    return EvalResult(
        name=case.name,
        passed=len(failures) == 0,
        failures=failures,
        stage1_ms=stage1_ms,
        stage2_ms=stage2_ms,
        stage3_ms=stage3_ms,
        total_ms=total_ms,
        total_tokens=total_tokens,
    )
