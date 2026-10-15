"""Eval assertions: compare a case's expectations with the agent output."""

import json
from typing import Any
from archie_shared.evals.models import EvalAssertionFailure, EvalCaseInfo
from archie_shared.ui.models import UIAnswer
from app.models.orchestration_sgr import DecisionResponse
from app.models.output_models import AgentResponse


_FORM_TYPES = {"event_form", "note_form", "email_form"}


def _failure(key: str, expected: Any, actual: Any) -> EvalAssertionFailure:
    return EvalAssertionFailure(
        key=key,
        expected=json.dumps(expected, ensure_ascii=False, default=str),
        actual=json.dumps(actual, ensure_ascii=False, default=str),
    )


def check_decision(
    case: EvalCaseInfo, decision: DecisionResponse
) -> list[EvalAssertionFailure]:
    """Stage 1 assertions."""
    failures = []
    expect = case.expect
    actual_type = decision.sgr.action.type
    if expect.action_type is not None and actual_type != expect.action_type:
        failures.append(_failure("action_type", expect.action_type, actual_type))
    if expect.tools_called is not None:
        actual_tools = [call.tool_name for call in decision.sgr.tool_calls]
        if not set(expect.tools_called) <= set(actual_tools):
            failures.append(_failure("tools_called", expect.tools_called, actual_tools))
    return failures


def response_text(response: AgentResponse) -> str:
    content = response.content
    return (content.text or content.model_dump_json()) if content else ""


def find_ui_answer(response: AgentResponse) -> UIAnswer | None:
    return response.content.ui_answer if response.content else None


def check_response(
    case: EvalCaseInfo,
    response: AgentResponse,
    total_tokens: int | None,
    total_ms: int | None,
) -> list[EvalAssertionFailure]:
    """Full pipeline assertions."""
    failures = []
    expect = case.expect
    text = response_text(response)
    if expect.content_not_empty and not text.strip():
        failures.append(_failure("content_not_empty", True, text))
    if expect.text_contains is not None and expect.text_contains not in text:
        failures.append(_failure("text_contains", expect.text_contains, text[:200]))
    if (
        expect.max_total_tokens is not None
        and total_tokens is not None
        and total_tokens > expect.max_total_tokens
    ):
        failures.append(
            _failure("max_total_tokens", expect.max_total_tokens, total_tokens)
        )
    if (
        expect.max_latency_ms is not None
        and total_ms is not None
        and total_ms > expect.max_latency_ms
    ):
        failures.append(_failure("max_latency_ms", expect.max_latency_ms, total_ms))

    ui_answer = find_ui_answer(response)
    if ui_answer is None:
        return failures
    item_types = sorted({item.type for item in ui_answer.items})
    if expect.min_ui_items is not None and len(ui_answer.items) < expect.min_ui_items:
        failures.append(
            _failure("min_ui_items", f">={expect.min_ui_items}", len(ui_answer.items))
        )
    if expect.has_quick_actions and not ui_answer.quick_action_buttons:
        failures.append(_failure("has_quick_actions", True, None))
    if expect.has_form and not _FORM_TYPES & set(item_types):
        failures.append(_failure("has_form", True, item_types))
    for item_type in expect.expected_item_types or []:
        if item_type not in item_types:
            failures.append(_failure("expected_item_types", item_type, item_types))
    return failures
