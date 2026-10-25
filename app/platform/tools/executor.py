import asyncio
import logging
from collections.abc import Collection
from typing import Any
from langfuse import get_client
from ...models.ws_models import StatusCallback, StatusNotifier
from ...utils.status_messages import get_tool_detail
from .models import Parameter, ToolCall, ToolContext, ToolResult
from .registry import ToolRegistry


logger = logging.getLogger(__name__)


def _result_error(result: Any) -> str | None:
    """Return an error message for the error dictionaries used by Archie tools."""
    if not isinstance(result, dict):
        return None
    if result.get("error"):
        return str(result["error"])
    if result.get("status") == "error":
        return str(result.get("message") or "Tool returned an error status")
    return None


def convert_parameters_to_dict(parameters: list[Parameter]) -> dict[str, Any]:
    return {param.name: param.value for param in parameters}


class RepeatedArgumentsError(ValueError):
    """Repeated parameters of one tool call have different counts (ARCHIE-255)."""


def expand_repeated_arguments(tool_call: ToolCall) -> list[ToolCall]:
    """Split a call that repeats a parameter into one call per value (ARCHIE-255).

    Stage 1 sometimes puts several values of one parameter into one call, e.g.
    seven `query` arguments for google_search_tool, and a dict keeps only the
    last one. Repeated parameters are paired by position; single ones go to
    every call.
    """
    values: dict[str, list[str]] = {}
    for param in tool_call.arguments:
        values.setdefault(param.name, []).append(param.value)
    repeated = {name: found for name, found in values.items() if len(found) > 1}
    if not repeated:
        return [tool_call]
    counts = {len(found) for found in repeated.values()}
    if len(counts) > 1:
        listed = ", ".join(f"{name} x{len(found)}" for name, found in repeated.items())
        raise RepeatedArgumentsError(
            f"{tool_call.tool_name}: repeated parameters have different counts "
            f"({listed}). Make one call per set of values."
        )
    return [
        tool_call.model_copy(
            update={
                "arguments": [
                    Parameter(name=name, value=found[index % len(found)])
                    for name, found in values.items()
                ]
            }
        )
        for index in range(counts.pop())
    ]


async def execute_tool_call(
    tool_call: ToolCall,
    registry: ToolRegistry,
    on_status: StatusCallback = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Run one tool call inside a Langfuse `tool` observation (ARCHIE-211)."""
    with get_client().start_as_current_observation(
        as_type="tool",
        name=tool_call.tool_name,
        input=convert_parameters_to_dict(tool_call.arguments),
        metadata={"reason": tool_call.reason},
    ) as observation:
        result = await _run_tool_call(tool_call, registry, on_status, context)
        observation.update(
            output=result.output,
            level="ERROR" if result.error else "DEFAULT",
            status_message=result.error,
        )
        return result


async def _run_tool_call(
    tool_call: ToolCall,
    registry: ToolRegistry,
    on_status: StatusCallback,
    context: ToolContext | None,
) -> ToolResult:
    logger.info(
        f"tool_executor_001: Executing tool: \033[36m{tool_call.tool_name}\033[0m"
    )
    notifier = StatusNotifier(on_status)
    arguments_dict = convert_parameters_to_dict(tool_call.arguments)
    detail = get_tool_detail(tool_call.tool_name, arguments_dict)
    await notifier.emit(
        "tools",
        "started",
        detail or f"Calling {tool_call.tool_name}",
        detail=detail,
    )
    logger.info(f"tool_executor_002: Arguments: \033[33m{arguments_dict}\033[0m")
    try:
        result = await registry.execute(tool_call.tool_name, arguments_dict, context)
        result_error = _result_error(result)
        if result_error:
            logger.error(
                f"tool_executor_error_002: Tool \033[31m{tool_call.tool_name}\033[0m returned an error: {result_error}"
            )
            await notifier.emit(
                "tools",
                "failed",
                f"{tool_call.tool_name} failed: {result_error}",
                detail=detail,
            )
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                output=result,
                error=result_error,
            )
        logger.info(
            f"tool_executor_003: Tool \033[36m{tool_call.tool_name}\033[0m executed successfully"
        )
        await notifier.emit(
            "tools",
            "completed",
            detail or f"{tool_call.tool_name} completed",
            detail=detail,
        )
        return ToolResult(
            tool_name=tool_call.tool_name,
            success=True,
            output=result,
        )
    except Exception as e:
        logger.error(
            f"tool_executor_error_001: Tool \033[31m{tool_call.tool_name}\033[0m failed: {e}"
        )
        await notifier.emit(
            "tools",
            "failed",
            f"{tool_call.tool_name} failed: {e!s}",
            detail=detail,
        )
        return ToolResult(
            tool_name=tool_call.tool_name,
            success=False,
            output={},
            error=str(e),
        )


async def execute_tool_calls(
    tool_calls: list[ToolCall],
    registry: ToolRegistry,
    allowed: Collection[str],
    on_status: StatusCallback = None,
    context: ToolContext | None = None,
) -> list[ToolResult]:
    """Run calls in parallel. Only tools in `allowed` (the tools this agent kind
    shows to its LLM) run; the LLM may still name another tool from ToolName."""
    calls: list[ToolCall] = []
    rejected: list[ToolResult] = []
    for tool_call in tool_calls:
        if tool_call.tool_name not in allowed:
            logger.warning(
                f"tool_executor_warning_002: \033[33m{tool_call.tool_name}\033[0m is not available here"
            )
            rejected.append(
                ToolResult(
                    tool_name=tool_call.tool_name,
                    success=False,
                    output={},
                    error=f"Unknown tool: {tool_call.tool_name}",
                )
            )
            continue
        try:
            calls += expand_repeated_arguments(tool_call)
        except RepeatedArgumentsError as e:
            logger.warning(f"tool_executor_warning_001: {e}")
            rejected.append(
                ToolResult(
                    tool_name=tool_call.tool_name,
                    success=False,
                    output={},
                    error=str(e),
                )
            )
    if len(calls) > len(tool_calls) - len(rejected):
        logger.info(
            f"tool_executor_006: Repeated arguments expanded into \033[33m{len(calls)}\033[0m calls"
        )
    logger.info(f"tool_executor_004: Executing \033[33m{len(calls)}\033[0m tools")
    tasks = [execute_tool_call(call, registry, on_status, context) for call in calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    successful = sum(1 for r in results if isinstance(r, ToolResult) and r.success)
    logger.info(
        f"tool_executor_005: Completed: \033[32m{successful}\033[0m successful, \033[31m{len(results) - successful}\033[0m failed"
    )
    return [
        (
            r
            if isinstance(r, ToolResult)
            else ToolResult(
                tool_name="unknown",
                success=False,
                output={},
                error=str(r),
            )
        )
        for r in results
    ] + rejected
