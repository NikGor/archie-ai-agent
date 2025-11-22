"""Utilities for executing tools based on orchestration decisions."""

import logging
from typing import Any
from ..models.orchestration_sgr import ToolCallRequest, Parameter
from ..tools.tool_factory import ToolFactory


logger = logging.getLogger(__name__)


def convert_parameters_to_dict(parameters: list[Parameter]) -> dict[str, Any]:
    """
    Convert list of Parameter objects to dict for tool execution.

    Args:
        parameters: List of Parameter objects with name/value pairs

    Returns:
        dict: Dictionary mapping parameter names to values
    """
    return {param.name: param.value for param in parameters}


async def execute_tool_call(
    tool_call: ToolCallRequest,
    tool_factory: ToolFactory | None = None,
) -> dict[str, Any]:
    """
    Execute a single tool call from orchestration decision.

    Args:
        tool_call: ToolCallRequest object with tool name, arguments, etc.
        tool_factory: Optional ToolFactory instance (creates new if not provided)

    Returns:
        dict: Result containing tool_name, success status, and output/error
    """
    logger.info(
        f"tool_executor_001: Executing tool: \033[36m{tool_call.tool_name}\033[0m"
    )

    if tool_factory is None:
        tool_factory = ToolFactory()

    arguments_dict = convert_parameters_to_dict(tool_call.arguments)
    logger.info(f"tool_executor_002: Arguments: \033[33m{arguments_dict}\033[0m")

    try:
        result = await tool_factory.execute_tool(
            tool_name=tool_call.tool_name,
            tool_arguments=arguments_dict,
        )

        logger.info(
            f"tool_executor_003: Tool \033[36m{tool_call.tool_name}\033[0m executed successfully"
        )

        return {
            "tool_name": tool_call.tool_name,
            "success": True,
            "output": result,
        }

    except Exception as e:
        logger.error(
            f"tool_executor_error_001: Tool \033[31m{tool_call.tool_name}\033[0m failed: {e}"
        )

        return {
            "tool_name": tool_call.tool_name,
            "success": False,
            "error": str(e),
        }


async def execute_tool_calls_parallel(
    tool_calls: list[ToolCallRequest],
    tool_factory: ToolFactory | None = None,
) -> list[dict[str, Any]]:
    """
    Execute multiple tool calls in parallel.

    Args:
        tool_calls: List of ToolCallRequest objects to execute
        tool_factory: Optional ToolFactory instance (creates new if not provided)

    Returns:
        list: List of results from all tool executions
    """
    import asyncio

    logger.info(
        f"tool_executor_004: Executing \033[33m{len(tool_calls)}\033[0m tools in parallel"
    )

    if tool_factory is None:
        tool_factory = ToolFactory()

    tasks = [execute_tool_call(tool_call, tool_factory) for tool_call in tool_calls]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    logger.info(
        f"tool_executor_005: Completed: \033[32m{successful}\033[0m successful, \033[31m{len(results) - successful}\033[0m failed"
    )

    return [
        r if isinstance(r, dict) else {"success": False, "error": str(r)}
        for r in results
    ]
