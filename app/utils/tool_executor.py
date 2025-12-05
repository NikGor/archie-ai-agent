import logging
import asyncio
from collections.abc import Callable, Awaitable
from typing import Any
from ..models.orchestration_sgr import ToolCallRequest, Parameter
from ..models.ws_models import StatusUpdate
from ..tools.tool_factory import ToolFactory


logger = logging.getLogger(__name__)

StatusCallback = Callable[[StatusUpdate], Awaitable[None]] | None


def convert_parameters_to_dict(parameters: list[Parameter]) -> dict[str, Any]:
    return {param.name: param.value for param in parameters}


async def execute_tool_call(
    tool_call: ToolCallRequest,
    tool_factory: ToolFactory | None = None,
    on_status: StatusCallback = None,
) -> dict[str, Any]:
    logger.info(
        f"tool_executor_001: Executing tool: \033[36m{tool_call.tool_name}\033[0m"
    )
    if on_status:
        await on_status(
            StatusUpdate(
                step="tools",
                status="started",
                message=f"Calling {tool_call.tool_name}",
            )
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
        if on_status:
            await on_status(
                StatusUpdate(
                    step="tools",
                    status="completed",
                    message=f"{tool_call.tool_name} completed",
                )
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
        if on_status:
            await on_status(
                StatusUpdate(
                    step="tools",
                    status="failed",
                    message=f"{tool_call.tool_name} failed: {e!s}",
                )
            )
        return {
            "tool_name": tool_call.tool_name,
            "success": False,
            "error": str(e),
        }


async def execute_tool_calls(
    tool_calls: list[ToolCallRequest],
    tool_factory: ToolFactory | None = None,
    on_status: StatusCallback = None,
) -> list[dict[str, Any]]:
    logger.info(f"tool_executor_004: Executing \033[33m{len(tool_calls)}\033[0m tools")
    if tool_factory is None:
        tool_factory = ToolFactory()
    tasks = [
        execute_tool_call(tool_call, tool_factory, on_status)
        for tool_call in tool_calls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    logger.info(
        f"tool_executor_005: Completed: \033[32m{successful}\033[0m successful, \033[31m{len(results) - successful}\033[0m failed"
    )
    return [
        r if isinstance(r, dict) else {"success": False, "error": str(r)}
        for r in results
    ]
