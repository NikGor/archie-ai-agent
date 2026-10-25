"""The tool loop every tool-calling agent kind shares (ARCHIE-298): call the
LLM, run the tools it asks for, show the results, repeat until it replies or
the budget ends the rounds."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any
from pydantic import BaseModel
from .budget import BudgetMeter

TurnT = BaseModel


async def run_tool_loop(
    call: Callable[[type[BaseModel]], Awaitable[BaseModel]],
    tool_calls_of: Callable[[BaseModel], Sequence[Any]],
    run_tools: Callable[[BaseModel], Awaitable[tuple[str, bool]]],
    messages: list[dict[str, Any]],
    turn_schema: type[BaseModel],
    reply_schema: type[BaseModel],
    final_task: str,
    meter: BudgetMeter,
) -> BaseModel:
    """`call(schema)` asks the LLM with `messages`; `run_tools(turn)` returns the
    results text and True when only the reply is left (e.g. after a side effect).
    The last round asks with `reply_schema`, which has no tool calls."""
    turn = await call(turn_schema)
    while isinstance(turn, turn_schema) and tool_calls_of(turn):
        meter.add_round()
        results, done = await run_tools(turn)
        messages.append({"role": "system", "content": results})
        last_round = done or meter.last_round()
        messages.append(
            {
                "role": "user",
                "content": final_task
                if last_round
                else f"Call more tools if you still need them. Else: {final_task}",
            }
        )
        turn = await call(reply_schema if last_round else turn_schema)
    return turn
