"""Budget-bounded structured LLM calls of agent runs (ARCHIE-300, ARCHIE-321).

Every call is priced before it starts and gets an output cap that fits the
remaining tokens and money; unknown prices fail closed.
"""

import json
import math
from collections.abc import Awaitable, Callable
from archie_shared.chat.models import LllmTrace, ReasoningEffort
from pydantic import BaseModel, Field
from app.platform.llm.catalog import MODEL_TOKEN_PRICES, get_provider_for_model
from app.platform.llm.openai_client import TruncatedOutputError
from app.platform.llm.registry import get_client
from app.platform.runtime.budget import BudgetExceededError, BudgetMeter
from app.utils.llm_parser import build_stream_trace, parse_assembled_stream


MAX_OUTPUT_TOKENS = 2048
# Reasoning tokens count as output: a higher effort needs a higher cap (ARCHIE-321)
EFFORT_OUTPUT_TOKENS = {"none": 2048, "minimal": 2048, "low": 4096, "medium": 8192}
REASONING_OUTPUT_TOKENS = 16384


def max_output_for(effort: str | None) -> int:
    if effort is None:
        return MAX_OUTPUT_TOKENS
    return EFFORT_OUTPUT_TOKENS.get(effort, REASONING_OUTPUT_TOKENS)


class Completion(BaseModel):
    content: BaseModel = Field(description="Validated structured model reply")
    trace: LllmTrace = Field(description="Observed usage of the completed call")


CompletionCall = Callable[..., Awaitable[Completion]]


class BudgetEstimateError(RuntimeError):
    """A provider call cannot be safely priced before it starts."""


class InvalidOutputError(ValueError):
    """The model reply does not match the schema; the trace still counts."""

    def __init__(self, message: str, trace: LllmTrace | None = None) -> None:
        super().__init__(message)
        self.trace = trace


def estimate_call(
    messages: list[dict[str, str]],
    schema: type[BaseModel],
    model: str,
    max_output_tokens: int,
) -> tuple[int, float]:
    """Conservative token and uncached USD ceilings for one structured call.

    Byte-level tokenizers cannot produce more text tokens than UTF-8 bytes.
    Include the complete schema and extra protocol framing before reserving
    money; cached input discounts never reduce the reservation. Unknown model
    prices fail closed rather than silently treating autonomous work as free.
    """
    key = max(
        (name for name in MODEL_TOKEN_PRICES if model.startswith(name)),
        key=len,
        default=None,
    )
    if key is None:
        raise BudgetEstimateError(f"No known token prices for model {model}")
    if max_output_tokens < 0:
        raise BudgetEstimateError("The output token bound must be nonnegative")
    prices = MODEL_TOKEN_PRICES[key]
    if any(
        not math.isfinite(prices[name]) or prices[name] < 0
        for name in ("input", "output")
    ):
        raise BudgetEstimateError(f"Invalid token prices for model {model}")
    input_upper_bound = (
        len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
        + len(
            json.dumps(schema.model_json_schema(), ensure_ascii=False).encode("utf-8")
        )
        + 1024
        + 64 * len(messages)
    )
    cost_upper_bound = (
        input_upper_bound * prices["input"] + max_output_tokens * prices["output"]
    ) / 1_000_000
    return input_upper_bound, cost_upper_bound


async def complete(
    messages: list[dict[str, str]],
    schema: type[BaseModel],
    model: str,
    max_output_tokens: int,
    effort: ReasoningEffort | None = None,
) -> Completion:
    """Use the shared streaming clients so each call also has an output cap."""
    client = get_client(get_provider_for_model(model))
    usage: dict[str, int] = {}
    chunks = []
    try:
        async for chunk in client.create_completion_stream(
            messages=messages,
            model=model,
            response_format=schema,
            max_output_tokens=max_output_tokens,
            usage_out=usage,
            observation_name="ephemeral-step",
            reasoning_effort=effort,
        ):
            chunks.append(chunk)
    except TruncatedOutputError as exc:
        raise InvalidOutputError(
            f"output cut off at {max_output_tokens} tokens",
            build_stream_trace(model, usage) if usage else None,
        ) from exc
    try:
        parsed = parse_assembled_stream("".join(chunks), model, schema, usage)
    except ValueError as exc:
        raise InvalidOutputError(
            f"{type(exc).__name__}: {str(exc)[:300]}",
            build_stream_trace(model, usage) if usage else None,
        ) from exc
    return Completion(content=parsed.parsed_content, trace=parsed.llm_trace)


def check_remaining(meter: BudgetMeter) -> None:
    """Stop before a step when a limit is already used up (not only exceeded)."""
    meter.check()
    for name, used in (
        ("max_tokens", meter.tokens),
        ("max_cost_usd", meter.cost),
        ("max_seconds", meter.seconds),
    ):
        limit = getattr(meter.budget, name)
        if used >= limit:
            raise BudgetExceededError(name, used, limit)


def output_cap(
    meter: BudgetMeter,
    messages: list[dict[str, str]],
    schema: type[BaseModel],
    model: str,
    maximum: int = MAX_OUTPUT_TOKENS,
) -> int:
    """Largest output that keeps the call inside the remaining tokens and money."""
    check_remaining(meter)
    input_bound, input_cost = estimate_call(messages, schema, model, 0)
    remaining_tokens = int(meter.budget.max_tokens - meter.tokens)
    cap = min(maximum, remaining_tokens - input_bound)
    if cap < 1:
        raise BudgetExceededError(
            "max_tokens", meter.tokens + input_bound + 1, meter.budget.max_tokens
        )
    remaining_cost = meter.budget.max_cost_usd - meter.cost
    _, one_token_cost = estimate_call(messages, schema, model, 1)
    output_token_price = one_token_cost - input_cost
    if remaining_cost <= input_cost:
        raise BudgetExceededError(
            "max_cost_usd", meter.cost + one_token_cost, meter.budget.max_cost_usd
        )
    if output_token_price > 0:
        cap = min(cap, math.floor((remaining_cost - input_cost) / output_token_price))
    if cap < 1:
        raise BudgetExceededError(
            "max_cost_usd", meter.cost + one_token_cost, meter.budget.max_cost_usd
        )
    return cap
