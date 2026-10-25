"""Execute one finite read-only task and retain evidence when a limit ends it."""

import asyncio
import json
import logging
import math
import uuid
from collections.abc import Awaitable, Callable
from archie_shared.chat.models import LllmTrace
from pydantic import BaseModel, Field
from app.agents.ephemeral.messages import build_messages
from app.agents.ephemeral.models import (
    DEFAULT_COST_USD,
    DEFAULT_SECONDS,
    DEFAULT_TOKENS,
    DEFAULT_TOOL_ROUNDS,
    READ_TOOLS,
    EphemeralReply,
    EphemeralResult,
    EphemeralTurn,
    EphemeralUsage,
    SpawnRequest,
    validate_budget_limits,
)
from app.config import settings
from app.platform.llm.catalog import MODEL_TOKEN_PRICES, get_provider_for_model
from app.platform.llm.registry import get_client
from app.platform.prompts.compiler import PromptCompiler
from app.platform.runtime.budget import (
    Budget,
    BudgetExceededError,
    BudgetMeter,
    daily_spend,
)
from app.platform.runtime.spawn import SpawnContext
from app.platform.runtime.tool_loop import run_tool_loop
from app.platform.tools.executor import execute_tool_calls
from app.platform.tools.models import ToolCall, ToolContext, ToolResult
from app.platform.tools.registry import ToolRegistry
from app.tools.catalog import tool_registry
from app.utils.llm_parser import parse_assembled_stream


logger = logging.getLogger(__name__)


class Completion(BaseModel):
    content: BaseModel = Field(description="Validated structured model reply")
    trace: LllmTrace = Field(description="Observed usage of the completed call")


CompletionCall = Callable[
    [list[dict[str, str]], type[BaseModel], str, int], Awaitable[Completion]
]
ToolCallRunner = Callable[[list[ToolCall], ToolContext], Awaitable[list[ToolResult]]]


class BudgetEstimateError(RuntimeError):
    """A provider call cannot be safely priced before it starts."""


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


class EphemeralCancelledError(asyncio.CancelledError):
    """Cancellation with retained evidence for terminal result publication."""

    def __init__(self, result: EphemeralResult) -> None:
        super().__init__("Ephemeral run cancelled")
        self.result = result


def capped_budget(requested: Budget, ceiling: Budget) -> Budget:
    """Missing requested limits never remove the service's mandatory caps."""
    validate_budget_limits(requested)
    validate_budget_limits(ceiling)
    values = {}
    defaults = Budget(
        max_tokens=DEFAULT_TOKENS,
        max_cost_usd=DEFAULT_COST_USD,
        max_seconds=DEFAULT_SECONDS,
        max_tool_rounds=DEFAULT_TOOL_ROUNDS,
    )
    for name, default in defaults.model_dump().items():
        cap = getattr(ceiling, name)
        cap = default if cap is None else min(cap, default)
        value = getattr(requested, name)
        values[name] = cap if value is None else min(value, cap)
    return Budget(**values)


async def complete(
    messages: list[dict[str, str]],
    schema: type[BaseModel],
    model: str,
    max_output_tokens: int,
) -> Completion:
    """Use the shared streaming clients so each call also has an output cap."""
    client = get_client(get_provider_for_model(model))
    usage: dict[str, int] = {}
    chunks = []
    async for chunk in client.create_completion_stream(
        messages=messages,
        model=model,
        response_format=schema,
        max_output_tokens=max_output_tokens,
        usage_out=usage,
        observation_name="ephemeral-step",
    ):
        chunks.append(chunk)
    parsed = parse_assembled_stream("".join(chunks), model, schema, usage)
    return Completion(content=parsed.parsed_content, trace=parsed.llm_trace)


class EphemeralRunner:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        call: CompletionCall | None = None,
        run_tools: ToolCallRunner | None = None,
        compiler: PromptCompiler | None = None,
        model: str | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.registry = registry or tool_registry
        self.call = call or complete
        self.run_tools = run_tools or self._run_tools
        self.compiler = compiler or PromptCompiler()
        self.model = model or settings.swarm_model
        self.budget = budget or Budget()
        validate_budget_limits(self.budget)

    async def _run_tools(
        self, calls: list[ToolCall], context: ToolContext
    ) -> list[ToolResult]:
        return await execute_tool_calls(
            calls, self.registry, allowed=READ_TOOLS, context=context
        )

    async def run(
        self, request: SpawnRequest, caller: SpawnContext, run_id: str | None = None
    ) -> EphemeralResult:
        meter = BudgetMeter(capped_budget(request.budget, self.budget))
        evidence: list[ToolResult] = []
        traces: list[LllmTrace] = []
        reply: EphemeralReply | None = None
        status = "completed"
        cancelled = False
        summary = ""
        identifier = run_id or uuid.uuid4().hex[:16]

        def check_remaining() -> None:
            meter.check()
            for name, used in (
                ("max_tokens", meter.tokens),
                ("max_cost_usd", meter.cost),
                ("max_seconds", meter.seconds),
            ):
                limit = getattr(meter.budget, name)
                if used >= limit:
                    raise BudgetExceededError(name, used, limit)

        async def call(schema: type[BaseModel]) -> BaseModel:
            nonlocal reply
            check_remaining()
            input_bound, input_cost = estimate_call(messages, schema, self.model, 0)
            remaining_tokens = int(meter.budget.max_tokens - meter.tokens)
            output_cap = min(2048, remaining_tokens - input_bound)
            if output_cap < 1:
                raise BudgetExceededError(
                    "max_tokens",
                    meter.tokens + input_bound + 1,
                    meter.budget.max_tokens,
                )
            remaining_cost = meter.budget.max_cost_usd - meter.cost
            _, one_token_cost = estimate_call(messages, schema, self.model, 1)
            output_token_price = one_token_cost - input_cost
            if remaining_cost <= input_cost:
                raise BudgetExceededError(
                    "max_cost_usd",
                    meter.cost + one_token_cost,
                    meter.budget.max_cost_usd,
                )
            if output_token_price > 0:
                output_cap = min(
                    output_cap,
                    math.floor((remaining_cost - input_cost) / output_token_price),
                )
            if output_cap < 1:
                raise BudgetExceededError(
                    "max_cost_usd",
                    meter.cost + one_token_cost,
                    meter.budget.max_cost_usd,
                )
            completion = await self.call(messages, schema, self.model, output_cap)
            if not isinstance(completion.content, EphemeralReply):
                raise ValueError("Ephemeral completion did not return a reply")
            reply = completion.content
            traces.append(completion.trace)
            daily_spend.add("ephemeral", completion.trace.total_cost or 0.0)
            meter.add(completion.trace)
            return completion.content

        async def tools(turn: BaseModel) -> tuple[str, bool]:
            assert isinstance(turn, EphemeralTurn)
            check_remaining()
            permitted = []
            for tool_call in turn.tool_calls:
                if (
                    tool_call.tool_name not in READ_TOOLS
                    or tool_call.tool_name not in request.tools
                ):
                    evidence.append(
                        ToolResult(
                            tool_name=tool_call.tool_name,
                            output={"error": "read-only policy blocked this call"},
                            success=False,
                            error="read-only policy",
                        )
                    )
                else:
                    permitted.append(tool_call)
            if permitted:
                evidence.extend(
                    await self.run_tools(
                        permitted,
                        ToolContext(
                            user_timezone=caller.user_timezone,
                            demo_mode=caller.demo_mode,
                        ),
                    )
                )
            meter.check()
            visible = json.dumps(
                [result.model_dump() for result in evidence], ensure_ascii=False
            ).encode("utf-8")
            if len(visible) > 4000:
                visible = visible[:4000]
                text = visible.decode("utf-8", errors="ignore") + (
                    "\n[Evidence clipped to the run budget. Full read results remain in the record. "
                    "Do not infer facts from the omitted evidence.]"
                )
            else:
                text = visible.decode("utf-8")
            return text, False

        try:
            if not set(request.tools) <= READ_TOOLS:
                raise PermissionError("Only approved read tools are allowed")
            messages = build_messages(request, self.model, self.registry, self.compiler)
            timeout = min(request.ttl, meter.budget.max_seconds) - meter.seconds
            if timeout <= 0:
                raise TimeoutError("wall time")
            async with asyncio.timeout(timeout):
                await run_tool_loop(
                    call,
                    lambda turn: (
                        turn.tool_calls if isinstance(turn, EphemeralTurn) else []
                    ),
                    tools,
                    messages,
                    EphemeralTurn if meter.budget.max_tool_rounds else EphemeralReply,
                    EphemeralReply,
                    "Report the diagnosis and the evidence gaps. Do not take action.",
                    meter,
                )
            summary = reply.summary if reply else "No diagnosis was produced."
        except asyncio.CancelledError:
            status = "partial"
            cancelled = True
            summary = "Stopped because the parent run was cancelled."
            logger.info(f"ephemeral_runner_002: {identifier} {summary}")
        except (TimeoutError, BudgetExceededError, BudgetEstimateError) as exc:
            status = "partial"
            summary = f"Stopped at the run limit: {exc or 'wall time'}."
            logger.warning(f"ephemeral_runner_warning_001: {identifier} {summary}")
        except Exception as exc:
            status = "error"
            summary = f"Diagnosis failed: {type(exc).__name__}: {exc}"
            logger.exception(f"ephemeral_runner_error_001: {identifier} {summary}")
        findings = list(reply.findings) if reply else []
        if status != "completed":
            findings.extend(
                f"{item.tool_name}: {json.dumps(item.output, ensure_ascii=False)[:1600]}"
                for item in evidence
            )
        result = EphemeralResult(
            run_id=identifier,
            parent_run_id=caller.parent_run_id,
            status=status,
            summary=summary,
            findings=findings,
            evidence=evidence,
            usage=EphemeralUsage(
                input_tokens=sum(t.input_tokens for t in traces),
                output_tokens=sum(t.output_tokens for t in traces),
                total_cost=meter.cost,
                seconds=round(meter.seconds, 3),
                tool_rounds=meter.rounds,
            ),
        )
        if cancelled:
            raise EphemeralCancelledError(result)
        return result
