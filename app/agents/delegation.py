"""Composition of chat delegation (ARCHIE-324): the chat's delegate_agent_tool
reaches DynamicAgents of this process through the platform hook. The chat
waits for the result; closing the WebSocket cancels the child run."""

import logging
from archie_shared.chat.models import ReasoningEffort
from pydantic import BaseModel, JsonValue
from app.agents.ephemeral.configurator import AgentConfigurator
from app.agents.ephemeral.dynamic import DynamicAgents
from app.agents.ephemeral.engine import AgentRuntime
from app.backend.redis_factory import get_async_redis
from app.config import settings
from app.platform.policy.spawn import SpawnDeniedError, SpawnGate
from app.platform.runtime.budget import Budget, BudgetExceededError
from app.platform.runtime.completion import (
    Completion,
    InvalidOutputError,
    complete,
    estimate_call,
)
from app.platform.runtime.daily_budget import DailyBudget, DailyBudgetExceeded
from app.platform.runtime.delegation import (
    DelegationCaller,
    StepSink,
    register_delegate_handler,
)
from app.platform.runtime.runs import (
    EventListener,
    RedisRunStore,
    RunEvent,
    RunManager,
    register_run_manager,
)
from app.platform.runtime.spec import CallerPolicy, Execution
from app.platform.runtime.status import register_status
from app.tools.catalog import tool_registry


logger = logging.getLogger(__name__)

# Ceilings of one delegated chat run (ARCHIE-324)
CHAT_BUDGET = Budget(
    max_tokens=60000, max_cost_usd=0.10, max_seconds=150, max_tool_rounds=4
)
CHAT_LIMITS = Execution(
    max_iterations=4, max_tool_calls=8, timeout_seconds=150, retries=1
)
APPROVAL_NOTE = (
    "The agent changed nothing. Ask the user to confirm each proposed action; "
    "run it only after the user confirms."
)


def step_text(event: RunEvent) -> str | None:
    """WS status detail for the events a user cares about."""
    detail = event.detail
    if event.kind == "agent_configured" and detail.get("name"):
        return f"Agent {detail['name']}: {', '.join(detail.get('tools') or []) or 'no tools'}"
    if event.kind == "tool_invoked":
        return f"Agent: {detail.get('tool')}"
    if event.kind == "model_escalated":
        return f"Agent: escalated to {detail.get('to_profile')}"
    return None


def step_listener(on_step: StepSink | None) -> EventListener | None:
    if on_step is None:
        return None

    async def listen(event: RunEvent) -> None:
        text = step_text(event)
        if text:
            await on_step(text)

    return listen


def chat_policy(caller: DelegationCaller) -> CallerPolicy:
    """The chat's own tools; side effects only as proposals the user confirms."""
    return CallerPolicy.from_spawn(
        caller.caller,
        caller.tools,
        budget=CHAT_BUDGET,
        limits=CHAT_LIMITS,
        # The user waits in the chat: DEEP (effort high) took 40-100 s per run
        max_profile="BALANCED",
        max_effort="medium",
        may_request_approval=True,
    )


def configure_delegation() -> DynamicAgents:
    ledger = DailyBudget(
        get_async_redis(),
        "agent",
        settings.default_timezone,
        settings.agent_daily_cost_usd,
    )

    async def bounded_call(
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        model: str,
        max_output_tokens: int,
        effort: ReasoningEffort | None = None,
    ) -> Completion:
        _, ceiling = estimate_call(messages, schema, model, max_output_tokens)
        try:
            reservation = await ledger.reserve(ceiling, ttl=180)
        except DailyBudgetExceeded as exc:
            raise BudgetExceededError(
                "daily_cost_usd", ceiling, settings.agent_daily_cost_usd
            ) from exc
        try:
            result = await complete(messages, schema, model, max_output_tokens, effort)
        except InvalidOutputError as exc:
            await ledger.settle(
                reservation, (exc.trace.total_cost or 0.0) if exc.trace else 0.0
            )
            raise
        await ledger.settle(reservation, result.trace.total_cost or 0.0)
        return result

    manager = RunManager(RedisRunStore(get_async_redis()))
    agents = DynamicAgents(
        manager=manager,
        runtime=AgentRuntime(tool_registry, bounded_call),
        configurator=AgentConfigurator(tool_registry, bounded_call),
    )
    gate = SpawnGate(
        frozenset({"chat"}),
        frozenset(tool_registry.names()),
        max_concurrent=2,
        max_depth=1,
    )

    async def handle(
        task: str, context: str, caller: DelegationCaller
    ) -> dict[str, JsonValue]:
        try:
            gate.acquire(caller.caller, ())
        except SpawnDeniedError as exc:
            return {"state": "failed", "stop_reason": "blocked", "summary": str(exc)}
        try:
            record = await agents.delegate(
                task, context, chat_policy(caller), step_listener(caller.on_step)
            )
        finally:
            gate.release()
        summary = record.summary()
        if record.stop_reason == "direct":
            summary["note"] = (
                "No agent was needed: " + record.detail + ". Do the task yourself."
            )
        elif record.state == "waiting_for_approval":
            summary["note"] = APPROVAL_NOTE
        return summary

    register_delegate_handler(handle)
    register_run_manager(manager)

    async def status() -> dict[str, JsonValue]:
        return {
            **manager.stats(),
            "active_gate": gate.active,
            "daily_spend_usd": await ledger.spent(),
            "daily_limit_usd": settings.agent_daily_cost_usd,
        }

    register_status("agents", status)
    logger.info("delegation_001: Dynamic agents ready for chat delegation")
    return agents
