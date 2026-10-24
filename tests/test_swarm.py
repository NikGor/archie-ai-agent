"""Tests for the agent swarm chat (ARCHIE-199)."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from archie_shared.swarm.models import (
    SWARM_TEXT_MAX_LENGTH,
    SwarmMessage,
    SwarmMessageEvent,
    SwarmSettings,
    SwarmUsage,
)
from app.config import settings
from app.models.orchestration_sgr import Parameter
from app.models.tool_models import ToolContext, ToolResult
from app.swarm.agents import SWARM_AGENTS, SwarmRoster
from app.swarm.bus import SwarmBus, SwarmHub, clip_text, parse_mentions
from app.swarm.council import CouncilPlan, SwarmCouncil
from app.swarm.dispatcher import SwarmDispatcher
from app.swarm.models import (
    ConsulDecision,
    ConsulTurn,
    CouncilStep,
    SwarmReply,
    SwarmToolCall,
    SwarmTurn,
    SwarmWatchReply,
    SwarmWatchTurn,
)
from app.swarm.policy import READ_ONLY_TOOLS, check_tool_call
from app.swarm.runner import SwarmAgentRunner, SwarmDraft, format_chat_line
from app.tools.tool_factory import ToolFactory
from tests.conftest import make_user_state


def _call(tool_name: str, **arguments: str) -> SwarmToolCall:
    return SwarmToolCall(
        tool_name=tool_name,
        arguments=[Parameter(name=k, value=v) for k, v in arguments.items()],
    )


def _message(
    sender: str = "user", text: str = "hi", depth: int = 0, mentions=None
) -> SwarmMessage:
    return SwarmMessage(
        message_id="1-0",
        created_at="2026-10-13T10:00:00+00:00",
        sender=sender,
        kind="user" if sender == "user" else "report",
        text=text,
        mentions=mentions if mentions is not None else parse_mentions(text),
        depth=depth,
    )


# --- policy ---


def test_action_agent_may_run_whitelisted_action():
    call = _call("smarthome_control_tool", action="turn_off", area="Гостиная")
    assert check_tool_call(SWARM_AGENTS["light"], call) == "allow"


def test_action_agent_blocked_outside_its_actions():
    call = _call("smarthome_control_tool", action="set_temperature", temperature="22")
    assert check_tool_call(SWARM_AGENTS["light"], call) == "block"


def test_state_agent_cannot_act():
    call = _call("smarthome_control_tool", action="turn_on")
    assert check_tool_call(SWARM_AGENTS["football"], call) == "block"


def test_agent_cannot_use_tool_outside_profile():
    assert (
        check_tool_call(
            SWARM_AGENTS["football"], _call("smarthome_status_tool", query="x")
        )
        == "block"
    )


def test_football_sees_long_search_results():
    assert SWARM_AGENTS["football"].tool_result_max_chars == 4000


def test_read_only_tool_in_profile_is_allowed():
    assert (
        check_tool_call(
            SWARM_AGENTS["football"], _call("google_search_tool", query="x")
        )
        == "allow"
    )


# --- bus ---


async def test_bus_round_trips_message_through_stream():
    redis = AsyncMock()
    redis.xadd = AsyncMock(return_value="5-0")
    with patch("app.swarm.bus.get_async_redis", return_value=redis):
        bus = SwarmBus(SwarmHub())
    queue = bus.hub.subscribe()
    posted = await bus.post(
        sender="light", kind="report", text="Свет выключен, @climate"
    )
    assert posted.message_id == "5-0"
    assert posted.mentions == ["climate"]
    assert queue.get_nowait().message == posted
    stored = redis.xadd.await_args.args[1]
    redis.xrevrange = AsyncMock(return_value=[("5-0", stored)])
    assert await bus.get_history(10) == [posted]


def test_parse_mentions_keeps_known_agents_only():
    assert parse_mentions("@Light выключи, @nobody, @climate и снова @light") == [
        "light",
        "climate",
    ]


def test_clip_text_limits_length():
    assert len(clip_text("a " * 400)) == SWARM_TEXT_MAX_LENGTH == 600


# --- dispatcher guards ---


ALL_AGENTS = set(SWARM_AGENTS)
COUNCIL = ["butler", "business", "bro", "flirty", "marcus"]


def _dispatcher() -> SwarmDispatcher:
    roster = MagicMock()
    roster.get_enabled_ids = AsyncMock(return_value=ALL_AGENTS)
    return SwarmDispatcher(
        hub=SwarmHub(),
        bus=MagicMock(post=AsyncMock()),
        roster=roster,
        runner=MagicMock(),
        council=MagicMock(discuss=AsyncMock(return_value=None), execute=AsyncMock()),
        memory=MagicMock(get_mode=AsyncMock(return_value="running")),
    )


def test_mentions_wake_mentioned_agents_directly():
    dispatcher = _dispatcher()
    message = _message(text="@light и @climate, спать")
    assert dispatcher.select_targets(message, ALL_AGENTS) == ["light", "climate"]
    assert not SwarmDispatcher.wants_council(message)


def test_user_message_without_mentions_goes_to_council():
    message = _message(text="какой свет мне включить?")
    assert SwarmDispatcher.wants_council(message)
    assert _dispatcher().select_targets(message, ALL_AGENTS) == []


def test_council_mention_goes_to_council_not_to_members():
    message = _message(text="@council что посмотреть?")
    assert SwarmDispatcher.wants_council(message)
    assert _dispatcher().select_targets(message, ALL_AGENTS) == []


def test_agent_message_without_mentions_does_not_go_to_council():
    assert not SwarmDispatcher.wants_council(_message(sender="light", text="готово"))


def test_opinions_wake_nobody():
    message = _message(sender="bro", text="@light включи всё", mentions=["light"])
    message = message.model_copy(update={"kind": "opinion"})
    assert _dispatcher().select_targets(message, ALL_AGENTS) == []
    assert not SwarmDispatcher.wants_council(message)


def test_decision_delegates_to_mentioned_agent():
    message = _message(sender="business", text="@light включи торшер", depth=1)
    message = message.model_copy(update={"kind": "decision"})
    assert _dispatcher().select_targets(message, ALL_AGENTS) == ["light"]


def test_depth_limit_stops_chain():
    message = _message(sender="light", text="@climate", depth=settings.swarm_max_depth)
    assert _dispatcher().select_targets(message, ALL_AGENTS) == []


def test_agent_does_not_wake_itself():
    message = _message(sender="light", text="x", mentions=["light"])
    assert _dispatcher().select_targets(message, ALL_AGENTS) == []


def test_disabled_and_running_agents_are_skipped():
    dispatcher = _dispatcher()
    dispatcher._running.add("climate")
    message = _message(text="@light @climate @football")
    assert dispatcher.select_targets(message, {"light", "climate"}) == ["light"]


def test_cooldown_applies_only_to_agent_triggers():
    dispatcher = _dispatcher()
    dispatcher._last_run["climate"] = 10**12
    with patch("app.swarm.dispatcher.time.monotonic", return_value=10**12 + 1):
        agent_message = _message(sender="light", text="@climate")
        assert dispatcher.select_targets(agent_message, ALL_AGENTS) == []
        user_message = _message(text="@climate")
        assert dispatcher.select_targets(user_message, ALL_AGENTS) == ["climate"]


async def test_council_runs_with_enabled_members():
    dispatcher = _dispatcher()
    await dispatcher._start_council(_message(text="@council?"), ALL_AGENTS - {"bro"})
    await asyncio.gather(*dispatcher._tasks)
    args = dispatcher.council.discuss.await_args.args
    assert args[1] == ["butler", "business", "flirty", "marcus"]
    assert not dispatcher._running


async def test_plan_runs_after_the_council_is_free():
    dispatcher = _dispatcher()
    plan = CouncilPlan(
        consul="butler",
        decision=_message(sender="butler", text="d"),
        steps=[CouncilStep(agent_id="light", task="выключи", wait=False)],
    )
    dispatcher.council.discuss = AsyncMock(return_value=plan)
    await dispatcher._start_council(_message(text="выключи свет"), ALL_AGENTS)
    await asyncio.gather(*dispatcher._tasks)
    await asyncio.gather(*dispatcher._tasks)
    dispatcher.council.execute.assert_awaited_once_with(plan)
    assert not dispatcher._running


async def test_busy_council_keeps_latest_message_and_answers_it_after():
    dispatcher = _dispatcher()
    dispatcher._running.add("bro")
    latest = _message(text="второй вопрос")
    await dispatcher._start_council(_message(text="первый"), ALL_AGENTS)
    await dispatcher._start_council(latest, ALL_AGENTS)
    assert dispatcher._pending["council"] is latest
    dispatcher.council.discuss.assert_not_awaited()
    dispatcher._running.clear()
    await dispatcher._run_council(_message(text="первый"), COUNCIL)
    await asyncio.gather(*dispatcher._tasks)
    assert dispatcher.council.discuss.await_args_list[-1].args[0] is latest


async def test_sleeping_council_posts_system_message_without_mentions():
    dispatcher = _dispatcher()
    await dispatcher._start_council(_message(text="привет"), {"light"})
    kwargs = dispatcher.bus.post.await_args.kwargs
    assert kwargs["kind"] == "system"
    assert "@" not in kwargs["text"]


async def test_council_failure_posts_system_message_and_frees_members():
    dispatcher = _dispatcher()
    dispatcher.council.discuss = AsyncMock(side_effect=RuntimeError("boom"))
    dispatcher._running.update(COUNCIL)
    await dispatcher._run_council(_message(text="x"), COUNCIL)
    assert dispatcher.bus.post.await_args.kwargs["text"] == "council failed: boom"
    assert not dispatcher._running


@pytest.fixture
def voice_listens(monkeypatch):
    """The registry has no listeners since ARCHIE-238; the mechanism stays."""
    monkeypatch.setattr(SWARM_AGENTS["voice"], "listens_to", ["light", "climate"])


def test_no_agent_listens_without_mentions():
    assert all(not profile.listens_to for profile in SWARM_AGENTS.values())


def test_listeners_hear_their_senders_only(voice_listens):
    assert SwarmDispatcher.select_listeners(_message(sender="light"), ALL_AGENTS) == [
        "voice"
    ]
    assert SwarmDispatcher.select_listeners(_message(sender="bro"), ALL_AGENTS) == []
    assert SwarmDispatcher.select_listeners(_message(sender="user"), ALL_AGENTS) == []
    assert (
        SwarmDispatcher.select_listeners(
            _message(sender="light"), ALL_AGENTS - {"voice"}
        )
        == []
    )


async def test_listener_ignores_depth_limit(voice_listens):
    dispatcher = _dispatcher()
    dispatcher._start = MagicMock()
    await dispatcher.handle(
        _message(sender="climate", text="батареи включены", depth=3)
    )
    started = [call.args[0] for call in dispatcher._start.call_args_list]
    assert started == ["voice"]


async def test_busy_listener_keeps_latest_event_and_runs_it_after(voice_listens):
    dispatcher = _dispatcher()
    dispatcher._running.add("voice")
    first = _message(sender="light", text="один")
    latest = _message(sender="climate", text="два")
    await dispatcher.handle(first)
    await dispatcher.handle(latest)
    assert dispatcher._pending["voice"] is latest
    dispatcher._run_agent = AsyncMock()
    dispatcher._start = MagicMock()
    await dispatcher._run_single("voice", first)
    dispatcher._start.assert_called_once_with("voice", latest)


async def test_run_agent_posts_reply_with_next_depth():
    dispatcher = _dispatcher()
    dispatcher.bus.get_history = AsyncMock(return_value=[])
    dispatcher.runner.run = AsyncMock(
        return_value=SwarmDraft(text="Готово", actions=[], usage={"model": "m"})
    )
    await dispatcher._run_agent("light", _message(text="@light", depth=1))
    kwargs = dispatcher.bus.post.await_args.kwargs
    assert kwargs["sender"] == "light"
    assert kwargs["kind"] == "report"
    assert kwargs["depth"] == 2
    assert "light" not in dispatcher._running


async def test_agent_reply_keeps_the_run_of_its_trigger():
    """ARCHIE-281: every message a trigger causes shares its run_id."""
    dispatcher = _dispatcher()
    dispatcher.bus.get_history = AsyncMock(return_value=[])
    dispatcher.runner.run = AsyncMock(
        return_value=SwarmDraft(text="Готово", actions=[], usage={"model": "m"})
    )
    trigger = _message(text="@light").model_copy(update={"run_id": "run-1"})
    await dispatcher._run_agent("light", trigger)
    assert dispatcher.bus.post.await_args.kwargs["run_id"] == "run-1"


async def test_failed_agent_note_keeps_the_run():
    dispatcher = _dispatcher()
    dispatcher.bus.get_history = AsyncMock(return_value=[])
    dispatcher.runner.run = AsyncMock(side_effect=RuntimeError("boom"))
    trigger = _message(text="@light").model_copy(update={"run_id": "run-1"})
    await dispatcher._run_agent("light", trigger)
    kwargs = dispatcher.bus.post.await_args.kwargs
    assert (kwargs["sender"], kwargs["run_id"]) == ("system", "run-1")


async def test_bus_starts_a_run_or_keeps_the_given_one():
    redis = AsyncMock()
    redis.xadd = AsyncMock(return_value="5-0")
    with patch("app.swarm.bus.get_async_redis", return_value=redis):
        bus = SwarmBus(SwarmHub())
    first = await bus.post(sender="user", kind="user", text="привет")
    second = await bus.post(sender="system", kind="system", text="x", run_id="run-1")
    assert first.run_id and first.run_id != "run-1"
    assert second.run_id == "run-1"


# --- council ---


def _council(answers: dict[str, list]) -> SwarmCouncil:
    """Each agent ID gets its own queue of answers: members run in parallel."""
    runner = MagicMock()
    runner.get_user_state = AsyncMock(return_value=make_user_state())
    runner.render_task = MagicMock(side_effect=lambda template, **_: template)
    usage = SwarmUsage(model="m", input_tokens=10, output_tokens=2, total_cost=0.001)
    queues = {agent_id: list(items) for agent_id, items in answers.items()}

    async def ask(profile, trigger, history, task, schema):
        return queues[profile.agent_id].pop(0), usage

    runner.ask = AsyncMock(side_effect=ask)
    bus = MagicMock(
        post=AsyncMock(return_value=_message(sender="butler", text="d")),
        get_history=AsyncMock(return_value=[]),
    )
    return SwarmCouncil(hub=SwarmHub(), bus=bus, runner=runner)


def _decision(reply: str = "", steps=None, **fields) -> ConsulDecision:
    return ConsulDecision(
        reasoning="r",
        reply=reply,
        steps=steps or [],
        remember=fields.get("remember", []),
        remember_today=[],
    )


def _more(remark: str) -> ConsulTurn:
    return ConsulTurn(
        reasoning="r",
        next_round=True,
        remark=remark,
        reply="",
        steps=[],
        remember=[],
        remember_today=[],
    )


def _done(reply: str, steps=None) -> ConsulTurn:
    return ConsulTurn(
        reasoning="r",
        next_round=False,
        remark="",
        reply=reply,
        steps=steps or [],
        remember=[],
        remember_today=[],
    )


MEMBERS = ["business", "bro", "flirty", "marcus"]


def _opinions(*rounds: str) -> dict[str, list]:
    return {m: [SwarmReply(reply=f"{m} {r}") for r in rounds] for m in MEMBERS}


def _asked(council: SwarmCouncil) -> list[tuple[str, str]]:
    return [
        (call.args[0].agent_id, call.args[3])
        for call in cast(AsyncMock, council.runner.ask).await_args_list
    ]


def _posted(council: SwarmCouncil) -> list[tuple[str, str, str]]:
    return [
        (c.kwargs["sender"], c.kwargs["kind"], c.kwargs["text"])
        for c in cast(AsyncMock, council.bus.post).await_args_list
    ]


async def test_reply_to_a_plan_step_wakes_nobody():
    dispatcher = _dispatcher()
    dispatcher.runner.run = AsyncMock(
        return_value=SwarmDraft(
            text="12 °C, @display", actions=[], usage=SwarmUsage(model="m")
        )
    )
    dispatcher.bus.get_history = AsyncMock(return_value=[])
    task = _message(sender="butler", text="1/2 @weather: погода").model_copy(
        update={"kind": "task"}
    )
    await dispatcher._run_agent("weather", task)
    assert dispatcher.bus.post.await_args.kwargs["mentions"] == []


def test_consul_is_the_chair_or_the_first_awake_member():
    assert SwarmCouncil.choose_consul(COUNCIL) == "butler"
    assert SwarmCouncil.choose_consul(["bro", "flirty"]) == "bro"


async def test_first_round_runs_in_parallel_on_one_chat_snapshot():
    council = _council({**_opinions("1"), "butler": [_done("Решено.")]})
    active, peak = 0, 0
    ask = council.runner.ask.side_effect

    async def counting_ask(*args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return await ask(*args)

    council.runner.ask.side_effect = counting_ask
    await council.discuss(_message(text="ехать или нет?"), COUNCIL)
    assert peak == len(MEMBERS)
    calls = council.runner.ask.await_args_list
    assert len({id(c.args[2]) for c in calls[: len(MEMBERS)]}) == 1
    assert _asked(council)[-1] == ("butler", "consul.jinja2")
    posted = _posted(council)
    assert sorted(posted[:-1]) == sorted((m, "opinion", f"{m} 1") for m in MEMBERS)
    assert posted[-1] == ("butler", "decision", "Решено.")
    rounds = {c.kwargs["round_number"] for c in council.bus.post.await_args_list[:-1]}
    assert rounds == {1}


async def test_consul_starts_another_round_with_a_remark():
    council = _council(
        {**_opinions("1", "2"), "butler": [_more("А соседи?"), _done("Итог.")]}
    )
    await council.discuss(_message(text="вечеринка в воскресенье?"), COUNCIL)
    posted = _posted(council)
    assert posted[len(MEMBERS)] == ("butler", "opinion", "А соседи?")
    assert {p[2] for p in posted[len(MEMBERS) + 1 : -1]} == {f"{m} 2" for m in MEMBERS}
    assert posted[-1] == ("butler", "decision", "Итог.")
    decision_usage = council.bus.post.await_args_list[-1].kwargs["usage"]
    assert decision_usage.input_tokens == 20


async def test_last_round_asks_for_the_decision_only(monkeypatch):
    monkeypatch.setattr("app.swarm.council.settings.swarm_council_max_rounds", 1)
    council = _council({**_opinions("1"), "butler": [_decision("Итог.")]})
    await council.discuss(_message(text="?"), COUNCIL)
    schema = council.runner.ask.await_args_list[-1].args[4]
    assert schema is ConsulDecision


async def test_silent_or_failed_member_does_not_stop_the_discussion():
    answers = {**_opinions("1"), "butler": [_done("Итог.")]}
    answers["marcus"] = [SwarmReply(reply="")]
    answers["bro"] = []  # pop from an empty list fails
    council = _council(answers)
    await council.discuss(_message(text="?"), COUNCIL)
    senders = [p[0] for p in _posted(council)]
    assert "marcus" not in senders and "bro" not in senders
    assert senders[-1] == "butler"


async def test_decision_shows_the_plan_and_wakes_nobody():
    steps = [
        CouncilStep(agent_id="light", task="приглуши торшер", wait=False),
        CouncilStep(agent_id="flirty", task="x", wait=False),
        CouncilStep(agent_id="nobody", task="x", wait=False),
    ]
    council = _council(
        {**_opinions("1"), "butler": [_done("Прошу @light приглушить.", steps)]}
    )
    plan = await council.discuss(_message(text="приглуши свет"), COUNCIL)
    kwargs = council.bus.post.await_args.kwargs
    assert kwargs["mentions"] == []
    assert kwargs["text"] == "Прошу light приглушить.\n@light"
    assert plan is not None and [s.agent_id for s in plan.steps] == ["light"]


async def test_alert_that_needs_nothing_posts_no_decision():
    alert = _alert("Письмо от банка")
    council = _council({**_opinions("1"), "butler": [_done("")]})
    assert await council.discuss(alert, COUNCIL) is None
    assert [p[1] for p in _posted(council)] == ["opinion"] * len(MEMBERS)


def _plan(*steps: CouncilStep, delivered=None) -> CouncilPlan:
    return CouncilPlan(
        consul="butler",
        decision=_message(sender="butler", text="d").model_copy(
            update={"run_id": "run-d"}
        ),
        steps=list(steps),
        delivered=delivered or [],
    )


async def test_steps_run_one_by_one_and_wait_for_the_data():
    council = _council({})
    posted: list[SwarmMessage] = []

    async def post(**kwargs):
        message = SwarmMessage(
            message_id=f"{len(posted) + 10}-0",
            created_at="2026-10-23T10:00:00+00:00",
            **{k: kwargs[k] for k in ("sender", "kind", "text", "mentions")},
            reply_to=kwargs["reply_to"],
        )
        posted.append(message)
        if kwargs["kind"] == "task" and kwargs["mentions"][0] == "weather":
            # The weather agent answers a moment later.
            asyncio.get_running_loop().call_soon(
                council.hub.publish,
                SwarmMessageEvent(
                    message=_message(sender="weather", text="12 °C").model_copy(
                        update={"reply_to": message.message_id}
                    )
                ),
            )
        return message

    council.bus.post = AsyncMock(side_effect=post)
    await council.execute(
        _plan(
            CouncilStep(agent_id="weather", task="погода", wait=True),
            CouncilStep(agent_id="voice", task="скажи погоду", wait=False),
        )
    )
    assert [m.text for m in posted] == [
        "1/2 @weather: погода",
        "2/2 @voice: скажи погоду",
    ]
    assert posted[1].mentions == ["voice", "display"]
    assert all(m.kind == "task" for m in posted)
    # ARCHIE-281: plan steps belong to the run of the decision
    assert [c.kwargs["run_id"] for c in council.bus.post.await_args_list] == [
        "run-d",
        "run-d",
    ]


async def test_step_without_reply_times_out_and_the_plan_goes_on(monkeypatch):
    monkeypatch.setattr("app.swarm.council.settings.swarm_step_timeout_seconds", 0.01)
    council = _council({})
    await council.execute(
        _plan(
            CouncilStep(agent_id="search", task="найди", wait=True),
            CouncilStep(agent_id="display", task="покажи", wait=False),
        )
    )
    kinds = [p[1] for p in _posted(council)]
    assert kinds == ["task", "system", "task"]


async def test_alert_output_is_not_delivered_twice():
    council = _council(
        {
            **_opinions("1"),
            "butler": [
                _done(
                    "Открою чат.",
                    [
                        CouncilStep(agent_id="display", task="покажи", wait=False),
                        CouncilStep(agent_id="chat", task="матч", wait=False),
                    ],
                )
            ],
        }
    )
    alert = _message(sender="cron", text="Матч через 5 минут").model_copy(
        update={"kind": "alert", "mentions": ["display"]}
    )
    plan = await council.discuss(alert, COUNCIL)
    assert plan is not None and [s.agent_id for s in plan.steps] == ["chat"]
    council.bus.post.reset_mock()
    await council.execute(plan)
    assert council.bus.post.await_args.kwargs["mentions"] == ["chat"]


def test_council_members_think_on_models_of_four_providers():
    models = {m: SWARM_AGENTS[m].default_model for m in MEMBERS}
    providers = {
        "openai" if "/" not in model else model.split("/")[0]
        for model in models.values()
    }
    assert providers == {"openai", "x-ai", "google", "anthropic"}
    assert SWARM_AGENTS["butler"].default_model == settings.swarm_consul_model
    assert SWARM_AGENTS["light"].default_model == settings.swarm_model


async def test_roster_stores_only_a_known_non_default_model():
    redis = AsyncMock()
    with patch("app.swarm.agents.get_async_redis", return_value=redis):
        roster = SwarmRoster()
    await roster.set_model("bro", "gpt-4.1")
    redis.hset.assert_awaited_once_with("archie:swarm:models", "bro", "gpt-4.1")
    await roster.set_model("bro", SWARM_AGENTS["bro"].default_model)
    redis.hdel.assert_awaited_once_with("archie:swarm:models", "bro")
    with pytest.raises(ValueError):
        await roster.set_model("bro", "no-such-model")


def test_marcus_prompt_has_germany_rules_and_persona():
    runner = SwarmAgentRunner(tool_factory=ToolFactory(), state_service=MagicMock())
    prompt = runner.build_system_prompt(SWARM_AGENTS["marcus"], "gpt-5.4-nano")
    assert "## Your job: Germany" in prompt
    assert "# Persona: Marcus" in prompt
    assert "@butler (consul)" in prompt


# --- runner ---


def _memory(facts=None, notes=None) -> MagicMock:
    memory = MagicMock()
    memory.list_facts = AsyncMock(return_value=facts or [])
    memory.get_day_notes = AsyncMock(return_value=notes or [])
    memory.get_today_facts = AsyncMock(return_value=[])
    return memory


def _runner() -> SwarmAgentRunner:
    state_service = MagicMock()
    state_service.get_user_state = AsyncMock(return_value=make_user_state())
    roster = MagicMock(
        get_owner=AsyncMock(return_value="Niko"),
        get_model=AsyncMock(side_effect=lambda profile: profile.default_model),
    )
    runner = SwarmAgentRunner(
        tool_factory=ToolFactory(),
        state_service=state_service,
        roster=roster,
        memory=_memory(),
    )
    runner.build_system_prompt = MagicMock(return_value="system")
    return runner


def _parsed(content):
    parsed = MagicMock()
    parsed.parsed_content = content
    parsed.llm_trace = MagicMock(input_tokens=10, output_tokens=5, total_cost=0.001)
    return parsed


async def test_runner_executes_allowed_and_blocks_other_calls():
    runner = _runner()
    turn = SwarmTurn(
        reasoning="r",
        tool_calls=[
            _call("smarthome_control_tool", action="turn_off", area="Кухня"),
            _call("smarthome_control_tool", action="set_temperature", temperature="25"),
        ],
        reply="",
    )
    executed = AsyncMock(
        return_value=[
            ToolResult(tool_name="smarthome_control_tool", output={"ok": True})
        ]
    )
    with (
        patch(
            "app.swarm.runner.get_client",
            return_value=MagicMock(create_completion=AsyncMock()),
        ),
        patch(
            "app.swarm.runner.parse_llm_response",
            side_effect=[
                _parsed(turn),
                _parsed(
                    SwarmTurn(
                        reasoning="r", tool_calls=[], reply="Свет на кухне выключен"
                    )
                ),
            ],
        ),
        patch("app.swarm.runner.execute_tool_calls", executed),
    ):
        draft = await runner.run(SWARM_AGENTS["light"], _message(text="@light"), [])
    assert len(executed.await_args.kwargs["tool_calls"]) == 1
    assert [a.status for a in draft.actions] == ["blocked", "success"]
    assert draft.text == "Свет на кухне выключен"
    assert draft.usage.input_tokens == 20


async def test_runner_stays_silent_on_empty_reply():
    runner = _runner()
    with (
        patch(
            "app.swarm.runner.get_client",
            return_value=MagicMock(create_completion=AsyncMock()),
        ),
        patch(
            "app.swarm.runner.parse_llm_response",
            return_value=_parsed(SwarmTurn(reasoning="r", tool_calls=[], reply="")),
        ),
    ):
        assert (
            await runner.run(SWARM_AGENTS["football"], _message(text="@football"), [])
            is None
        )


def test_system_prompt_lists_only_profile_tools():
    runner = SwarmAgentRunner(tool_factory=ToolFactory(), state_service=MagicMock())
    prompt = runner.build_system_prompt(SWARM_AGENTS["football"], "gpt-5.4-nano")
    assert "google_search_tool" in prompt
    assert "smarthome_control_tool" not in prompt
    assert "@light" in prompt
    assert "&#" not in prompt


@pytest.mark.parametrize("agent_id", list(SWARM_AGENTS))
def test_every_profile_tool_is_configured(agent_id):
    schemas = ToolFactory().get_tool_schemas(
        "gpt-5.4-nano", "plain", allowed_tools=set(SWARM_AGENTS[agent_id].tools)
    )
    assert {s["name"] for s in schemas} == set(SWARM_AGENTS[agent_id].tools)


def test_system_prompt_preloads_skill_references():
    runner = SwarmAgentRunner(tool_factory=ToolFactory(), state_service=MagicMock())
    prompt = runner.build_system_prompt(SWARM_AGENTS["light"], "gpt-5.4-nano")
    assert "## Reference: apartment_map" in prompt
    assert "Гостиная" in prompt


def test_council_prompt_has_persona_and_group_rules():
    runner = SwarmAgentRunner(tool_factory=ToolFactory(), state_service=MagicMock())
    prompt = runner.build_system_prompt(SWARM_AGENTS["bro"], "gpt-5.4-nano")
    assert "council member" in prompt
    assert "# Persona: Bro" in prompt
    assert "@council (group)" in prompt


def test_voice_prompt_has_job_rules_and_may_announce():
    runner = SwarmAgentRunner(tool_factory=ToolFactory(), state_service=MagicMock())
    prompt = runner.build_system_prompt(SWARM_AGENTS["voice"], "gpt-5.4-nano")
    assert "voice announcements" in prompt
    assert "announce_tool" in prompt
    call = _call("announce_tool", text="Матч начинается")
    assert check_tool_call(SWARM_AGENTS["voice"], call) == "allow"
    assert check_tool_call(SWARM_AGENTS["bro"], call) == "block"


def test_display_prompt_has_job_rules_and_may_only_notify():
    runner = SwarmAgentRunner(tool_factory=ToolFactory(), state_service=MagicMock())
    prompt = runner.build_system_prompt(SWARM_AGENTS["display"], "gpt-5.4-nano")
    assert "LED display" in prompt
    assert "awtrix_tool" in prompt
    display = SWARM_AGENTS["display"]
    notify = _call("awtrix_tool", action="notify", text="Bayern 20:30")
    assert check_tool_call(display, notify) == "allow"
    assert check_tool_call(display, _call("awtrix_tool", action="save_icon")) == "block"
    assert check_tool_call(display, _call("announce_tool", text="x")) == "block"
    assert check_tool_call(SWARM_AGENTS["voice"], notify) == "block"


def test_chat_line_marks_replies():
    line = format_chat_line(
        _message(sender="light", text="ok").model_copy(update={"reply_to": "1-0"})
    )
    assert line == "[1-0] @light (reply to [1-0]): ok"


def test_parse_mentions_includes_groups():
    assert parse_mentions("@council и @voice") == ["council", "voice"]


# --- ARCHIE-204 ---


def test_decision_ignores_cooldown():
    dispatcher = _dispatcher()
    dispatcher._last_run["light"] = 10**12
    decision = _message(sender="business", text="@light приглуши", depth=1)
    decision = decision.model_copy(update={"kind": "decision"})
    with patch("app.swarm.dispatcher.time.monotonic", return_value=10**12 + 1):
        assert dispatcher.select_targets(decision, ALL_AGENTS) == ["light"]


async def test_decision_for_busy_agent_is_queued_not_lost():
    dispatcher = _dispatcher()
    dispatcher._running.add("light")
    decision = _message(sender="business", text="@light приглуши", depth=1)
    decision = decision.model_copy(update={"kind": "decision"})
    dispatcher._start = MagicMock()
    await dispatcher.handle(decision)
    assert dispatcher._pending["light"] is decision
    dispatcher._start.assert_not_called()


async def test_started_target_is_not_queued_twice():
    dispatcher = _dispatcher()
    dispatcher._run_single = AsyncMock()
    await dispatcher.handle(_message(text="@light выключи"))
    assert "light" not in dispatcher._pending


async def test_runner_uses_the_swarm_owner_state():
    runner = _runner()
    with patch("app.swarm.runner.settings.swarm_user_name", None):
        await runner.get_user_state()
    runner.state_service.get_user_state.assert_awaited_with(user_name="Niko")


async def test_runner_reads_state_then_acts_in_two_rounds():
    runner = _runner()
    read = SwarmTurn(
        reasoning="read",
        tool_calls=[_call("smarthome_status_tool", query="торшер")],
        reply="",
    )
    act = SwarmTurn(
        reasoning="act",
        tool_calls=[
            _call("smarthome_control_tool", action="set_light", brightness="15")
        ],
        reply="",
    )
    done = SwarmTurn(reasoning="done", tool_calls=[], reply="Приглушил до 15%")
    executed = AsyncMock(
        side_effect=[
            [ToolResult(tool_name="smarthome_status_tool", output={"b": 45})],
            [ToolResult(tool_name="smarthome_control_tool", output={"ok": True})],
        ]
    )
    with (
        patch(
            "app.swarm.runner.get_client",
            return_value=MagicMock(create_completion=AsyncMock()),
        ),
        patch(
            "app.swarm.runner.parse_llm_response",
            side_effect=[_parsed(read), _parsed(act), _parsed(done)],
        ),
        patch("app.swarm.runner.execute_tool_calls", executed),
    ):
        draft = await runner.run(SWARM_AGENTS["light"], _message(text="@light"), [])
    assert executed.await_count == 2
    assert [a.tool_name for a in draft.actions] == ["smarthome_control_tool"]
    assert draft.text == "Приглушил до 15%"


async def test_runner_stops_tools_after_max_rounds():
    runner = _runner()
    loop = SwarmTurn(
        reasoning="again",
        tool_calls=[_call("smarthome_status_tool", query="x")],
        reply="",
    )
    with (
        patch("app.swarm.runner.settings.swarm_max_tool_rounds", 2),
        patch(
            "app.swarm.runner.get_client",
            return_value=MagicMock(create_completion=AsyncMock()),
        ),
        patch(
            "app.swarm.runner.parse_llm_response",
            side_effect=[
                _parsed(loop),
                _parsed(
                    loop.model_copy(
                        update={
                            "tool_calls": [_call("smarthome_status_tool", query="y")]
                        }
                    )
                ),
                _parsed(SwarmReply(reply="ok")),
            ],
        ),
        patch(
            "app.swarm.runner.execute_tool_calls",
            AsyncMock(
                return_value=[ToolResult(tool_name="smarthome_status_tool", output={})]
            ),
        ) as executed,
    ):
        draft = await runner.run(SWARM_AGENTS["light"], _message(text="@light"), [])
    assert executed.await_count == 2
    assert draft.text == "ok"


def test_blocked_call_explains_allowed_arguments():
    from app.swarm.policy import block_reason

    display = SWARM_AGENTS["display"]
    assert block_reason(display, _call("awtrix_tool")) == (
        "wrong arguments (action must be one of: notify)"
    )
    assert block_reason(SWARM_AGENTS["football"], _call("announce_tool")) == (
        "this tool is not yours"
    )


def test_light_prompt_reads_state_before_relative_change():
    runner = SwarmAgentRunner(
        tool_factory=ToolFactory(), state_service=MagicMock(), roster=MagicMock()
    )
    prompt = runner.build_system_prompt(SWARM_AGENTS["light"], "gpt-5.4-nano")
    assert "first read the current state" in prompt


async def test_dry_run_simulates_actions_but_reads_real_state():
    with patch("app.swarm.runner.settings.swarm_dry_run", True):
        runner = _runner()
    assert runner.action_factory.demo_mode is True
    assert runner.tool_factory.demo_mode is False
    assert runner.action_factory.tools is runner.tool_factory.tools
    executed = AsyncMock(
        side_effect=lambda tool_calls, **_: [
            ToolResult(tool_name=c.tool_name, output={}) for c in tool_calls
        ]
    )
    with patch("app.swarm.runner.execute_tool_calls", executed):
        await runner._execute(
            SWARM_AGENTS["light"],
            [
                _call("smarthome_control_tool", action="turn_off"),
                _call("smarthome_status_tool", query="x"),
            ],
            ToolContext(user_timezone="Europe/Berlin"),
        )
    # The user's timezone reaches reads and actions (ARCHIE-257)
    assert {c.kwargs["context"].user_timezone for c in executed.await_args_list} == {
        "Europe/Berlin"
    }
    factories = {
        c.kwargs["tool_calls"][0].tool_name: c.kwargs["tool_factory"]
        for c in executed.await_args_list
    }
    assert factories["smarthome_status_tool"] is runner.tool_factory
    assert factories["smarthome_control_tool"] is runner.action_factory


def test_search_agent_can_be_delegated_and_only_reads():
    profile = SWARM_AGENTS["search"]
    assert profile.role == "state" and not profile.groups
    assert all(tool in READ_ONLY_TOOLS for tool in profile.tools)
    runner = SwarmAgentRunner(
        tool_factory=ToolFactory(), state_service=MagicMock(), roster=MagicMock()
    )
    council_prompt = runner.build_system_prompt(SWARM_AGENTS["bro"], "gpt-5.4-mini")
    assert "@search" in council_prompt


# --- ARCHIE-206 ---


def _skip_notes(dispatcher) -> list[str]:
    return [
        c.kwargs["text"]
        for c in dispatcher.bus.post.await_args_list
        if c.kwargs["kind"] == "system"
    ]


async def test_depth_skip_is_reported_without_mentions():
    dispatcher = _dispatcher()
    message = _message(
        sender="voice", text="@climate дай данные", depth=settings.swarm_max_depth
    )
    await dispatcher.handle(message)
    notes = _skip_notes(dispatcher)
    assert notes == [
        f"climate skipped: chain limit of {settings.swarm_max_depth} steps reached"
    ]
    assert "@" not in notes[0]


async def test_cooldown_skip_is_reported():
    dispatcher = _dispatcher()
    dispatcher._last_run["light"] = 10**12
    with patch("app.swarm.dispatcher.time.monotonic", return_value=10**12 + 1):
        await dispatcher.handle(_message(sender="voice", text="@light статус", depth=1))
    assert _skip_notes(dispatcher) == [
        "light skipped: asked again too soon by an agent"
    ]


async def test_started_or_queued_agents_are_not_reported():
    dispatcher = _dispatcher()
    dispatcher._run_single = AsyncMock()
    await dispatcher.handle(_message(text="@light выключи"))
    dispatcher._running.add("climate")
    decision = _message(sender="bro", text="@climate теплее", depth=1)
    await dispatcher.handle(decision.model_copy(update={"kind": "decision"}))
    assert _skip_notes(dispatcher) == []


async def test_runner_stops_tool_rounds_after_successful_action():
    runner = _runner()
    speak = SwarmTurn(
        reasoning="speak",
        tool_calls=[_call("announce_tool", text="Ужин готов")],
        reply="",
    )
    with (
        patch(
            "app.swarm.runner.get_client",
            return_value=MagicMock(create_completion=AsyncMock()),
        ),
        patch(
            "app.swarm.runner.parse_llm_response",
            side_effect=[_parsed(speak), _parsed(SwarmReply(reply="🔊 Ужин готов"))],
        ),
        patch(
            "app.swarm.runner.execute_tool_calls",
            AsyncMock(return_value=[ToolResult(tool_name="announce_tool", output={})]),
        ) as executed,
    ):
        draft = await runner.run(SWARM_AGENTS["voice"], _message(text="@voice"), [])
    assert executed.await_count == 1
    assert draft.text == "🔊 Ужин готов"


def test_voice_can_read_home_state():
    assert "smarthome_status_tool" in SWARM_AGENTS["voice"].tools


async def test_silent_agent_on_user_task_gets_a_note():
    dispatcher = _dispatcher()
    dispatcher.bus.get_history = AsyncMock(return_value=[])
    dispatcher.runner.run = AsyncMock(return_value=None)
    await dispatcher._run_agent("voice", _message(text="@voice озвучь"))
    assert _skip_notes(dispatcher) == ["voice gave no answer"]


async def test_silent_listener_gets_no_note():
    dispatcher = _dispatcher()
    dispatcher.bus.get_history = AsyncMock(return_value=[])
    dispatcher.runner.run = AsyncMock(return_value=None)
    await dispatcher._run_agent("voice", _message(sender="light", text="готово"))
    assert _skip_notes(dispatcher) == []


def test_fallback_reply_shows_action_arguments():
    action = SwarmAgentRunner._action_info(
        _call("announce_tool", text="Ужин готов"), "success", None
    )
    assert (
        SwarmAgentRunner._describe_action(action)
        == "announce_tool (success): Ужин готов"
    )


def test_context_after_midnight_names_yesterday(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    runner = _runner()
    night = datetime(2026, 10, 23, 0, 40, tzinfo=ZoneInfo("Europe/Berlin"))
    monkeypatch.setattr("app.swarm.runner.local_now", lambda _: night)
    context = runner.build_context(make_user_state(), [], None, [])
    assert "usually means 22.10.2026" in context
    assert '"tomorrow" means 23.10.2026' in context
    noon = night.replace(hour=12)
    monkeypatch.setattr("app.swarm.runner.local_now", lambda _: noon)
    assert "after midnight" not in runner.build_context(make_user_state(), [], None, [])


async def test_mention_of_sleeping_agent_is_reported():
    dispatcher = _dispatcher()
    dispatcher.roster.get_enabled_ids = AsyncMock(return_value=ALL_AGENTS - {"voice"})
    dispatcher._start = MagicMock()
    await dispatcher.handle(_message(text="@voice произнеси тест"))
    dispatcher._start.assert_not_called()
    assert _skip_notes(dispatcher) == [
        "voice skipped: asleep, wake it up in the agent list"
    ]


def _alert(text: str = "Торшер горит, в спальне никого") -> SwarmMessage:
    return SwarmMessage(
        message_id="7-0",
        created_at="2026-10-23T11:25:00+00:00",
        sender="home",
        kind="alert",
        text=text,
        mentions=["display"],
        importance="low",
    )


def test_alerts_go_to_the_council():
    assert SwarmDispatcher.wants_council(_alert())


async def test_alert_does_not_push_out_a_waiting_user_message():
    dispatcher = _dispatcher()
    dispatcher._running.update(COUNCIL)
    user = _message(text="как дела?")
    await dispatcher._start_council(user, ALL_AGENTS)
    await dispatcher._start_council(_alert(), ALL_AGENTS)
    assert dispatcher._pending["council"] is user
    assert dispatcher._pending["council:alert"].kind == "alert"


# --- alert chats and paired outputs (ARCHIE-247) ---


def test_voice_mention_also_wakes_the_display():
    from app.swarm.agents import add_paired_outputs

    assert add_paired_outputs(["voice"]) == ["voice", "display"]
    assert add_paired_outputs(["light", "voice", "chat"]) == [
        "light",
        "voice",
        "display",
        "chat",
    ]
    assert add_paired_outputs(["display", "voice"]) == ["display", "voice"]
    assert add_paired_outputs(["display"]) == ["display"]
    assert add_paired_outputs(["chat"]) == ["chat", "display"]


async def test_bus_adds_display_to_a_voice_mention():
    redis = AsyncMock()
    redis.xadd = AsyncMock(return_value="6-0")
    with patch("app.swarm.bus.get_async_redis", return_value=redis):
        bus = SwarmBus(SwarmHub())
    posted = await bus.post(sender="user", kind="user", text="@voice скажи тест")
    assert posted.mentions == ["voice", "display"]


def test_chat_agent_may_open_a_chat_but_not_control_devices():
    chat = SWARM_AGENTS["chat"]
    assert (
        check_tool_call(chat, _call("alert_chat_tool", title="t", message="m"))
        == "allow"
    )
    assert check_tool_call(chat, _call("smarthome_control_tool")) == "block"


async def test_alert_chat_tool_creates_the_chat_in_django(monkeypatch):
    from app.tools import alert_chat_tool as module

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = MagicMock(raise_for_status=MagicMock())
    monkeypatch.setattr(module.httpx, "AsyncClient", MagicMock(return_value=client))
    monkeypatch.setattr(settings, "swarm_user_name", "niko")
    monkeypatch.setattr(settings, "web_django_url", "http://web:8000/")
    result = await module.alert_chat_tool(
        title="Торшер  горит", message="Днём горит торшер в гостиной. Что сделать?"
    )
    conversation_id = result["chat"].split("/")[2]
    assert result == {"chat": f"/c/{conversation_id}/", "title": "🔔 Торшер горит"}
    post = client.post.await_args
    assert post.args[0] == "http://web:8000/ai-assistant/api/messages/"
    assert post.kwargs["json"] == {
        "conversation_id": conversation_id,
        "input": "Днём горит торшер в гостиной. Что сделать?",
        "user_name": "niko",
        "response_format": "ui_answer",
        "demo_mode": False,
    }
    patch_call = client.patch.await_args
    assert patch_call.args[0].endswith(f"/conversations/{conversation_id}/")
    assert patch_call.kwargs["json"] == {"title": "🔔 Торшер горит"}


def test_main_chat_does_not_get_the_alert_chat_tool():
    names = [t["name"] for t in ToolFactory().get_tool_schemas("gpt-5.6-luna", "plain")]
    assert "alert_chat_tool" not in names


# --- timed swarm alerts (ARCHIE-248) ---


def _cron_call(**arguments: str) -> SwarmToolCall:
    return SwarmToolCall(
        tool_name="cron_tool",
        arguments=[Parameter(name=k, value=v) for k, v in arguments.items()],
    )


def test_cron_agent_plans_only_swarm_alerts():
    cron = SWARM_AGENTS["cron"]
    plan = _cron_call(
        action="create",
        tool_name="swarm_alert_tool",
        arguments_json='{"text": "Матч через 5 минут"}',
        run_at="2026-10-24T18:25:00",
    )
    assert check_tool_call(cron, plan) == "allow"
    lamp = _cron_call(action="create", tool_name="smarthome_control_tool")
    assert check_tool_call(cron, lamp) == "block"
    assert check_tool_call(cron, _cron_call(action="list")) == "allow"
    assert check_tool_call(cron, _cron_call(action="delete", job_id="j")) == "allow"
    assert check_tool_call(cron, _cron_call(action="run_now", job_id="j")) == "block"


async def test_swarm_alert_tool_posts_an_alert_of_cron(monkeypatch):
    from app.tools import swarm_alert_tool as module

    service = MagicMock()
    service.memory.get_mode = AsyncMock(return_value="paused")
    service.ticker.post_alert = AsyncMock(return_value=_message(sender="cron"))
    monkeypatch.setattr(module, "swarm_service", service)
    result = await module.swarm_alert_tool("Матч через 5 минут", importance="high")
    assert result == {"status": "success", "message_id": "1-0"}
    service.ticker.post_alert.assert_awaited_once_with(
        "cron", "Матч через 5 минут", "high"
    )
    service.memory.get_mode = AsyncMock(return_value="stopped")
    assert (await module.swarm_alert_tool("x"))["status"] == "skipped"


async def test_post_alert_picks_outputs_by_importance(monkeypatch):
    from app.swarm.ticker import SwarmTicker

    ticker = SwarmTicker.__new__(SwarmTicker)
    ticker.memory = MagicMock(get_settings=AsyncMock(return_value=SwarmSettings()))
    ticker.runner = MagicMock(get_user_state=AsyncMock(return_value=make_user_state()))
    ticker.roster = MagicMock(get_enabled_ids=AsyncMock(return_value=ALL_AGENTS))
    ticker.bus = MagicMock(post=AsyncMock())
    monkeypatch.setattr("app.swarm.ticker.voice_allowed", lambda now, s: True)
    await ticker.post_alert("cron", "Матч", "high")
    kwargs = ticker.bus.post.await_args.kwargs
    assert kwargs["kind"] == "alert"
    assert kwargs["mentions"] == ["display", "voice"]
    assert kwargs["importance"] == "high"
