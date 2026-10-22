"""Tests for the agent swarm chat (ARCHIE-199)."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from archie_shared.swarm.models import SwarmMessage, SwarmUsage
from app.config import settings
from app.models.orchestration_sgr import Parameter
from app.models.tool_models import ToolResult
from app.swarm.agents import SWARM_AGENTS
from app.swarm.bus import SwarmBus, SwarmHub, clip_text, parse_mentions
from app.swarm.council import SwarmCouncil
from app.swarm.dispatcher import SwarmDispatcher
from app.swarm.models import (
    CouncilDecision,
    CouncilDelegation,
    CouncilTriage,
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
    assert len(clip_text("a " * 400)) == 280


# --- dispatcher guards ---


ALL_AGENTS = set(SWARM_AGENTS)
COUNCIL = ["bro", "business", "flirty"]


def _dispatcher() -> SwarmDispatcher:
    roster = MagicMock()
    roster.get_enabled_ids = AsyncMock(return_value=ALL_AGENTS)
    return SwarmDispatcher(
        hub=SwarmHub(),
        bus=MagicMock(post=AsyncMock()),
        roster=roster,
        runner=MagicMock(),
        council=MagicMock(run=AsyncMock()),
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


async def test_council_runs_with_enabled_members_and_force_on_mention():
    dispatcher = _dispatcher()
    await dispatcher._start_council(_message(text="@council?"), ALL_AGENTS - {"bro"})
    await asyncio.gather(*dispatcher._tasks)
    args = dispatcher.council.run.await_args.args
    assert args[1] == ["business", "flirty"]
    assert args[2] is True
    assert not dispatcher._running


async def test_busy_council_keeps_latest_message_and_answers_it_after():
    dispatcher = _dispatcher()
    dispatcher._running.add("bro")
    latest = _message(text="второй вопрос")
    await dispatcher._start_council(_message(text="первый"), ALL_AGENTS)
    await dispatcher._start_council(latest, ALL_AGENTS)
    assert dispatcher._pending["council"] is latest
    dispatcher.council.run.assert_not_awaited()
    dispatcher._running.clear()
    await dispatcher._run_council(_message(text="первый"), COUNCIL, False)
    await asyncio.gather(*dispatcher._tasks)
    assert dispatcher.council.run.await_args_list[-1].args[0] is latest


async def test_sleeping_council_posts_system_message_without_mentions():
    dispatcher = _dispatcher()
    await dispatcher._start_council(_message(text="привет"), {"light"})
    kwargs = dispatcher.bus.post.await_args.kwargs
    assert kwargs["kind"] == "system"
    assert "@" not in kwargs["text"]


async def test_council_failure_posts_system_message_and_frees_members():
    dispatcher = _dispatcher()
    dispatcher.council.run = AsyncMock(side_effect=RuntimeError("boom"))
    dispatcher._running.update(COUNCIL)
    await dispatcher._run_council(_message(text="x"), COUNCIL, False)
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


# --- council ---


def _council(persona: str, answers: list) -> SwarmCouncil:
    runner = MagicMock()
    runner.get_user_state = AsyncMock(return_value=make_user_state(persona=persona))
    runner.render_task = MagicMock(side_effect=lambda template, **_: template)
    usage = SwarmUsage(model="m", input_tokens=10, output_tokens=2, total_cost=0.001)
    runner.ask = AsyncMock(side_effect=[(answer, usage) for answer in answers])
    bus = MagicMock(post=AsyncMock(), get_history=AsyncMock(return_value=[]))
    return SwarmCouncil(hub=SwarmHub(), bus=bus, runner=runner)


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


async def test_chair_is_the_user_persona():
    council = _council("bro", [])
    assert await council.choose_chair(COUNCIL) == "bro"
    council = _council("butler", [])
    assert await council.choose_chair(COUNCIL) == "business"
    assert await council.choose_chair(["flirty"]) == "flirty"


async def test_simple_request_gets_decision_in_one_call():
    triage = CouncilTriage(
        reasoning="r",
        convene=False,
        reply="Выключаю.",
        delegations=[CouncilDelegation(agent_id="light", task="выключи свет")],
        remember=[],
    )
    council = _council("business", [triage])
    await council.run(_message(text="выключи свет"), COUNCIL, force=False)
    assert _asked(council) == [("business", "council_triage.jinja2")]
    assert _posted(council) == [
        ("business", "decision", "Выключаю.\n→ @light: выключи свет")
    ]


async def test_convened_council_posts_opinions_then_one_decision():
    council = _council(
        "flirty",
        [
            CouncilTriage(
                reasoning="r", convene=True, reply="", delegations=[], remember=[]
            ),
            SwarmReply(reply="Ехать"),
            SwarmReply(reply="Остаться"),
            CouncilDecision(
                reply="Решили: едем в субботу.", delegations=[], remember=[]
            ),
        ],
    )
    await council.run(_message(text="ехать или нет?"), COUNCIL, force=False)
    assert _asked(council) == [
        ("flirty", "council_triage.jinja2"),
        ("bro", "council_opinion.jinja2"),
        ("business", "council_opinion.jinja2"),
        ("flirty", "council_decision.jinja2"),
    ]
    assert _posted(council) == [
        ("bro", "opinion", "Ехать"),
        ("business", "opinion", "Остаться"),
        ("flirty", "decision", "Решили: едем в субботу."),
    ]
    decision_usage = (
        cast(AsyncMock, council.bus.post).await_args_list[-1].kwargs["usage"]
    )
    assert decision_usage.input_tokens == 20


async def test_force_convenes_even_if_chair_has_an_answer():
    council = _council(
        "business",
        [
            CouncilTriage(
                reasoning="r",
                convene=False,
                reply="Сразу ответ",
                delegations=[],
                remember=[],
            ),
            SwarmReply(reply="a"),
            SwarmReply(reply="b"),
            CouncilDecision(reply="Итог", delegations=[], remember=[]),
        ],
    )
    await council.run(_message(text="@council?"), COUNCIL, force=True)
    assert [kind for _, kind, _ in _posted(council)] == [
        "opinion",
        "opinion",
        "decision",
    ]


# --- runner ---


def _memory(facts=None, notes=None) -> MagicMock:
    memory = MagicMock()
    memory.list_facts = AsyncMock(return_value=facts or [])
    memory.get_day_notes = AsyncMock(return_value=notes or [])
    return memory


def _runner() -> SwarmAgentRunner:
    state_service = MagicMock()
    state_service.get_user_state = AsyncMock(return_value=make_user_state())
    roster = MagicMock(get_owner=AsyncMock(return_value="Niko"))
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


async def test_agent_mention_of_council_gets_triage_not_forced_discussion():
    dispatcher = _dispatcher()
    message = _message(sender="light", text="@council уточните время", depth=1)
    assert SwarmDispatcher.wants_council(message)
    await dispatcher._start_council(message, ALL_AGENTS)
    await asyncio.gather(*dispatcher._tasks)
    assert dispatcher.council.run.await_args.args[2] is False


def test_council_members_use_the_council_model():
    assert {SWARM_AGENTS[m].model for m in COUNCIL} == {settings.swarm_council_model}
    assert SWARM_AGENTS["light"].model is None


# --- ARCHIE-204 ---


async def test_offer_with_at_sign_does_not_delegate():
    triage = CouncilTriage(
        reasoning="r",
        convene=False,
        reply="Могу попросить @light включить торшер.",
        delegations=[],
        remember=[],
    )
    council = _council("bro", [triage])
    await council.run(_message(text="какой свет включить?"), COUNCIL, force=False)
    kwargs = council.bus.post.await_args.kwargs
    assert kwargs["mentions"] == []
    assert "@" not in kwargs["text"]


async def test_delegations_become_mentions_and_skip_unknown_or_council_agents():
    triage = CouncilTriage(
        reasoning="r",
        convene=False,
        reply="Делаю.",
        delegations=[
            CouncilDelegation(agent_id="light", task="приглуши торшер"),
            CouncilDelegation(agent_id="flirty", task="x"),
            CouncilDelegation(agent_id="nobody", task="x"),
        ],
        remember=[],
    )
    council = _council("bro", [triage])
    await council.run(_message(text="приглуши свет"), COUNCIL, force=False)
    kwargs = council.bus.post.await_args.kwargs
    assert kwargs["mentions"] == ["light"]
    assert kwargs["text"] == "Делаю.\n→ @light: приглуши торшер"
    assert kwargs["sender"] == "bro"


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


async def test_runner_does_not_repeat_the_same_call():
    runner = _runner()
    loop = SwarmTurn(
        reasoning="again",
        tool_calls=[_call("smarthome_status_tool", query="x")],
        reply="",
    )
    with (
        patch(
            "app.swarm.runner.get_client",
            return_value=MagicMock(create_completion=AsyncMock()),
        ),
        patch(
            "app.swarm.runner.parse_llm_response",
            side_effect=[_parsed(loop), _parsed(loop), _parsed(SwarmReply(reply="ok"))],
        ),
        patch(
            "app.swarm.runner.execute_tool_calls",
            AsyncMock(
                return_value=[ToolResult(tool_name="smarthome_status_tool", output={})]
            ),
        ) as executed,
    ):
        draft = await runner.run(SWARM_AGENTS["light"], _message(text="@light"), [])
    assert executed.await_count == 1
    assert draft.text == "ok"


async def test_runner_drops_arguments_without_name():
    runner = _runner()
    turn = SwarmTurn(
        reasoning="r",
        tool_calls=[
            SwarmToolCall(
                tool_name="smarthome_status_tool",
                arguments=[
                    Parameter(name="query", value="x"),
                    Parameter(name="", value=":"),
                ],
            )
        ],
        reply="",
    )
    executed = AsyncMock(
        return_value=[ToolResult(tool_name="smarthome_status_tool", output={})]
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
                _parsed(SwarmTurn(reasoning="r", tool_calls=[], reply="ok")),
            ],
        ),
        patch("app.swarm.runner.execute_tool_calls", executed),
    ):
        await runner.run(SWARM_AGENTS["light"], _message(text="@light"), [])
    sent = executed.await_args.kwargs["tool_calls"][0].arguments
    assert [p.name for p in sent] == ["query"]


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
        )
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


async def test_failed_call_may_be_repeated_and_watch_reports_no_data():
    runner = _runner()
    broken = SwarmWatchTurn(
        reasoning="r",
        tool_calls=[_call("smarthome_status_tool")],
        reply="",
        importance="low",
        today_note="",
        quiet_until="",
    )
    final = SwarmWatchReply(reply="OK", importance="low", today_note="", quiet_until="")
    executed = AsyncMock(
        return_value=[
            ToolResult(
                tool_name="smarthome_status_tool", output={}, success=False, error="x"
            )
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
                _parsed(broken),
                _parsed(broken),
                _parsed(broken),
                _parsed(final),
            ],
        ),
        patch("app.swarm.runner.execute_tool_calls", executed),
    ):
        draft = await runner.watch(SWARM_AGENTS["home"], "home", [], "task")
    assert executed.await_count == 3
    assert draft.got_data is False


async def test_empty_turn_on_a_task_gets_one_nudge():
    runner = _runner()
    empty = SwarmTurn(reasoning="r", tool_calls=[], reply="")
    answer = SwarmTurn(reasoning="r", tool_calls=[], reply="Не нашёл данных")
    with (
        patch(
            "app.swarm.runner.get_client",
            return_value=MagicMock(create_completion=AsyncMock()),
        ),
        patch(
            "app.swarm.runner.parse_llm_response",
            side_effect=[_parsed(empty), _parsed(answer)],
        ),
    ):
        draft = await runner.run(
            SWARM_AGENTS["football"], _message(text="@football счёт?"), []
        )
    assert draft.text == "Не нашёл данных"


async def test_empty_turn_on_agent_news_gets_no_nudge():
    runner = _runner()
    with (
        patch(
            "app.swarm.runner.get_client",
            return_value=MagicMock(create_completion=AsyncMock()),
        ),
        patch(
            "app.swarm.runner.parse_llm_response",
            return_value=_parsed(SwarmTurn(reasoning="r", tool_calls=[], reply="")),
        ) as parsed,
    ):
        draft = await runner.run(
            SWARM_AGENTS["voice"], _message(sender="light", text="свет включён"), []
        )
    assert draft is None
    assert parsed.call_count == 1


async def test_broken_json_is_retried_once():
    runner = _runner()
    with (
        patch(
            "app.swarm.runner.get_client",
            return_value=MagicMock(create_completion=AsyncMock()),
        ),
        patch(
            "app.swarm.runner.parse_llm_response",
            side_effect=[
                ValueError("Expecting property name"),
                _parsed(SwarmTurn(reasoning="r", tool_calls=[], reply="ok")),
            ],
        ),
    ):
        draft = await runner.run(
            SWARM_AGENTS["football"], _message(text="@football"), []
        )
    assert draft.text == "ok"


def test_context_after_midnight_names_yesterday(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    runner = _runner()
    night = datetime(2026, 10, 23, 0, 40, tzinfo=ZoneInfo("Europe/Berlin"))
    monkeypatch.setattr("app.swarm.runner.local_now", lambda _: night)
    context = runner.build_context(make_user_state(), [], None, [])
    assert "usually means 22.10.2026" in context
    noon = night.replace(hour=12)
    monkeypatch.setattr("app.swarm.runner.local_now", lambda _: noon)
    assert "after midnight" not in runner.build_context(make_user_state(), [], None, [])
