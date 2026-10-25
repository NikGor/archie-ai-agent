"""Tests for the spontaneous swarm checks, controls and memory (ARCHIE-238)."""

from datetime import datetime, timedelta
from itertools import pairwise
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo
import pytest
from archie_shared.swarm.models import SwarmMessage, SwarmSettings, SwarmUsage
from app.models.orchestration_sgr import Parameter
from app.agents.swarm.agents import SWARM_AGENTS
from app.agents.swarm.bus import SwarmHub
from app.agents.swarm.council import SwarmCouncil
from app.agents.swarm.dispatcher import SwarmDispatcher
from app.agents.swarm.memory import DayNote
from app.agents.swarm.models import ConsulDecision, SwarmToolCall, SwarmWatchReply
from app.agents.swarm.policy import check_tool_call
from app.agents.swarm.messages import format_chat_line
from app.agents.swarm.runner import SwarmAgentRunner, SwarmWatchDraft
from app.agents.swarm.schedule import (
    due_tasks,
    in_window,
    is_active,
    parse_quiet_until,
    pick_delay_seconds,
    pick_follow_up,
    pick_task,
    plan_wake,
    voice_allowed,
)
from app.agents.swarm.ticker import SwarmTicker, select_outputs
import app.agents.swarm.tools  # noqa: F401
from app.agents.chat.tool_access import chat_tool_names
from app.tools.catalog import tool_registry
from tests.conftest import make_user_state


BERLIN = ZoneInfo("Europe/Berlin")
ALL_AGENTS = set(SWARM_AGENTS)


def _at(day: int, hour: int, minute: int = 0) -> datetime:
    """22.10.2026 is a Thursday, 24.10.2026 a Saturday."""
    return datetime(2026, 10, day, hour, minute, tzinfo=BERLIN)


def _call(tool_name: str, **arguments: str) -> SwarmToolCall:
    return SwarmToolCall(
        tool_name=tool_name,
        arguments=[Parameter(name=k, value=v) for k, v in arguments.items()],
    )


# --- schedule ---


def test_window_until_midnight_and_over_midnight():
    assert in_window(_at(22, 8), "08:00", "00:00")
    assert in_window(_at(22, 23, 59), "08:00", "00:00")
    assert not in_window(_at(22, 7, 59), "08:00", "00:00")
    assert in_window(_at(22, 1), "22:00", "02:00")
    assert not in_window(_at(22, 12), "22:00", "02:00")


def test_active_hours_default_8_to_midnight():
    swarm_settings = SwarmSettings()
    assert is_active(_at(22, 8), swarm_settings)
    assert not is_active(_at(22, 0, 30), swarm_settings)


def test_voice_hours_differ_on_weekend():
    swarm_settings = SwarmSettings()
    assert not voice_allowed(_at(22, 8, 30), swarm_settings)
    assert voice_allowed(_at(22, 9), swarm_settings)
    assert not voice_allowed(_at(22, 23, 30), swarm_settings)
    assert not voice_allowed(_at(24, 10, 30), swarm_settings)
    assert voice_allowed(_at(24, 11), swarm_settings)


def test_delay_stays_in_the_interval():
    swarm_settings = SwarmSettings(interval_min_minutes=1, interval_max_minutes=2)
    delays = [pick_delay_seconds(swarm_settings) for _ in range(50)]
    assert all(60 <= delay <= 120 for delay in delays)


def test_settings_reject_bad_interval():
    with pytest.raises(ValueError):
        SwarmSettings(interval_min_minutes=30, interval_max_minutes=10)
    with pytest.raises(ValueError):
        SwarmSettings(interval_max_minutes=61)


def test_due_tasks_skip_quiet_tasks():
    now = _at(22, 12)
    swarm_settings = SwarmSettings()
    assert due_tasks(swarm_settings, now, {}) == [
        "home",
        "mail",
        "calendar",
        "football",
        "weather",
    ]
    quiet = {"football": _at(22, 20, 25), "mail": _at(22, 11)}
    assert due_tasks(swarm_settings, now, quiet) == [
        "home",
        "mail",
        "calendar",
        "weather",
    ]
    assert "football" in due_tasks(swarm_settings, _at(22, 20, 30), quiet)


def test_football_check_uses_skill_teams_by_default():
    runner = SwarmAgentRunner(tool_registry=tool_registry, state_service=MagicMock())
    text = runner.prompts.render_task("swarm_watch", task="football", teams=[], language="ru")
    assert "from your skill" in text
    text = runner.prompts.render_task(
        "swarm_watch", task="football", teams=["Bayern"], language="ru"
    )
    assert "Teams: Bayern." in text


def test_pick_task_prefers_never_run_then_oldest():
    assert pick_task(["home", "mail"], {"home": 5.0}) == "mail"
    assert pick_task(["home", "mail"], {"home": 5.0, "mail": 3.0}) == "mail"
    assert pick_task([], {}) is None


def test_quiet_until_only_later_today():
    now = _at(22, 12)
    assert parse_quiet_until("20:25", now) == _at(22, 20, 25)
    assert parse_quiet_until("11:00", now) is None
    assert parse_quiet_until("", now) is None
    assert parse_quiet_until("soon", now) is None


def test_outputs_follow_importance_and_voice_hours():
    assert select_outputs("low", True, ALL_AGENTS) == []
    assert select_outputs("normal", True, ALL_AGENTS) == ["display"]
    assert select_outputs("high", True, ALL_AGENTS) == ["display", "voice"]
    assert select_outputs("high", False, ALL_AGENTS) == ["display"]
    assert select_outputs("high", True, ALL_AGENTS - {"display"}) == ["voice"]


# --- policy and agents ---


def test_calendar_agent_only_reads():
    calendar = SWARM_AGENTS["calendar"]
    assert check_tool_call(calendar, _call("events_tool", action="upcoming")) == "allow"
    assert check_tool_call(calendar, _call("events_tool", action="create")) == "block"
    assert check_tool_call(calendar, _call("events_tool", action="delete")) == "block"


def test_mail_agent_reads_mail():
    mail = SWARM_AGENTS["mail"]
    assert check_tool_call(mail, _call("gmail_tool", action="search")) == "allow"
    assert check_tool_call(mail, _call("announce_tool", text="x")) == "block"


@pytest.mark.parametrize("task", ["home", "mail", "calendar", "football"])
def test_watch_prompt_has_task_section(task):
    runner = SwarmAgentRunner(tool_registry=tool_registry, state_service=MagicMock())
    text = runner.prompts.render_task("swarm_watch", task=task, teams=["Bayern"], language="ru")
    assert "in this language: ru" in text
    assert "## What to check" in text
    assert "today_note" in text


def test_alert_line_shows_hidden_mentions_and_importance():
    message = SwarmMessage(
        message_id="5-0",
        created_at="2026-10-22T10:00:00+00:00",
        sender="mail",
        kind="alert",
        text="Письмо из банка",
        mentions=["display", "voice"],
        importance="high",
    )
    assert format_chat_line(message) == (
        "[5-0] @mail (to @display, @voice) [high alert]: Письмо из банка"
    )


# --- memory in prompts ---


def _memory(facts=None, notes=None) -> MagicMock:
    memory = MagicMock()
    memory.list_facts = AsyncMock(return_value=facts or [])
    memory.get_day_notes = AsyncMock(return_value=notes or [])
    memory.get_today_facts = AsyncMock(return_value=[])
    return memory


NOTES = [
    DayNote(time="09:10", task="mail", note="ids a1,a2"),
    DayNote(time="09:20", task="home", note="kitchen window open, 21°"),
    DayNote(time="09:40", task="mail", note="ids a3", reported="Письмо от Анны"),
]


async def test_watcher_sees_own_notes_and_others_see_alerts():
    fact = MagicMock(text="Не сообщать про влажность")
    runner = SwarmAgentRunner(
        tool_registry=tool_registry,
        state_service=MagicMock(),
        memory=_memory([fact], NOTES),
    )
    state = make_user_state()
    own = await runner.prompts.build_memory_lines(state, "mail")
    assert own[:2] == ["Long-term memory:", "- Не сообщать про влажность"]
    assert "- 09:10 mail: ids a1,a2" in own
    assert all("kitchen" not in line for line in own)
    others = await runner.prompts.build_memory_lines(state, None)
    assert others[-2:] == ["Alerts today:", "- 09:40 mail, reported: Письмо от Анны"]


# --- ticker ---


def _ticker(outcome: SwarmWatchReply, mode: str = "running") -> SwarmTicker:
    memory = MagicMock()
    for name in [
        "set_quiet_until",
        "set_follow_up",
        "add_day_note",
        "mark_task_run",
        "set_last_run",
        "set_mode",
        "set_settings",
        "add_today_facts",
    ]:
        setattr(memory, name, AsyncMock())
    memory.get_mode = AsyncMock(return_value=mode)
    memory.get_settings = AsyncMock(
        return_value=SwarmSettings(football_teams=["Bayern"])
    )
    memory.get_quiet_until = AsyncMock(return_value={})
    memory.get_follow_ups = AsyncMock(return_value={})
    memory.get_last_task_runs = AsyncMock(return_value={})
    memory.get_last_run = AsyncMock(return_value=None)
    memory.list_facts = AsyncMock(return_value=[])
    memory.get_today_facts = AsyncMock(return_value=[])
    runner = MagicMock()
    runner.get_user_state = AsyncMock(
        return_value=make_user_state(user_timezone="Europe/Berlin")
    )
    runner.prompts.render_task = MagicMock(return_value="task")
    runner.watch = AsyncMock(
        return_value=SwarmWatchDraft(outcome=outcome, usage=SwarmUsage(model="m"))
    )
    roster = MagicMock()
    roster.get_enabled_ids = AsyncMock(return_value=ALL_AGENTS)
    roster.list_agents = AsyncMock(return_value=[])
    bus = MagicMock(post=AsyncMock(), get_history=AsyncMock(return_value=[]))
    bus.clear = AsyncMock()
    return SwarmTicker(
        hub=SwarmHub(),
        bus=bus,
        roster=roster,
        runner=runner,
        memory=memory,
        dispatcher=MagicMock(reset=AsyncMock()),
    )


def _outcome(
    reply: str, importance: str = "normal", quiet: str = "", follow_up: bool = False
) -> SwarmWatchReply:
    return SwarmWatchReply(
        reply=reply,
        importance=importance,
        today_note="seen",
        follow_up=follow_up,
        quiet_until=quiet,
        remember_today=[],
    )


async def test_silent_check_posts_a_check_line_with_the_note():
    ticker = _ticker(_outcome(""))
    await ticker._watch("mail", SwarmSettings(), _at(22, 12), "ru")
    kwargs = ticker.bus.post.await_args.kwargs
    assert (kwargs["sender"], kwargs["kind"], kwargs["text"], kwargs["mentions"]) == (
        "mail",
        "check",
        "seen",
        [],
    )
    note = ticker.memory.add_day_note.await_args.args[1]
    assert (note.task, note.note, note.reported) == ("mail", "seen", None)
    assert ticker.memory.set_last_run.await_args.args[0].outcome == "silent"


async def test_high_alert_goes_to_display_and_voice_in_voice_hours():
    ticker = _ticker(_outcome("Встреча через 10 минут", "high"))
    await ticker._watch("calendar", SwarmSettings(), _at(22, 12), "ru")
    kwargs = ticker.bus.post.await_args.kwargs
    assert kwargs["sender"] == "calendar"
    assert kwargs["kind"] == "alert"
    assert kwargs["mentions"] == ["display", "voice"]
    assert kwargs["importance"] == "high"
    assert ticker.memory.add_day_note.await_args.args[1].reported == (
        "Встреча через 10 минут"
    )


async def test_high_alert_is_not_spoken_at_night():
    ticker = _ticker(_outcome("Окно открыто, 17°", "high"))
    await ticker._watch("home", SwarmSettings(), _at(22, 23, 45), "ru")
    assert ticker.bus.post.await_args.kwargs["mentions"] == ["display"]


async def test_quiet_until_is_saved_for_the_task():
    ticker = _ticker(_outcome("", quiet="20:25"))
    await ticker._watch("football", SwarmSettings(), _at(22, 12), "ru")
    ticker.memory.set_quiet_until.assert_awaited_once_with("football", _at(22, 20, 25))


async def test_check_outside_active_hours_does_nothing(monkeypatch):
    ticker = _ticker(_outcome("x"))
    monkeypatch.setattr("app.agents.swarm.ticker.local_now", lambda _: _at(22, 7))
    await ticker.run_check()
    ticker.runner.watch.assert_not_awaited()


async def test_check_picks_a_due_task(monkeypatch):
    ticker = _ticker(_outcome(""))
    ticker.memory.get_last_task_runs = AsyncMock(
        return_value={
            "home": 1.0,
            "mail": 2.0,
            "calendar": 3.0,
            "football": 4.0,
            "weather": 5.0,
        }
    )
    monkeypatch.setattr("app.agents.swarm.ticker.local_now", lambda _: _at(22, 12))
    await ticker.run_check()
    assert ticker.runner.watch.await_args.args[1] == "home"


async def test_failed_check_is_recorded():
    ticker = _ticker(_outcome(""))
    ticker.runner.watch = AsyncMock(side_effect=RuntimeError("boom"))
    await ticker.run_check("mail")
    assert ticker.memory.set_last_run.await_args.args[0].outcome == "failed"
    kwargs = ticker.bus.post.await_args.kwargs
    assert (kwargs["kind"], kwargs["text"]) == ("system", "mail check failed: boom")


async def test_stop_cancels_turns_and_clears_chat():
    ticker = _ticker(_outcome(""))
    queue = ticker.hub.subscribe()
    await ticker.set_mode("stopped")
    ticker.memory.set_mode.assert_awaited_once_with("stopped")
    ticker.dispatcher.reset.assert_awaited_once()
    ticker.bus.clear.assert_awaited_once()
    events = [queue.get_nowait().type for _ in range(queue.qsize())]
    assert events == ["history", "control"]


async def test_pause_keeps_chat():
    ticker = _ticker(_outcome(""))
    await ticker.set_mode("paused")
    ticker.bus.clear.assert_not_awaited()
    ticker.dispatcher.reset.assert_not_awaited()


async def test_paused_ticker_plans_no_check():
    ticker = _ticker(_outcome(""), mode="paused")
    await ticker.start()
    await __import__("asyncio").sleep(0)
    assert ticker.next_run_at is None
    await ticker.stop()


async def test_running_ticker_plans_next_check(monkeypatch):
    ticker = _ticker(_outcome(""))
    monkeypatch.setattr("app.agents.swarm.ticker.local_now", lambda _: _at(22, 12))
    before = datetime.now(BERLIN)
    await ticker.start()
    await __import__("asyncio").sleep(0)
    assert before + timedelta(minutes=3) <= ticker.next_run_at
    assert ticker.next_run_at <= before + timedelta(minutes=16, seconds=1)
    await ticker.stop()


# --- dispatcher and council ---


async def test_stopped_dispatcher_ignores_messages():
    roster = MagicMock(get_enabled_ids=AsyncMock(return_value=ALL_AGENTS))
    dispatcher = SwarmDispatcher(
        hub=SwarmHub(),
        bus=MagicMock(post=AsyncMock()),
        roster=roster,
        runner=MagicMock(),
        council=MagicMock(run=AsyncMock()),
        memory=MagicMock(get_mode=AsyncMock(return_value="stopped")),
    )
    dispatcher._start = MagicMock()
    message = SwarmMessage(
        message_id="1-0",
        created_at="2026-10-22T10:00:00+00:00",
        sender="user",
        kind="user",
        text="@light выключи",
        mentions=["light"],
    )
    await dispatcher.handle(message)
    dispatcher._start.assert_not_called()
    roster.get_enabled_ids.assert_not_awaited()


async def test_council_saves_remembered_facts():
    runner = MagicMock()
    runner.get_user_state = AsyncMock(return_value=make_user_state(persona="bro"))
    runner.prompts.render_task = MagicMock(return_value="t")
    runner.memory.add_fact = AsyncMock(return_value=MagicMock())
    decision = ConsulDecision(
        reasoning="r",
        reply="Запомнил.",
        steps=[],
        remember=["Не сообщать про влажность"],
        remember_today=[],
    )
    runner.ask = AsyncMock(return_value=(decision, SwarmUsage(model="m")))
    bus = MagicMock(
        post=AsyncMock(return_value=MagicMock(spec=SwarmMessage, depth=1)),
        get_history=AsyncMock(return_value=[]),
    )
    council = SwarmCouncil(hub=SwarmHub(), bus=bus, runner=runner)
    council.on_memory_change = AsyncMock()
    trigger = SwarmMessage(
        message_id="1-0",
        created_at="2026-10-22T10:00:00+00:00",
        sender="user",
        kind="user",
        text="не говори мне про влажность",
    )
    # Only the consul is awake: no discussion round, one decision
    await council.discuss(trigger, ["butler"])
    runner.memory.add_fact.assert_awaited_once_with("Не сообщать про влажность")
    council.on_memory_change.assert_awaited_once()


async def test_home_check_ignores_quiet_until():
    ticker = _ticker(_outcome("", quiet="23:00"))
    await ticker._watch("home", SwarmSettings(), _at(22, 12), "ru")
    ticker.memory.set_quiet_until.assert_not_awaited()


async def test_empty_note_is_not_saved():
    ticker = _ticker(
        SwarmWatchReply(
            reply="",
            importance="low",
            today_note=" ",
            follow_up=False,
            quiet_until="",
            remember_today=[],
        )
    )
    await ticker._watch("mail", SwarmSettings(), _at(22, 12), "ru")
    ticker.memory.add_day_note.assert_not_awaited()


def test_night_check_moves_to_the_morning():
    from app.agents.swarm.schedule import plan_delay_seconds

    swarm_settings = SwarmSettings(interval_min_minutes=3, interval_max_minutes=5)
    night = _at(23, 0, 38)
    delays = [plan_delay_seconds(night, swarm_settings) for _ in range(30)]
    morning = (_at(23, 8) - night).total_seconds()
    assert all(morning <= d <= morning + 180 for d in delays)
    late = _at(22, 23, 58)
    assert all(
        (_at(23, 8) - late).total_seconds() <= plan_delay_seconds(late, swarm_settings)
        for _ in range(30)
    )
    day = _at(22, 12)
    assert all(180 <= plan_delay_seconds(day, swarm_settings) <= 300 for _ in range(30))


async def test_agents_do_not_get_check_lines():
    from app.agents.swarm.bus import SwarmBus

    bus = SwarmBus.__new__(SwarmBus)
    entries = [
        (
            f"{i}-0",
            {
                "data": SwarmMessage(
                    message_id="",
                    created_at="2026-10-23T10:00:00+00:00",
                    sender="mail",
                    kind=kind,
                    text=str(i),
                ).model_dump_json(exclude={"message_id"})
            },
        )
        for i, kind in [(4, "check"), (3, "user"), (2, "check"), (1, "alert")]
    ]
    bus.redis = MagicMock(xrevrange=AsyncMock(return_value=entries))
    agents = await bus.get_history(2)
    assert [m.text for m in agents] == ["1", "3"]
    feed = await bus.get_history(4, with_checks=True)
    assert [m.text for m in feed] == ["1", "2", "3", "4"]


def test_alert_reaches_display_in_cooldown():
    dispatcher = SwarmDispatcher(
        hub=SwarmHub(),
        bus=MagicMock(post=AsyncMock()),
        roster=MagicMock(),
        runner=MagicMock(),
        council=MagicMock(),
        memory=MagicMock(),
    )
    dispatcher._last_run["display"] = __import__("time").monotonic()
    alert = SwarmMessage(
        message_id="2-0",
        created_at="2026-10-23T10:00:00+00:00",
        sender="home",
        kind="alert",
        text="Окна открыты",
        mentions=["display"],
        importance="normal",
    )
    assert dispatcher.select_targets(alert, ALL_AGENTS) == ["display"]


async def test_check_keeps_the_facts_it_chose_for_today():
    outcome = SwarmWatchReply(
        reply="",
        importance="low",
        today_note="13°, дождь к вечеру",
        follow_up=False,
        quiet_until="",
        remember_today=["Сегодня 6–15°, дождь после 19:00"],
    )
    ticker = _ticker(outcome)
    await ticker._watch("weather", SwarmSettings(), _at(23, 9), "ru")
    ticker.memory.add_today_facts.assert_awaited_once_with(
        _at(23, 9).date(), ["Сегодня 6–15°, дождь после 19:00"]
    )


async def test_all_agents_see_today_memory():
    memory = _memory()
    memory.get_today_facts = AsyncMock(return_value=["Сегодня дождь после 19:00"])
    runner = SwarmAgentRunner(
        tool_registry=tool_registry, state_service=MagicMock(), memory=memory
    )
    lines = await runner.prompts.build_memory_lines(make_user_state(), None)
    assert lines == ["Today's memory:", "- Сегодня дождь после 19:00"]


def test_weather_agent_reads_weather_only():
    weather = SWARM_AGENTS["weather"]
    assert (
        check_tool_call(weather, _call("weather_tool", city="Bad Mergentheim"))
        == "allow"
    )
    assert check_tool_call(weather, _call("announce_tool", text="x")) == "block"


def test_main_chat_does_not_get_the_weather_tool():
    names = {t["name"] for t in tool_registry.schemas("gpt-5.6-luna", chat_tool_names("plain"))}
    assert "weather_tool" not in names
    swarm = tool_registry.schemas("gpt-5.6-luna", ["weather_tool"])
    assert [t["name"] for t in swarm] == ["weather_tool"]


async def test_check_line_and_alert_keep_the_run_of_the_check():
    """ARCHIE-281: the audit groups a check and its alert by run_id."""
    ticker = _ticker(_outcome(""))
    await ticker._watch("mail", SwarmSettings(), _at(22, 12), "ru", "run-c")
    assert ticker.bus.post.await_args.kwargs["run_id"] == "run-c"
    ticker = _ticker(_outcome("Новое письмо от банка"))
    await ticker._watch("mail", SwarmSettings(), _at(22, 12), "ru", "run-a")
    assert ticker.bus.post.await_args.kwargs["run_id"] == "run-a"


async def test_cron_alert_starts_its_own_run():
    ticker = _ticker(_outcome(""))
    await ticker.post_alert("cron", "Матч через 5 минут", "high")
    assert ticker.bus.post.await_args.kwargs["run_id"] is None


# --- live events (ARCHIE-286) ---


def test_pick_follow_up_takes_the_earliest_due_check():
    now = _at(24, 18, 41)
    follow_ups = {"football": _at(24, 18, 40), "calendar": _at(24, 18, 30)}
    assert pick_follow_up(["home", "football", "calendar"], follow_ups, now) == (
        "calendar"
    )
    assert pick_follow_up(["home", "football"], follow_ups, now) == "football"
    assert pick_follow_up(["football"], {"football": _at(24, 18, 45)}, now) is None
    assert pick_follow_up([], follow_ups, now) is None


def test_follow_up_adds_a_wake_up_before_the_regular_one():
    swarm_settings = SwarmSettings()
    now, regular = _at(24, 18, 41), _at(24, 18, 53)
    assert plan_wake(now, regular, _at(24, 18, 46), swarm_settings) == _at(24, 18, 46)
    # A follow-up time that passed while another check ran: wake up at once
    assert plan_wake(now, regular, _at(24, 18, 40), swarm_settings) == now
    assert plan_wake(now, regular, _at(24, 19, 30), swarm_settings) == regular
    assert plan_wake(now, regular, None, swarm_settings) == regular
    # Outside the active hours the follow-up has no effect
    night, morning = _at(25, 0, 30), _at(25, 8, 2)
    assert plan_wake(night, morning, _at(25, 0, 20), swarm_settings) == morning


async def test_live_match_asks_follow_up_and_drops_quiet_until():
    ticker = _ticker(_outcome("Гол! 1:0", quiet="17:55", follow_up=True))
    now = _at(24, 15, 1)
    await ticker._watch("football", SwarmSettings(), now, "ru")
    ticker.memory.set_quiet_until.assert_awaited_once_with("football", None)
    ticker.memory.set_follow_up.assert_awaited_once_with(
        "football", now + timedelta(minutes=5)
    )


async def test_check_runs_when_its_quiet_time_ends():
    ticker = _ticker(_outcome("", quiet="17:55"))
    await ticker._watch("football", SwarmSettings(), _at(24, 15, 1), "ru")
    ticker.memory.set_quiet_until.assert_awaited_once_with("football", _at(24, 17, 55))
    ticker.memory.set_follow_up.assert_awaited_once_with("football", _at(24, 17, 55))


async def test_home_check_gets_no_follow_up():
    ticker = _ticker(_outcome("", follow_up=True))
    await ticker._watch("home", SwarmSettings(), _at(24, 15), "ru")
    ticker.memory.set_follow_up.assert_not_awaited()


async def test_due_follow_up_goes_before_the_oldest_check(monkeypatch):
    ticker = _ticker(_outcome(""))
    ticker.memory.get_last_task_runs = AsyncMock(
        return_value={"home": 1.0, "mail": 2.0, "calendar": 3.0, "football": 9.0}
    )
    ticker.memory.get_follow_ups = AsyncMock(return_value={"football": _at(24, 18, 5)})
    monkeypatch.setattr("app.agents.swarm.ticker.local_now", lambda _: _at(24, 18, 6))
    await ticker.run_check()
    assert ticker.runner.watch.await_args.args[1] == "football"


async def test_failed_check_drops_its_follow_up():
    ticker = _ticker(_outcome(""))
    ticker.runner.watch = AsyncMock(side_effect=RuntimeError("boom"))
    await ticker.run_check("football")
    ticker.memory.set_follow_up.assert_awaited_once_with("football", None)


async def test_follow_up_of_a_disabled_check_is_ignored():
    ticker = _ticker(_outcome(""))
    ticker.memory.get_follow_ups = AsyncMock(return_value={"football": _at(24, 18)})
    swarm_settings = SwarmSettings(tasks=["home", "football"])
    assert await ticker._next_follow_up(swarm_settings) == _at(24, 18)
    ticker.roster.get_enabled_ids = AsyncMock(return_value=ALL_AGENTS - {"football"})
    assert await ticker._next_follow_up(swarm_settings) is None
    assert await ticker._next_follow_up(SwarmSettings(tasks=["home"])) is None


class _FakeMemory:
    """Redis-free memory for a simulated swarm day."""

    def __init__(self) -> None:
        self.quiet: dict[str, datetime] = {}
        self.follow_ups: dict[str, datetime] = {}
        self.last_runs: dict[str, float] = {}
        self.clock = _at(24, 12)

    async def get_settings(self) -> SwarmSettings:
        return SwarmSettings()

    async def get_quiet_until(self) -> dict[str, datetime]:
        return dict(self.quiet)

    async def set_quiet_until(self, task: str, until: datetime | None) -> None:
        self.quiet.pop(task, None)
        if until:
            self.quiet[task] = until

    async def get_follow_ups(self) -> dict[str, datetime]:
        return dict(self.follow_ups)

    async def set_follow_up(self, task: str, at: datetime | None) -> None:
        self.follow_ups.pop(task, None)
        if at:
            self.follow_ups[task] = at

    async def get_last_task_runs(self) -> dict[str, float]:
        return dict(self.last_runs)

    async def mark_task_run(self, task: str, _: float) -> None:
        # The fake clock orders the runs, not the wall clock
        self.last_runs[task] = self.clock.timestamp()

    async def add_today_facts(self, *_) -> None:
        pass

    async def add_day_note(self, *_) -> None:
        pass

    async def set_last_run(self, *_) -> None:
        pass

    async def get_mode(self) -> str:
        return "running"

    async def get_last_run(self) -> None:
        return None

    async def list_facts(self) -> list:
        return []

    async def get_today_facts(self, _) -> list:
        return []


async def test_live_match_is_checked_every_few_minutes(monkeypatch):
    """24.10.2026: Napoli - Roma 18:00-19:55 while four other checks share the
    queue. Before kickoff the football check sets quiet_until to the kickoff;
    during the match it asks for a follow-up and also writes a quiet_until,
    which must not apply (on 24.10 a quiet_until hid a live match)."""
    kickoff, final = _at(24, 18), _at(24, 19, 55)
    memory = _FakeMemory()
    ticker = _ticker(_outcome(""))
    ticker.memory = memory

    async def watch(_profile, task, _history, _prompt):
        live = task == "football" and kickoff <= memory.clock < final
        quiet = "18:00" if task == "football" and memory.clock < kickoff else ""
        return SwarmWatchDraft(
            outcome=_outcome("", quiet="23:59" if live else quiet, follow_up=live),
            usage=SwarmUsage(model="m"),
        )

    ticker.runner.watch = AsyncMock(side_effect=watch)
    monkeypatch.setattr("app.agents.swarm.ticker.local_now", lambda _: memory.clock)
    football: list[datetime] = []
    others = 0
    while memory.clock < _at(24, 21):
        memory.clock += timedelta(seconds=await ticker.plan_next_run())
        before = ticker.runner.watch.await_count
        await ticker.run_check()
        assert ticker.runner.watch.await_count == before + 1
        task = ticker.runner.watch.await_args.args[1]
        if task == "football":
            football.append(memory.clock)
        elif kickoff <= memory.clock < final:
            others += 1
    # The first check sets the quiet time; when it ends, the check runs at once
    assert football[0] < _at(24, 17) and football[1] == kickoff
    during = [t for t in football if kickoff <= t <= final]
    gaps = [(b - a).total_seconds() for a, b in pairwise(during)]
    assert max(gaps) <= 5 * 60 + 1
    assert during[-1] >= final - timedelta(minutes=5)
    # The final score check is followed by no more follow-ups
    assert "football" not in memory.follow_ups
    assert others >= 4
