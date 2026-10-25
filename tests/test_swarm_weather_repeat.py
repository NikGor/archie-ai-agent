"""The weather check does not repeat an alert about the same weather (ARCHIE-287).

Runs the real swarm_watch prompt for the "weather" task through
SwarmAgentRunner.watch on settings.swarm_model, with a fake weather_tool result
and day notes that already reported the same rain period. Select with -m llm.
"""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo
import pytest
from app.config import settings
from app.platform.tools.models import ToolResult
from app.swarm.agents import SWARM_AGENTS
from app.swarm.memory import DayNote
from app.swarm.runner import SwarmAgentRunner
import app.swarm.tools  # noqa: F401
from app.tools.catalog import tool_registry
from tests.conftest import make_user_state

pytestmark = pytest.mark.llm

ZONE = ZoneInfo("Europe/Berlin")

# The day as the check saw it on 24.10.2026: the night rain is already reported.
NOTES = [
    DayNote(
        time="14:02",
        task="weather",
        note="14:00: сухо, 10,8 °C; до 23:00 сухо, в 02:00 лёгкий дождь 93%.",
        reported=(
            "Ночью прогноз ухудшился: около 02:00 ожидается лёгкий дождь "
            "с вероятностью 93%. До 23:00 сегодня сухо."
        ),
    ),
    DayNote(
        time="17:03",
        task="weather",
        note=(
            "17:00, Бад-Мергентхайм: 10,6 °C, переменная облачность, осадков нет. "
            "Сухо до 23:00; в 02:00 дождь 93%, завтра 21,2 мм, вероятность 100%."
        ),
    ),
    DayNote(
        time="19:54",
        task="weather",
        note=(
            "19:45, Бад-Мергентхайм: 8,8 °C, облачно, осадков 0 мм. В 22:00 дождь 0%, "
            "в 01:00 — 43%, в 04:00 — 100%; 25.10 — 22,7 мм, вероятность 100%."
        ),
    ),
]
TODAY_FACTS = [
    "По свежему прогнозу, лёгкий дождь ночью ожидается около 02:00 с вероятностью 93%; до 23:00 сухо.",
    "На воскресенье, 25.10, прогноз дождливый: вероятность осадков 100%, около 22,7 мм.",
]

# The evening rain is forecast and reported; at 21:15 it starts as forecast.
NOTES_EVENING_RAIN = [
    DayNote(
        time="18:26",
        task="weather",
        note="18:15: сухо, 8,7 °C; с 21:00 лёгкий дождь 90%, ночью 100%.",
        reported=(
            "Вечером около 21:00 начнётся лёгкий дождь (90%), ночью он продолжится "
            "с вероятностью 100%. Сейчас сухо, 8,7 °C."
        ),
    ),
    DayNote(
        time="20:54",
        task="weather",
        note=(
            "20:45: облачно, сухо, 8,4 °C. В 21:00 лёгкий дождь 90%, в 00:00 — 100%, "
            "в 03:00 — 100%; 25.10 — 23,5 мм, вероятность 100%."
        ),
    ),
]
TODAY_FACTS_EVENING_RAIN = [
    "Вечером около 21:00 начнётся лёгкий дождь, ночью он продолжится.",
    "На воскресенье, 25.10, прогноз дождливый: вероятность осадков 100%, около 22,7 мм.",
]


def _hour(time: str, temperature: float, probability: int, condition: str) -> dict:
    return {
        "time": time,
        "temperature_c": temperature,
        "precipitation_probability": probability,
        "condition": condition,
    }


def _weather(
    time: str,
    condition: str,
    precipitation_mm: float,
    next_hours: list[dict],
    day_mm: float,
    day_condition: str = "light rain",
) -> dict[str, Any]:
    return {
        "location": "Bad Mergentheim, Germany",
        "current": {
            "time": time,
            "temperature_c": 8.7,
            "feels_like_c": 5.8,
            "humidity_percent": 88,
            "precipitation_mm": precipitation_mm,
            "cloud_cover_percent": 100,
            "wind_ms": 3.1,
            "wind_gusts_ms": 7.2,
            "condition": condition,
        },
        "next_hours": next_hours,
        "days": [
            {
                "date": "24.10.2026",
                "condition": day_condition,
                "temperature_min_c": 6.8,
                "temperature_max_c": 11.6,
                "precipitation_mm": day_mm,
                "precipitation_probability": 95,
                "wind_max_ms": 6.9,
                "sunrise": "07:55",
                "sunset": "18:13",
            }
        ],
    }


CASES = {
    # (a) The same night rain, the time and probability changed a little.
    "refined_forecast": (
        "20:15",
        NOTES,
        TODAY_FACTS,
        _weather(
            "20:15",
            "overcast",
            0.0,
            [
                _hour("20:00", 8.7, 0, "overcast"),
                _hour("23:00", 8.3, 3, "overcast"),
                _hour("02:00", 8.4, 95, "light rain"),
                _hour("05:00", 8.6, 100, "light rain"),
                _hour("08:00", 9.1, 100, "light rain"),
            ],
            1.2,
        ),
    ),
    # (b) The rain starts as forecast and as already reported.
    "rain_started_as_forecast": (
        "21:15",
        NOTES_EVENING_RAIN,
        TODAY_FACTS_EVENING_RAIN,
        _weather(
            "21:15",
            "light rain",
            0.2,
            [
                _hour("21:00", 8.3, 90, "light rain"),
                _hour("00:00", 8.2, 100, "light rain"),
                _hour("03:00", 8.4, 100, "light rain"),
                _hour("06:00", 8.6, 100, "light rain"),
                _hour("09:00", 9.4, 95, "light rain"),
            ],
            1.4,
        ),
    ),
    # (c) A real change: heavy rain and a thunderstorm instead of light rain.
    "storm_instead_of_light_rain": (
        "20:15",
        NOTES,
        TODAY_FACTS,
        _weather(
            "20:15",
            "overcast",
            0.0,
            [
                _hour("20:00", 8.7, 10, "overcast"),
                _hour("23:00", 8.9, 100, "thunderstorm"),
                _hour("02:00", 8.4, 100, "heavy rain"),
                _hour("05:00", 8.6, 100, "heavy rain"),
                _hour("08:00", 9.1, 90, "rain"),
            ],
            25.0,
            "thunderstorm",
        ),
    ),
}
EXPECT_ALERT = {"storm_instead_of_light_rain"}


def _runner(notes: list[DayNote], facts: list[str]) -> SwarmAgentRunner:
    state_service = MagicMock()
    state_service.get_user_state = AsyncMock(
        return_value=make_user_state(
            default_city="Bad Mergentheim",
            default_country="Germany",
            user_timezone="Europe/Berlin",
            language="ru",
        )
    )
    memory = MagicMock()
    memory.list_facts = AsyncMock(return_value=[])
    memory.get_day_notes = AsyncMock(return_value=notes)
    memory.get_today_facts = AsyncMock(return_value=facts)
    roster = MagicMock(
        get_owner=AsyncMock(return_value="Niko"),
        get_model=AsyncMock(return_value=settings.swarm_model),
    )
    return SwarmAgentRunner(
        tool_registry=tool_registry,
        state_service=state_service,
        roster=roster,
        memory=memory,
    )


@pytest.mark.parametrize("case", list(CASES))
async def test_weather_check_reports_only_a_real_change(case: str):
    time, notes, facts, weather = CASES[case]
    now = datetime(2026, 10, 24, *map(int, time.split(":")), tzinfo=ZONE)
    runner = _runner(notes, facts)
    runner.datetime_note = MagicMock(
        return_value=f"Current date/time: Saturday, 24.10.2026, {time} (Europe/Berlin)"
    )
    tools = AsyncMock(
        side_effect=lambda tool_calls, **_: [
            ToolResult(tool_name=c.tool_name, output=weather) for c in tool_calls
        ]
    )
    task = runner.render_task("swarm_watch", task="weather", teams=[], language="ru")
    with (
        patch("app.swarm.runner.local_now", return_value=now),
        patch("app.swarm.runner.execute_tool_calls", tools),
    ):
        draft = await runner.watch(SWARM_AGENTS["weather"], "weather", [], task)
    outcome = draft.outcome
    print(f"\n[{case}] importance={outcome.importance} reply={outcome.reply!r}")
    assert tools.await_count >= 1
    if case in EXPECT_ALERT:
        assert outcome.reply.strip(), "a real change of the weather must be reported"
        assert outcome.importance != "low", "dangerous weather goes to the display"
    else:
        assert not outcome.reply.strip(), f"repeated alert: {outcome.reply}"
