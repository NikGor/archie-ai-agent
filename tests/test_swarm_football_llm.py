"""Real swarm_watch prompt for the football check against fake search results (ARCHIE-286).

24.10.2026 the search gave stale live scores ("0:0, 41'" while the match was
2:2) and the check trusted them; a live match also got a quiet_until, so its
final whistle was never reported. Run: ./execute_tests.sh -q -m llm -k football_live
"""

import re
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo
import pytest
from app.platform.tools.models import ToolResult
from app.swarm.agents import SWARM_AGENTS
from app.swarm.memory import DayNote
from app.swarm.runner import SwarmAgentRunner
import app.swarm.tools  # noqa: F401
from app.tools.catalog import tool_registry
from tests.conftest import make_user_state


BERLIN = ZoneInfo("Europe/Berlin")

# What the search gave at 19:21: 0:0 and a minute that does not fit an 18:00 kickoff
NAPOLI_STALE = (
    "Live: Napoli 0-0 Roma, about 85th minute (Serie A, kickoff 18:00 CEST). "
    "Borussia Dortmund 0-0 Eintracht Frankfurt, about 55th minute (kickoff 18:30 CEST)."
)
NAPOLI_FRESH = (
    "Live, 24.10.2026: Napoli 2-2 Roma, 67th minute (kickoff 18:00 CEST). "
    "Goals: 28' McTominay (Napoli), 48' Dybala (Roma), 59' Dovbyk (Roma), "
    "64' Lukaku (Napoli). Borussia Dortmund 0-0 Eintracht Frankfurt, 51st minute "
    "(kickoff 18:30 CEST)."
)
VILLA_LIVE = (
    "Live, 24.10.2026: Aston Villa 2-1 Manchester City, 70th minute (kickoff "
    "13:30 CEST). Goals: 6' Jackson, 53' Haaland, 60' Jackson. Later today: "
    "Napoli - Roma 18:00 CEST, Borussia Dortmund - Eintracht Frankfurt 18:30 CEST."
)

SCENARIOS = {
    # The first search gives the stale score; any later search the fresh one.
    "stale_then_fresh": {
        "now": datetime(2026, 10, 24, 19, 21, tzinfo=BERLIN),
        "answers": [NAPOLI_STALE, NAPOLI_FRESH],
        "notes": [
            DayNote(
                time="08:16",
                task="football",
                note="24.10.2026: Napoli - Roma 18:00, Borussia Dortmund - Eintracht 18:30.",
            ),
            DayNote(
                time="18:00",
                task="football",
                note="Napoli - Roma live, 1st minute, 0:0. Dortmund - Eintracht 18:30.",
                reported="Наполи — Рома: матч начался, 0:0.",
            ),
            DayNote(
                time="18:41",
                task="football",
                note="Napoli - Roma 0:0, 41st minute; Dortmund - Eintracht 0:0, 11th minute.",
            ),
        ],
        "goal": "2:2",
    },
    # 15:01: a goal in a live match, the next match starts at 18:00.
    "live_with_later_match": {
        "now": datetime(2026, 10, 24, 15, 1, tzinfo=BERLIN),
        "answers": [VILLA_LIVE],
        "notes": [
            DayNote(
                time="08:16",
                task="football",
                note=(
                    "24.10.2026: Aston Villa - Manchester City 13:30, Napoli - Roma "
                    "18:00, Borussia Dortmund - Eintracht 18:30."
                ),
            ),
            DayNote(
                time="14:21",
                task="football",
                note="Aston Villa - Manchester City, half-time 1:0, Jackson 6'.",
                reported="Астон Вилла ведёт 1:0 у Манчестер Сити к перерыву, Джексон 6'.",
            ),
        ],
        "goal": "2:1",
    },
}


def _runner(scenario: dict) -> SwarmAgentRunner:
    state_service = MagicMock()
    state_service.get_user_state = AsyncMock(
        return_value=make_user_state(
            user_timezone="Europe/Berlin", language="ru", default_city="Bad Mergentheim"
        )
    )
    memory = MagicMock()
    memory.list_facts = AsyncMock(return_value=[])
    memory.get_today_facts = AsyncMock(return_value=[])
    memory.get_day_notes = AsyncMock(return_value=scenario["notes"])
    roster = MagicMock(
        get_owner=AsyncMock(return_value="Niko"),
        get_model=AsyncMock(side_effect=lambda profile: profile.default_model),
    )
    runner = SwarmAgentRunner(
        tool_registry=tool_registry,
        state_service=state_service,
        roster=roster,
        memory=memory,
    )
    now = scenario["now"]
    runner.datetime_note = MagicMock(
        return_value=f"Current date/time: {now:%A, %d.%m.%Y, %H:%M} (Europe/Berlin)"
    )
    return runner


def _fake_search(answers: list[str], queries: list[str]):
    """Tool round N gets answers[N] (the last one for later rounds)."""
    rounds = [0]

    async def execute(tool_calls, **_):
        answer = answers[min(rounds[0], len(answers) - 1)]
        rounds[0] += 1
        results = []
        for call in tool_calls:
            query = " ".join(str(p.value) for p in call.arguments)
            queries.append(query)
            results.append(
                ToolResult(
                    tool_name=call.tool_name,
                    output={"success": True, "query": query, "answer": answer},
                )
            )
        return results

    return execute, rounds


@pytest.mark.llm
@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_football_live_check_reports_goal_and_follows_up(name):
    scenario = SCENARIOS[name]
    runner = _runner(scenario)
    queries: list[str] = []
    execute, rounds = _fake_search(scenario["answers"], queries)
    with (
        patch("app.swarm.runner.local_now", return_value=scenario["now"]),
        patch(
            "app.swarm.runner.execute_tool_calls",
            side_effect=execute,
        ),
    ):
        draft = await runner.watch(
            SWARM_AGENTS["football"],
            "football",
            [],
            runner.render_task("swarm_watch", task="football", teams=[], language="ru"),
        )
    outcome = draft.outcome
    print(
        f"\n{name}: rounds={rounds[0]} queries={queries}\noutcome={outcome.model_dump()}"
    )
    assert len(queries) >= len(
        scenario["answers"]
    ), "stale score was not searched again"
    home, away = scenario["goal"].split(":")
    assert re.search(
        rf"{home}\s*[:\-–—]\s*{away}", outcome.reply
    ), "no alert for the goal"
    assert outcome.follow_up is True
    assert outcome.quiet_until.strip() == ""
