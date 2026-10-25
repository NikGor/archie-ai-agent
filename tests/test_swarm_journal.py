"""Tests for the swarm journal that sends swarm messages to Django (ARCHIE-281)."""

import asyncio
from unittest.mock import AsyncMock, patch
import httpx
from archie_shared.swarm.models import (
    SwarmJournalBatch,
    SwarmMessage,
    SwarmMessageEvent,
)
from app.agents.swarm.bus import SwarmHub
from app.agents.swarm.journal import SwarmJournal


def _message(message_id: str) -> SwarmMessage:
    return SwarmMessage(
        message_id=message_id,
        run_id="run-1",
        created_at="2026-10-24T10:00:00+00:00",
        sender="calendar",
        kind="alert",
        text="Встреча в 11:00",
    )


def _journal(monkeypatch, token: str | None = "t") -> SwarmJournal:
    monkeypatch.setattr("app.agents.swarm.journal.settings.eval_api_token", token)
    monkeypatch.setattr("app.agents.swarm.journal.settings.swarm_journal_buffer_size", 3)
    return SwarmJournal(SwarmHub())


async def test_flush_sends_the_buffer_as_one_batch(monkeypatch):
    journal = _journal(monkeypatch)
    journal.add(_message("1-0"))
    journal.add(_message("2-0"))
    with patch.object(journal, "_send", AsyncMock()) as send:
        await journal.flush()
    assert [m.message_id for m in send.await_args.args[0]] == ["1-0", "2-0"]
    assert not journal.buffer


async def test_failed_send_keeps_the_buffer_for_the_next_flush(monkeypatch):
    journal = _journal(monkeypatch)
    journal.add(_message("1-0"))
    with patch.object(
        journal, "_send", AsyncMock(side_effect=httpx.ConnectError("down"))
    ):
        await journal.flush()
    assert [m.message_id for m in journal.buffer] == ["1-0"]


def test_full_buffer_drops_the_oldest(monkeypatch):
    journal = _journal(monkeypatch)
    for number in range(5):
        journal.add(_message(f"{number}-0"))
    assert [m.message_id for m in journal.buffer] == ["2-0", "3-0", "4-0"]


async def test_journal_without_token_does_not_listen(monkeypatch):
    journal = _journal(monkeypatch, token=None)
    await journal.start()
    assert journal._worker is None


async def test_send_posts_the_batch_with_the_service_token(monkeypatch):
    journal = _journal(monkeypatch)
    monkeypatch.setattr("app.agents.swarm.journal.settings.web_django_url", "http://web:8000/")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"stored": 1})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    with patch(
        "app.agents.swarm.journal.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    ):
        await journal._send([_message("1-0")])
    request = requests[0]
    assert str(request.url) == "http://web:8000/swarm-journal/api/ingest/"
    assert request.headers["Authorization"] == "Bearer t"
    batch = SwarmJournalBatch.model_validate_json(request.content)
    assert batch.messages[0].run_id == "run-1"


async def test_started_journal_buffers_posted_messages(monkeypatch):
    journal = _journal(monkeypatch)
    monkeypatch.setattr("app.agents.swarm.journal.settings.swarm_journal_flush_seconds", 60)
    await journal.start()
    journal.hub.publish(SwarmMessageEvent(message=_message("1-0")))
    await asyncio.sleep(0)
    assert [m.message_id for m in journal.buffer] == ["1-0"]
    with patch.object(journal, "_send", AsyncMock()) as send:
        await journal.stop()
    # Stop sends what is left
    send.assert_awaited_once()
