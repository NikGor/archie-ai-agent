from unittest.mock import AsyncMock, patch
import httpx
import pytest
from archie_shared.ui.models import (
    AssistantButton,
    Content,
    FrontendButton,
    Level2Answer,
    LocationCard,
    QuickActionButtons,
    TextAnswer,
)
from app.utils.url_validator import _collect_tasks, validate_and_fix_urls


@pytest.fixture
def mock_reachable_url(monkeypatch):
    async def fake_head(self, url, follow_redirects=True):
        return httpx.Response(200, request=httpx.Request("HEAD", url))

    monkeypatch.setattr(httpx.AsyncClient, "head", fake_head)


@pytest.fixture
def mock_unreachable_url(monkeypatch):
    async def fake_head(self, url, follow_redirects=True):
        return httpx.Response(404, request=httpx.Request("HEAD", url))

    monkeypatch.setattr(httpx.AsyncClient, "head", fake_head)


def _button(url: str, command: str = "url_to", text: str = "Open link") -> FrontendButton:
    return FrontendButton(type="frontend_button", text=text, command=command, url=url)


def _no_fallback_mock() -> AsyncMock:
    return AsyncMock(return_value={"success": True, "sources": []})


def _fallback_mock(url: str) -> AsyncMock:
    return AsyncMock(return_value={"success": True, "sources": [{"index": 1, "title": "Result", "url": url}]})


async def test_valid_button_url_passes_unchanged(mock_reachable_url):
    buttons = [_button("https://example.com/valid")]
    tasks: list = []
    _collect_tasks(buttons, tasks)
    assert len(tasks) == 1
    await tasks[0]
    assert isinstance(buttons[0], FrontendButton)
    assert buttons[0].url == "https://example.com/valid"


async def test_broken_button_url_with_no_fallback_converts_to_assistant_button(mock_unreachable_url):
    buttons = [_button("https://example.com/broken", command="open_on_youtube_video")]
    with patch("app.utils.url_validator.google_search_tool", new=_no_fallback_mock()):
        tasks: list = []
        _collect_tasks(buttons, tasks)
        await tasks[0]

    assert isinstance(buttons[0], AssistantButton)
    assert buttons[0].assistant_request == "Open link"


async def test_broken_button_url_with_search_fallback_replaces_url(mock_unreachable_url):
    buttons = [_button("https://example.com/broken", command="check_amazon")]
    with patch(
        "app.utils.url_validator.google_search_tool",
        new=_fallback_mock("https://amazon.com/replacement"),
    ):
        tasks: list = []
        _collect_tasks(buttons, tasks)
        await tasks[0]

    assert isinstance(buttons[0], FrontendButton)
    assert buttons[0].url == "https://amazon.com/replacement"


async def test_valid_location_card_map_url_passes_unchanged(mock_reachable_url):
    card = LocationCard(title="Cafe", open_map_url="https://maps.google.com/valid")
    tasks: list = []
    _collect_tasks(card, tasks)
    assert len(tasks) == 1
    await tasks[0]
    assert card.open_map_url == "https://maps.google.com/valid"


async def test_broken_location_card_map_url_is_set_to_none(mock_unreachable_url):
    card = LocationCard(
        title="Cafe",
        address="123 Main St",
        open_map_url="https://maps.google.com/broken",
    )
    tasks: list = []
    _collect_tasks(card, tasks)
    await tasks[0]

    assert card.open_map_url is None
    assert card.title == "Cafe"
    assert card.address == "123 Main St"


async def test_valid_markdown_link_passes_unchanged(mock_reachable_url):
    text_answer = TextAnswer(type="markdown", text="see [this](https://example.com/valid)")
    tasks: list = []
    _collect_tasks(text_answer, tasks)
    assert len(tasks) == 1
    await tasks[0]
    assert text_answer.text == "see [this](https://example.com/valid)"


async def test_broken_markdown_link_with_fallback_is_replaced(mock_unreachable_url):
    text_answer = TextAnswer(type="markdown", text="see [this article](https://example.com/broken)")
    with patch(
        "app.utils.url_validator.google_search_tool",
        new=_fallback_mock("https://example.com/replacement"),
    ):
        tasks: list = []
        _collect_tasks(text_answer, tasks)
        await tasks[0]

    assert text_answer.text == "see [this article](https://example.com/replacement)"


async def test_broken_markdown_link_with_no_fallback_is_stripped_to_plain_text(mock_unreachable_url):
    text_answer = TextAnswer(type="markdown", text="see [this article](https://example.com/broken)")
    with patch("app.utils.url_validator.google_search_tool", new=_no_fallback_mock()):
        tasks: list = []
        _collect_tasks(text_answer, tasks)
        await tasks[0]

    assert text_answer.text == "see this article"


async def test_content_with_no_links_is_unaffected():
    content = Content(content_format="plain", text="hello world")
    result = await validate_and_fix_urls(content)
    assert result.text == "hello world"


async def test_validate_and_fix_urls_end_to_end_on_level2_answer(mock_unreachable_url):
    text_answer = TextAnswer(type="markdown", text="check [this](https://example.com/broken)")
    level2 = Level2Answer(
        text=text_answer,
        quick_action_buttons=QuickActionButtons(
            buttons=[AssistantButton(type="assistant_button", text="Ask more", assistant_request="tell me more")]
        ),
    )
    content = Content(content_format="level2_answer", level2_answer=level2)

    with patch("app.utils.url_validator.google_search_tool", new=_no_fallback_mock()):
        result = await validate_and_fix_urls(content)

    assert result.level2_answer is not None
    assert result.level2_answer.text.text == "check this"
