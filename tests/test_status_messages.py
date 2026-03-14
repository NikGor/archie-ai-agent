"""Tests for status_messages utility."""

from app.utils.status_messages import get_tool_detail


class TestGetToolDetail:
    def test_google_search(self):
        detail = get_tool_detail("google_search_tool", {"query": "best restaurants in Berlin"})
        assert detail == "Google: best restaurants in Berlin"

    def test_google_places(self):
        detail = get_tool_detail("google_places_search_tool", {"query": "pizza near me"})
        assert detail == "Google Places: pizza near me"

    def test_events_create(self):
        detail = get_tool_detail("events_tool", {"action": "create", "title": "Dinner with John"})
        assert detail == "Создаю событие: Dinner with John"

    def test_events_today(self):
        detail = get_tool_detail("events_tool", {"action": "today"})
        assert detail == "Проверяю календарь на сегодня"

    def test_events_upcoming(self):
        detail = get_tool_detail("events_tool", {"action": "upcoming"})
        assert detail == "Проверяю ближайшие события"

    def test_task_create(self):
        detail = get_tool_detail("task_tool", {"action": "create", "title": "Buy groceries"})
        assert detail == "Создаю задачу: Buy groceries"

    def test_task_list(self):
        detail = get_tool_detail("task_tool", {"action": "list"})
        assert detail == "Загружаю список задач"

    def test_notes_search(self):
        detail = get_tool_detail("notes_tool", {"action": "search", "search_query": "meeting notes"})
        assert detail == "Ищу в заметках: meeting notes"

    def test_spotify_search(self):
        detail = get_tool_detail("spotify_tool", {"action": "search", "query": "Beatles"})
        assert detail == "Ищу в Spotify: Beatles"

    def test_spotify_get_current(self):
        detail = get_tool_detail("spotify_tool", {"action": "get_current"})
        assert detail == "Текущий трек"

    def test_light_with_device(self):
        detail = get_tool_detail("light_control_tool", {"device_name": "Kitchen Light"})
        assert detail == "Управляю светом: Kitchen Light"

    def test_climate_set_temp(self):
        detail = get_tool_detail("climate_control_tool", {"action": "set_temperature", "temperature": "22"})
        assert detail == "Устанавливаю температуру: 22°"

    def test_football_live(self):
        detail = get_tool_detail("football_tool", {"action": "live_scores", "team": "Barcelona"})
        assert detail == "Проверяю live-счёт: Barcelona"

    def test_document_search(self):
        detail = get_tool_detail("document_search_tool", {"query": "project plan"})
        assert detail == "Ищу в документах: project plan"

    def test_image_gen(self):
        detail = get_tool_detail("fast_image_generation_tool", {"prompt": "sunset over the ocean"})
        assert detail == "Генерирую изображение: sunset over the ocean"

    def test_image_gen_long_prompt_truncated(self):
        long_prompt = "a" * 100
        detail = get_tool_detail("fast_image_generation_tool", {"prompt": long_prompt})
        assert detail is not None
        assert len(detail) < 100
        assert detail.endswith("...")

    def test_unknown_tool_returns_none(self):
        detail = get_tool_detail("unknown_tool", {"query": "test"})
        assert detail is None

    def test_empty_args_returns_none(self):
        detail = get_tool_detail("google_search_tool", {})
        assert detail is None
