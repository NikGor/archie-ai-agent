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
        assert detail == "Creating event: Dinner with John"

    def test_events_today(self):
        detail = get_tool_detail("events_tool", {"action": "today"})
        assert detail == "Checking today's calendar"

    def test_events_upcoming(self):
        detail = get_tool_detail("events_tool", {"action": "upcoming"})
        assert detail == "Checking upcoming events"

    def test_task_create(self):
        detail = get_tool_detail("task_tool", {"action": "create", "title": "Buy groceries"})
        assert detail == "Creating task: Buy groceries"

    def test_task_list(self):
        detail = get_tool_detail("task_tool", {"action": "list"})
        assert detail == "Loading tasks list"

    def test_notes_search(self):
        detail = get_tool_detail("notes_tool", {"action": "search", "search_query": "meeting notes"})
        assert detail == "Searching notes: meeting notes"

    def test_spotify_search(self):
        detail = get_tool_detail("spotify_tool", {"action": "search", "query": "Beatles"})
        assert detail == "Searching Spotify: Beatles"

    def test_spotify_get_current(self):
        detail = get_tool_detail("spotify_tool", {"action": "get_current"})
        assert detail == "Current track"

    def test_light_with_device(self):
        detail = get_tool_detail("light_control_tool", {"device_name": "Kitchen Light"})
        assert detail == "Controlling light: Kitchen Light"

    def test_climate_set_temp(self):
        detail = get_tool_detail("climate_control_tool", {"action": "set_temperature", "temperature": "22"})
        assert detail == "Setting temperature: 22°"

    def test_football_live(self):
        detail = get_tool_detail("football_tool", {"action": "live_scores", "team": "Barcelona"})
        assert detail == "Checking live scores: Barcelona"

    def test_document_search(self):
        detail = get_tool_detail("document_search_tool", {"query": "project plan"})
        assert detail == "Searching documents: project plan"

    def test_unknown_tool_returns_none(self):
        detail = get_tool_detail("unknown_tool", {"query": "test"})
        assert detail is None

    def test_empty_args_returns_none(self):
        detail = get_tool_detail("google_search_tool", {})
        assert detail is None
