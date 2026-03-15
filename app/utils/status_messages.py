"""Human-readable status messages for tool calls.

Maps tool names to functions that extract a descriptive detail
from tool call arguments, so StatusUpdate messages are informative
instead of generic.
"""

from typing import Any


def _google_search_detail(args: dict[str, Any]) -> str:
    query = args.get("query", "")
    return f"Google: {query}" if query else ""


def _google_places_detail(args: dict[str, Any]) -> str:
    query = args.get("query", "")
    return f"Google Places: {query}" if query else ""


def _events_detail(args: dict[str, Any]) -> str:  # noqa: PLR0911
    action = args.get("action", "")
    title = args.get("title", "")
    if action == "create" and title:
        return f"Создаю событие: {title}"
    if action == "today":
        return "Проверяю календарь на сегодня"
    if action == "upcoming":
        return "Проверяю ближайшие события"
    if action == "list":
        return "Загружаю список событий"
    if action == "delete" and title:
        return f"Удаляю событие: {title}"
    if action == "update" and title:
        return f"Обновляю событие: {title}"
    return f"Календарь: {action}" if action else ""


def _task_detail(args: dict[str, Any]) -> str:
    action = args.get("action", "")
    title = args.get("title", "")
    if action == "create" and title:
        return f"Создаю задачу: {title}"
    if action == "list":
        return "Загружаю список задач"
    if action == "complete" and title:
        return f"Завершаю задачу: {title}"
    return f"Задачи: {action}" if action else ""


def _notes_detail(args: dict[str, Any]) -> str:
    action = args.get("action", "")
    title = args.get("title", "")
    search_query = args.get("search_query", "")
    if action == "search" and search_query:
        return f"Ищу в заметках: {search_query}"
    if action == "create" and title:
        return f"Создаю заметку: {title}"
    if action == "list":
        return "Загружаю список заметок"
    return f"Заметки: {action}" if action else ""


def _spotify_detail(args: dict[str, Any]) -> str:
    action = args.get("action", "")
    query = args.get("query", "")
    if action == "search" and query:
        return f"Ищу в Spotify: {query}"
    if action == "play":
        return f"Воспроизвожу: {query}" if query else "Воспроизведение"
    if action == "get_current":
        return "Текущий трек"
    return f"Spotify: {action}" if action else ""


def _light_detail(args: dict[str, Any]) -> str:
    device = args.get("device_name", "")
    return f"Управляю светом: {device}" if device else "Управляю светом"


def _climate_detail(args: dict[str, Any]) -> str:
    action = args.get("action", "")
    temp = args.get("temperature", "")
    if action == "set_temperature" and temp:
        return f"Устанавливаю температуру: {temp}°"
    return f"Климат: {action}" if action else "Управляю климатом"


def _football_detail(args: dict[str, Any]) -> str:
    action = args.get("action", "")
    team = args.get("team", "")
    if action == "live_scores":
        return "Проверяю live-счёт" + (f": {team}" if team else "")
    if action == "fixtures":
        return "Расписание матчей" + (f": {team}" if team else "")
    if action == "standings":
        league = args.get("league", "")
        return "Таблица" + (f": {league}" if league else "")
    return f"Футбол: {action}" if action else ""


def _document_search_detail(args: dict[str, Any]) -> str:
    query = args.get("query", "")
    return f"Ищу в документах: {query}" if query else ""


def _image_gen_detail(args: dict[str, Any]) -> str:
    prompt = args.get("prompt", "")
    if prompt and len(prompt) > 60:
        prompt = prompt[:57] + "..."
    return f"Генерирую изображение: {prompt}" if prompt else "Генерирую изображение"


_TOOL_DETAIL_MAP: dict[str, Any] = {
    "google_search_tool": _google_search_detail,
    "google_places_search_tool": _google_places_detail,
    "events_tool": _events_detail,
    "task_tool": _task_detail,
    "notes_tool": _notes_detail,
    "spotify_tool": _spotify_detail,
    "light_control_tool": _light_detail,
    "climate_control_tool": _climate_detail,
    "football_tool": _football_detail,
    "document_search_tool": _document_search_detail,
    "fast_image_generation_tool": _image_gen_detail,
    "profi_image_generation_tool": _image_gen_detail,
}


def get_tool_detail(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Get human-readable detail for a tool call, or None if unavailable."""
    fn = _TOOL_DETAIL_MAP.get(tool_name)
    if fn is None:
        return None
    detail = fn(arguments)
    return detail or None
