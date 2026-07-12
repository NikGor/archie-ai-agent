import asyncio
import logging
import re
from collections.abc import Awaitable
from typing import Any
import httpx
from archie_shared.ui.models import AssistantButton, Content, FrontendButton, LocationCard, TextAnswer
from pydantic import BaseModel
from ..tools.google_search_tool import google_search_tool


logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_SECONDS = 3.0
_URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_FALLBACK_ELIGIBLE_COMMANDS = frozenset(
    {
        "navigate_to",
        "url_to",
        "open_on_youtube_video",
        "open_on_youtube_music",
        "check_amazon",
    }
)


async def validate_and_fix_urls(content: Content) -> Content:
    """Check every URL reachable from content and fix or remove broken ones."""
    tasks: list[Awaitable[None]] = []
    _collect_tasks(content, tasks)
    if tasks:
        await asyncio.gather(*tasks)
    return content


def _collect_tasks(node: Any, tasks: list[Awaitable[None]]) -> None:
    if isinstance(node, TextAnswer):
        if node.type == "markdown" and node.text:
            tasks.append(_fix_text_answer(node))
        return
    if isinstance(node, LocationCard):
        if node.open_map_url:
            tasks.append(_fix_location_card(node))
        _collect_tasks(node.buttons, tasks)
        return
    if isinstance(node, list):
        for index, item in enumerate(node):
            if isinstance(item, FrontendButton):
                if item.url and item.command in _FALLBACK_ELIGIBLE_COMMANDS:
                    tasks.append(_fix_frontend_button(node, index))
            else:
                _collect_tasks(item, tasks)
        return
    if isinstance(node, BaseModel):
        for field_name in node.__class__.model_fields:
            _collect_tasks(getattr(node, field_name), tasks)


async def _is_url_reachable(url: str) -> bool:
    if not _URL_PATTERN.match(url):
        return False
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
            response = await client.head(url, follow_redirects=True)
            return response.is_success
    except httpx.HTTPError as error:
        logger.warning(f"url_validator_001: URL unreachable \033[36m{url}\033[0m: {error}")
        return False


async def _find_replacement_url(query: str) -> str | None:
    search_result = await google_search_tool(query)
    sources = search_result.get("sources") or []
    if not sources:
        return None
    return sources[0].get("url") or None


async def _fix_frontend_button(container: list[Any], index: int) -> None:
    button: FrontendButton = container[index]
    url = button.url or ""
    if await _is_url_reachable(url):
        return
    replacement = await _find_replacement_url(button.text)
    if replacement:
        logger.warning(
            f"url_validator_002: Replacing broken button URL \033[36m{url}\033[0m "
            f"with \033[36m{replacement}\033[0m"
        )
        button.url = replacement
        return
    logger.warning(
        f"url_validator_003: Converting button with broken URL \033[36m{url}\033[0m "
        f"to AssistantButton"
    )
    container[index] = AssistantButton(
        type="assistant_button", text=button.text, assistant_request=button.text
    )


async def _fix_location_card(card: LocationCard) -> None:
    url = card.open_map_url or ""
    if await _is_url_reachable(url):
        return
    logger.warning(f"url_validator_004: Removing broken map URL \033[36m{url}\033[0m")
    card.open_map_url = None


async def _fix_text_answer(text_answer: TextAnswer) -> None:
    new_text = text_answer.text
    for match in _MARKDOWN_LINK_PATTERN.finditer(text_answer.text):
        label, url = match.group(1), match.group(2)
        if await _is_url_reachable(url):
            continue
        replacement = await _find_replacement_url(label)
        if replacement:
            logger.warning(
                f"url_validator_005: Replacing broken text link \033[36m{url}\033[0m "
                f"with \033[36m{replacement}\033[0m"
            )
            new_text = new_text.replace(match.group(0), f"[{label}]({replacement})")
        else:
            logger.warning(f"url_validator_006: Stripping broken text link \033[36m{url}\033[0m")
            new_text = new_text.replace(match.group(0), label)
    text_answer.text = new_text
