"""Android TV tool for IPTV channels, VLC streams, and YouTube links."""

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from app.backend.android_tv_client import AndroidTVClient, AndroidTVError
from app.config import settings


logger = logging.getLogger(__name__)

_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_SUPPORTED_PLAYERS = {"auto", "vlc", "youtube"}
_REMOTE_KEYS = {
    "volume_up": "KEYCODE_VOLUME_UP",
    "volume_down": "KEYCODE_VOLUME_DOWN",
    "mute": "KEYCODE_VOLUME_MUTE",
}
_SUPPORTED_ACTIONS = {
    "list_channels",
    "play_channel",
    "play_url",
    "play_youtube",
    *_REMOTE_KEYS,
}


@dataclass(frozen=True)
class TVChannel:
    """One launchable entry from an M3U playlist."""

    title: str
    url: str
    group: str | None = None


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def _load_channels(playlist_path: str, token: str | None) -> list[TVChannel]:
    path = Path(playlist_path).expanduser()
    if not path.is_file():
        raise ValueError(f"TV playlist not found: {path}")

    channels: list[TVChannel] = []
    title: str | None = None
    group: str | None = None
    for raw_line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#EXTINF"):
            title = line.rsplit(",", 1)[-1].strip()
            group = None
            continue
        if line.startswith("#EXTGRP:"):
            group = line.partition(":")[2].strip() or None
            continue
        if line.startswith("#") or title is None:
            continue

        if "YOUR_TOKEN" in line:
            if not token:
                raise ValueError(
                    "SHARAVOZ_TOKEN is required by the configured playlist"
                )
            line = line.replace("YOUR_TOKEN", token)
        channels.append(TVChannel(title=title, url=line, group=group))
        title = None
        group = None

    if not channels:
        raise ValueError(f"TV playlist has no channels: {path}")
    return channels


def _channel_matches(
    channels: list[TVChannel], query: str
) -> list[tuple[float, TVChannel]]:
    normalized_query = _normalize(query)
    if not normalized_query:
        return []

    scored: list[tuple[float, TVChannel]] = []
    for channel in channels:
        normalized_title = _normalize(channel.title)
        if normalized_title == normalized_query:
            score = 1.0
        elif normalized_query in normalized_title:
            score = 0.9 - min(len(normalized_title) - len(normalized_query), 50) / 500
        else:
            score = SequenceMatcher(None, normalized_query, normalized_title).ratio()
        scored.append((score, channel))
    return sorted(scored, key=lambda item: (-item[0], item[1].title.casefold()))


def _find_channel(
    channels: list[TVChannel], query: str
) -> tuple[TVChannel | None, list[str]]:
    matches = _channel_matches(channels, query)
    if not matches:
        return None, []
    best_score, best_channel = matches[0]
    suggestions = [channel.title for _, channel in matches[:5]]
    if best_score < 0.58:
        return None, suggestions
    return best_channel, suggestions


def _extract_youtube_id(value: str) -> str | None:
    value = value.strip()
    if _YOUTUBE_ID_RE.fullmatch(value):
        return value
    parsed = urlparse(value)
    host = parsed.hostname.casefold() if parsed.hostname else ""
    if host in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/").split("/", 1)[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            candidate = parsed.path.strip("/").split("/", 1)[1]
        else:
            candidate = ""
    else:
        candidate = ""
    return candidate if _YOUTUBE_ID_RE.fullmatch(candidate) else None


def _youtube_link(value: str) -> tuple[str, bool]:
    video_id = _extract_youtube_id(value)
    if video_id:
        return f"vnd.youtube:{video_id}", True
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        host = parsed.hostname.casefold() if parsed.hostname else ""
        if host not in {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtu.be",
            "www.youtu.be",
        }:
            raise ValueError("URL is not a YouTube link")
        raise ValueError("YouTube URL does not identify a video")
    raise ValueError(
        "A direct YouTube video URL or 11-character video ID is required. "
        "If only a title or description is available, search the internet first "
        "and pass the resulting video URL."
    )


def _vlc_link(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https", "rtsp", "rtmp", "udp", "ftp"}:
        raise ValueError(f"Unsupported VLC URL scheme: {parsed.scheme or 'missing'}")
    # Android TV Remote v2 can launch a URI but cannot choose an Android package.
    # The TV must have VLC saved as the default handler for direct media URLs.
    return url


def _is_youtube_url(value: str) -> bool:
    parsed = urlparse(value)
    host = parsed.hostname.casefold() if parsed.hostname else ""
    return host in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }


def _get_client() -> AndroidTVClient:
    return AndroidTVClient(
        host=settings.tv_host,
        cert_file=settings.tv_cert_file,
        key_file=settings.tv_key_file,
        client_name=settings.tv_remote_name,
        connect_timeout=settings.tv_connect_timeout,
    )


async def tv_tool(  # noqa: PLR0911, PLR0912
    action: str,
    query: str | None = None,
    url: str | None = None,
    player: str = "auto",
    limit: int = 20,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Browse IPTV channels and launch media on the configured Android TV.
    Use this tool for requests such as "show football channels", "play Match Football 2",
    "open this stream in VLC", or "play this YouTube video on the TV".
    To play a YouTube video named only by title or description, first search the internet
    for its direct YouTube video URL, then call this tool with that URL or its video ID.
    It never returns private stream URLs or the Sharavoz token.

    Args:
        action: One of: list_channels, play_channel, play_url, play_youtube,
            volume_up, volume_down, mute. The mute action toggles mute/unmute,
            matching the mute button on a physical remote.
        query: Channel name for play_channel; optional channel filter for list_channels;
            direct YouTube video URL or 11-character video ID for play_youtube. If the
            video is specified by title or description, search the internet first.
        url: Direct media URL for play_url. For play_youtube, query is preferred but url is accepted.
        player: Playback app: auto, vlc, or youtube. Direct media streams and channels
            use VLC; YouTube URLs/IDs use YouTube. The TV must have VLC saved as the
            default handler for direct media URLs.
        limit: Maximum channels returned by list_channels, from 1 to 100

    Returns:
        dict[str, Any]: Launch status or channel names, without private media URLs
    """
    action = action.strip().casefold()
    player = player.strip().casefold()
    if action not in _SUPPORTED_ACTIONS:
        return {"status": "error", "message": f"Unknown TV action: {action}"}
    if player not in _SUPPORTED_PLAYERS:
        return {"status": "error", "message": f"Unknown TV player: {player}"}

    if demo_mode:
        return {
            "status": "demo",
            "action": action,
            "message": f"[DEMO] TV action {action} would be executed",
        }

    try:
        if action in _REMOTE_KEYS:
            await _get_client().send_key(_REMOTE_KEYS[action])
            return {
                "status": "success",
                "action": action,
                "message": f"TV remote command sent: {action}",
            }

        if action in {"list_channels", "play_channel"}:
            channels = _load_channels(
                settings.tv_playlist_path, settings.sharavoz_token
            )
            if action == "list_channels":
                limit = max(1, min(limit, 100))
                if query:
                    matches = _channel_matches(channels, query)
                    selected = [channel for score, channel in matches if score >= 0.35][
                        :limit
                    ]
                else:
                    selected = channels[:limit]
                return {
                    "status": "success",
                    "action": action,
                    "channels": [
                        {"title": channel.title, "group": channel.group}
                        for channel in selected
                    ],
                    "count": len(selected),
                }

            if not query:
                return {
                    "status": "error",
                    "message": "query is required for play_channel",
                }
            if player == "youtube":
                return {
                    "status": "error",
                    "message": "IPTV channels can only be played with VLC",
                }
            channel, suggestions = _find_channel(channels, query)
            if channel is None:
                return {
                    "status": "not_found",
                    "message": f"Channel not found: {query}",
                    "suggestions": suggestions,
                }
            launch_link = _vlc_link(channel.url)
            await _get_client().launch_link(launch_link)
            return {
                "status": "success",
                "action": action,
                "channel": channel.title,
                "group": channel.group,
                "player": "vlc" if player == "auto" else player,
                "message": f"Playing {channel.title} on Android TV",
            }

        if action == "play_url":
            media_url = url or query
            if not media_url:
                return {"status": "error", "message": "url is required for play_url"}
            is_youtube = bool(_extract_youtube_id(media_url)) or _is_youtube_url(
                media_url
            )
            if player == "youtube" or (player == "auto" and is_youtube):
                launch_link, direct = _youtube_link(media_url)
                resolved_player = "youtube"
            elif player in {"auto", "vlc"}:
                if is_youtube:
                    return {
                        "status": "error",
                        "message": (
                            "This TV can route YouTube links only to the YouTube app; "
                            "use player=youtube"
                        ),
                    }
                launch_link = _vlc_link(media_url)
                direct = True
                resolved_player = "vlc"
            await _get_client().launch_link(launch_link)
            return {
                "status": "success",
                "action": action,
                "player": resolved_player,
                "direct_play": direct,
                "message": f"Media opened with {resolved_player} on Android TV",
            }

        youtube_value = query or url
        if not youtube_value:
            return {"status": "error", "message": "query is required for play_youtube"}
        if player == "vlc":
            return {
                "status": "error",
                "message": (
                    "This TV can route YouTube links only to the YouTube app; "
                    "VLC is supported for direct media streams"
                ),
            }
        launch_link, direct = _youtube_link(youtube_value)
        resolved_player = "youtube"
        await _get_client().launch_link(launch_link)
        return {
            "status": "success",
            "action": action,
            "player": resolved_player,
            "direct_play": direct,
            "message": "YouTube video opened on Android TV",
        }
    except (ValueError, AndroidTVError) as exc:
        logger.warning("tv_tool_error: %s", exc)
        return {"status": "error", "message": str(exc)}
