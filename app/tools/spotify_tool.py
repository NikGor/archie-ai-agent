"""Spotify playback control and DJ-style thematic playlist tool."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from openai import OpenAI
from pydantic import BaseModel, Field
from app.backend.spotify_client import (
    SpotifyAuthError,
    SpotifyClient,
    SpotifyNoActiveDeviceError,
)
from app.backend.spotify_client import (
    get_spotify_client as _get_spotify_client,
)
from app.config import DEFAULT_MODEL, settings
from app.utils.arg_coercion import coerce_bool, coerce_int


logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=settings.openai_api_key)

MIN_PLAYLIST_QUERIES = 5
MAX_PLAYLIST_QUERIES = 15
MINUTES_PER_QUERY = 4
PLAYBACK_STATE_SETTLE_SECONDS = 0.5


class DjCommentResponse(BaseModel):
    """A short witty DJ-style comment for the current playback action."""

    comment: str = Field(description="One witty DJ line, 140 characters or fewer")


class PlaylistSeedQueries(BaseModel):
    """Search queries covering a playlist theme."""

    queries: list[str] = Field(
        description="5-15 Spotify search queries matching the theme"
    )


class PlaybackRequestClassification(BaseModel):
    """Classifies a free-form playback request as a specific track or a mood/theme."""

    kind: str = Field(
        description="'track' if the request names a specific song and/or artist, "
        "'theme' if it describes a mood, genre, activity, or is otherwise vague"
    )
    normalized_query: str = Field(
        description="For 'track': a concise 'artist - title' search string. "
        "For 'theme': a short theme description suitable for building a playlist."
    )


_FALLBACK_COMMENTS = {
    "play": "Now playing something good.",
    "pause": "Taking a break.",
    "next": "Next up.",
    "previous": "Back to the last one.",
}


def _track_id_from_uri(track_uri: str) -> str:
    """Extracts the bare track ID from a 'spotify:track:{id}' URI."""
    return track_uri.rsplit(":", 1)[-1]


def generate_dj_comment(action: str, context: dict[str, Any]) -> str:
    """Generates a short witty DJ comment for the given action using an LLM."""
    try:
        response = openai_client.beta.chat.completions.parse(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a witty radio DJ. Write one short, upbeat comment "
                    "(max 140 characters) about the current playback action.",
                },
                {
                    "role": "user",
                    "content": f"Action: {action}\nContext: {context}",
                },
            ],
            response_format=DjCommentResponse,
        )
        result = response.choices[0].message.parsed
        if result is None:
            raise ValueError("spotify_tool_error_001: No parsed DJ comment from LLM")
        return result.comment
    except Exception as e:
        logger.warning(
            f"spotify_tool_warn_001: DJ comment generation failed: \033[33m{e}\033[0m"
        )
        return _FALLBACK_COMMENTS.get(action, "Enjoy the music.")


def classify_playback_request(request: str) -> PlaybackRequestClassification:
    """Classifies a free-form music request as a specific track or a mood/theme, via LLM.

    Falls back to treating the request as a literal track search on LLM failure —
    the existing 'no track found' handling in the 'play' action already degrades
    that case to a themed playlist.
    """
    try:
        response = openai_client.beta.chat.completions.parse(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Classify the user's music request. kind='track' ONLY if it names "
                    "a specific song title (optionally with an artist) — then normalized_query is "
                    "a concise 'artist - title' search string. Otherwise kind='theme', including "
                    "requests that only name an artist/genre/mood/activity without a specific song "
                    "(e.g. 'something by Ozzy', 'play some jazz') — normalized_query is a short "
                    "theme description.",
                },
                {"role": "user", "content": request},
            ],
            response_format=PlaybackRequestClassification,
        )
        result = response.choices[0].message.parsed
        if result is None:
            raise ValueError(
                "spotify_tool_error_004: No parsed classification from LLM"
            )
        return result
    except Exception as e:
        logger.warning(
            f"spotify_tool_warn_005: Playback request classification failed: \033[33m{e}\033[0m"
        )
        return PlaybackRequestClassification(kind="track", normalized_query=request)


async def build_thematic_playlist(
    theme: str,
    duration_minutes: int,
    client: SpotifyClient,
    playlist_name: str | None = None,
) -> dict[str, Any]:
    """Builds a themed playlist by generating search queries via LLM and collecting tracks."""
    query_count = min(
        MAX_PLAYLIST_QUERIES,
        max(MIN_PLAYLIST_QUERIES, duration_minutes // MINUTES_PER_QUERY),
    )
    try:
        response = openai_client.beta.chat.completions.parse(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"Generate {query_count} diverse Spotify search queries "
                    "(track or artist names) that fit the given playlist theme. If the theme "
                    "is vague or broad (e.g. just an artist name, a generic mood, or otherwise "
                    "unspecific), prefer lesser-known, deeper-cut tracks over the most obvious "
                    "hits — hidden gems make for a more interesting playlist. If the theme is "
                    "already specific (particular era, sub-genre, activity, etc.), follow it "
                    "literally instead.",
                },
                {"role": "user", "content": f"Theme: {theme}"},
            ],
            response_format=PlaylistSeedQueries,
        )
        result = response.choices[0].message.parsed
        queries = result.queries if result else [theme]
    except Exception as e:
        logger.warning(
            f"spotify_tool_warn_002: Playlist query generation failed: \033[33m{e}\033[0m"
        )
        queries = [theme]

    seen_uris: set[str] = set()
    tracks: list[dict[str, Any]] = []
    total_seconds = 0
    target_seconds = duration_minutes * 60
    for query in queries:
        if total_seconds >= target_seconds:
            break
        found = await client.search_tracks(query, limit=2)
        for track in found:
            if track["uri"] in seen_uris:
                continue
            seen_uris.add(track["uri"])
            tracks.append(track)
            total_seconds += track["duration_seconds"]
            break

    name = playlist_name if playlist_name else f"Archie DJ: {theme}"
    playlist = await client.create_playlist(name=name)
    await client.add_tracks_to_playlist(
        playlist["playlist_id"], [t["uri"] for t in tracks]
    )

    # The user only expressed a mood/theme, not a track to play — so beyond saving
    # the playlist, queue every track for immediate playback ("open" the playlist)
    # instead of leaving the user to find and start it themselves.
    queued = False
    try:
        for track in tracks:
            await client.add_to_queue(track["uri"])
        queued = True
    except SpotifyNoActiveDeviceError:
        logger.warning(
            "spotify_tool_warn_004: \033[33mNo active Spotify device — playlist saved but not queued\033[0m"
        )

    return {
        "success": True,
        "action": "build_playlist",
        "playlist_id": playlist["playlist_id"],
        "playlist_name": playlist["playlist_name"],
        "playlist_url": playlist["playlist_url"],
        "tracks": tracks,
        "track_count": len(tracks),
        "queued": queued,
    }


_DEMO_TRACK = {
    "track_id": "demo_track_001",
    "uri": "spotify:track:demo_track_001",
    "title": "Demo Song",
    "artist": "Demo Artist",
    "album": "Demo Album",
    "duration_seconds": 210,
    "cover_url": None,
    "is_favorite": False,
}
_DEMO_ARTIST = {
    "artist_id": "demo_artist_001",
    "name": "Demo Artist",
    "genres": ["demo"],
    "popularity": 80,
    "image_url": None,
}
# Actions whose demo answer only says that they would run
_DEMO_EXECUTED = {
    "save_track",
    "remove_saved_track",
    "queue_add",
    "set_volume",
    "set_shuffle",
    "set_repeat",
    "seek",
}


def _demo_response(
    action: str, query: str | None, theme: str | None, playlist_name: str | None = None
) -> dict[str, Any]:
    """Returns canned demo data without calling the Spotify API."""
    track = dict(_DEMO_TRACK)
    if action in _DEMO_EXECUTED:
        return {
            "success": True,
            "action": action,
            "message": f"[DEMO] {action} would be executed",
        }
    if action in ("play", "pause", "next", "previous"):
        return {
            "success": True,
            "action": action,
            "message": f"[DEMO] Playback {action} would be executed",
            "current_track": track,
            "dj_comment": _FALLBACK_COMMENTS.get(action, "Enjoy the music."),
        }
    answers: dict[str, dict[str, Any]] = {
        "search": {"query": query, "tracks": [track]},
        "get_current": {
            "is_playing": True,
            "current_track": track,
            "progress_seconds": 42,
            "volume": 50,
            "shuffle": False,
            "repeat": "off",
        },
        "build_playlist": {
            "playlist_id": "demo_playlist_001",
            "playlist_name": playlist_name if playlist_name else f"Archie DJ: {theme}",
            "playlist_url": None,
            "tracks": [track],
            "track_count": 1,
            "queued": True,
        },
        "get_saved_tracks": {"tracks": [track]},
        "get_top_tracks": {"tracks": [track]},
        "get_top_artists": {"artists": [dict(_DEMO_ARTIST)]},
        "get_queue": {"current_track": track, "queue": [track]},
    }
    if action not in answers:
        return {"success": False, "message": f"Unknown action: {action}"}
    return {"success": True, "action": action, **answers[action]}


class SpotifyRequest(BaseModel):
    """Arguments of one spotify_tool call after coercion (ARCHIE-304)."""

    query: str | None = Field(default=None, description="Search or play request")
    track_uri: str | None = Field(default=None, description="Spotify URI")
    theme: str | None = Field(default=None, description="Playlist theme")
    playlist_name: str | None = Field(
        default=None, description="Explicit playlist name"
    )
    duration_minutes: int | None = Field(default=30, description="Playlist length")
    volume_percent: int | None = Field(default=None, description="Volume 0-100")
    shuffle: bool | None = Field(default=None, description="Shuffle on/off")
    repeat_mode: str | None = Field(default=None, description="track | context | off")
    position_ms: int | None = Field(default=None, description="Seek position")
    time_range: str = Field(default="medium_term", description="Top items range")


SpotifyHandler = Callable[[Any, str, SpotifyRequest], Awaitable[dict[str, Any]]]


async def _resolve_track_uri(
    client: Any, action: str, request: SpotifyRequest
) -> str | dict[str, Any]:
    """track_uri, or the first search hit for query; an error dict if neither works."""
    if request.track_uri:
        return request.track_uri
    if request.query:
        found = await client.search_tracks(request.query, limit=1)
        if not found:
            return {
                "success": False,
                "message": f"No tracks found for: {request.query}",
            }
        return found[0]["uri"]
    return {
        "success": False,
        "message": f"track_uri or query is required for '{action}'",
    }


async def _search(client: Any, _action: str, request: SpotifyRequest) -> dict[str, Any]:
    if not request.query:
        return {"success": False, "message": "query is required for 'search'"}
    tracks = await client.search_tracks(request.query)
    return {
        "success": True,
        "action": "search",
        "query": request.query,
        "tracks": tracks,
    }


async def _play(client: Any, _action: str, request: SpotifyRequest) -> dict[str, Any]:
    played_track = None
    if request.track_uri:
        await client.play(track_uris=[request.track_uri])
    elif request.query:
        classification = classify_playback_request(request.query)
        if classification.kind == "theme":
            playlist_result = await build_thematic_playlist(
                classification.normalized_query, request.duration_minutes, client
            )
            return {**playlist_result, "action": "play", "fallback": "theme"}
        found = await client.search_tracks(classification.normalized_query, limit=1)
        if not found:
            # Looked like a specific track but nothing matched — build a themed
            # playlist from the same request instead of just failing outright.
            playlist_result = await build_thematic_playlist(
                request.query, request.duration_minutes, client
            )
            return {**playlist_result, "action": "play", "fallback": "no_track_match"}
        played_track = found[0]
        await client.play(track_uris=[played_track["uri"]])
    else:
        await client.play()
    state = await client.get_playback_state() or {}
    if played_track:
        # Spotify's playback state endpoint is eventually consistent and can
        # still report the previous track right after a play command, so use
        # the track we just told it to play instead of trusting a stale read.
        state["current_track"] = played_track
        state["is_playing"] = True
    return {
        "success": True,
        "action": "play",
        **state,
        "dj_comment": generate_dj_comment("play", state),
    }


async def _transport(
    client: Any, action: str, _request: SpotifyRequest
) -> dict[str, Any]:
    method = {
        "pause": client.pause,
        "next": client.next_track,
        "previous": client.previous_track,
    }[action]
    await method()
    await asyncio.sleep(PLAYBACK_STATE_SETTLE_SECONDS)
    state = await client.get_playback_state()
    return {
        "success": True,
        "action": action,
        **(state or {}),
        "dj_comment": generate_dj_comment(action, state or {}),
    }


async def _get_current(
    client: Any, _action: str, _request: SpotifyRequest
) -> dict[str, Any]:
    playback = await client.get_playback_state()
    if playback is None:
        return {"success": True, "action": "get_current", "is_playing": False}
    return {"success": True, "action": "get_current", **playback}


async def _build_playlist(
    client: Any, _action: str, request: SpotifyRequest
) -> dict[str, Any]:
    if not request.theme:
        return {"success": False, "message": "theme is required for 'build_playlist'"}
    return await build_thematic_playlist(
        request.theme,
        request.duration_minutes,
        client,
        playlist_name=request.playlist_name,
    )


async def _saved_track(
    client: Any, action: str, request: SpotifyRequest
) -> dict[str, Any]:
    resolved = await _resolve_track_uri(client, action, request)
    if isinstance(resolved, dict):
        return resolved
    track_id = _track_id_from_uri(resolved)
    if action == "save_track":
        await client.save_tracks([track_id])
    else:
        await client.remove_saved_tracks([track_id])
    return {"success": True, "action": action, "track_uri": resolved}


async def _get_saved_tracks(
    client: Any, action: str, _request: SpotifyRequest
) -> dict[str, Any]:
    return {
        "success": True,
        "action": action,
        "tracks": await client.get_saved_tracks(),
    }


async def _top_items(
    client: Any, action: str, request: SpotifyRequest
) -> dict[str, Any]:
    if action == "get_top_tracks":
        items = await client.get_top_tracks(time_range=request.time_range)
        return {"success": True, "action": action, "tracks": items}
    artists = await client.get_top_artists(time_range=request.time_range)
    return {"success": True, "action": action, "artists": artists}


async def _queue_add(
    client: Any, action: str, request: SpotifyRequest
) -> dict[str, Any]:
    resolved = await _resolve_track_uri(client, action, request)
    if isinstance(resolved, dict):
        return resolved
    await client.add_to_queue(resolved)
    return {"success": True, "action": action, "track_uri": resolved}


async def _get_queue(
    client: Any, action: str, _request: SpotifyRequest
) -> dict[str, Any]:
    return {"success": True, "action": action, **(await client.get_queue())}


async def _set_volume(
    client: Any, action: str, request: SpotifyRequest
) -> dict[str, Any]:
    if request.volume_percent is None:
        return {
            "success": False,
            "message": "volume_percent is required for 'set_volume'",
        }
    await client.set_volume(request.volume_percent)
    return {"success": True, "action": action, "volume": request.volume_percent}


async def _set_shuffle(
    client: Any, action: str, request: SpotifyRequest
) -> dict[str, Any]:
    if request.shuffle is None:
        return {"success": False, "message": "shuffle is required for 'set_shuffle'"}
    await client.set_shuffle(request.shuffle)
    return {"success": True, "action": action, "shuffle": request.shuffle}


async def _set_repeat(
    client: Any, action: str, request: SpotifyRequest
) -> dict[str, Any]:
    if request.repeat_mode not in ("track", "context", "off"):
        return {
            "success": False,
            "message": "repeat_mode must be one of 'track', 'context', 'off'",
        }
    await client.set_repeat(request.repeat_mode)
    return {"success": True, "action": action, "repeat": request.repeat_mode}


async def _seek(client: Any, action: str, request: SpotifyRequest) -> dict[str, Any]:
    if request.position_ms is None:
        return {"success": False, "message": "position_ms is required for 'seek'"}
    await client.seek(request.position_ms)
    return {
        "success": True,
        "action": action,
        "progress_seconds": request.position_ms // 1000,
    }


_ACTIONS: dict[str, SpotifyHandler] = {
    "search": _search,
    "play": _play,
    "pause": _transport,
    "next": _transport,
    "previous": _transport,
    "get_current": _get_current,
    "build_playlist": _build_playlist,
    "save_track": _saved_track,
    "remove_saved_track": _saved_track,
    "get_saved_tracks": _get_saved_tracks,
    "get_top_tracks": _top_items,
    "get_top_artists": _top_items,
    "queue_add": _queue_add,
    "get_queue": _get_queue,
    "set_volume": _set_volume,
    "set_shuffle": _set_shuffle,
    "set_repeat": _set_repeat,
    "seek": _seek,
}


async def spotify_tool(
    action: str,
    query: str | None = None,
    track_uri: str | None = None,
    theme: str | None = None,
    playlist_name: str | None = None,
    duration_minutes: int | str = 30,
    volume_percent: int | str | None = None,
    shuffle: bool | str | None = None,
    repeat_mode: str | None = None,
    position_ms: int | str | None = None,
    time_range: str = "medium_term",
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Control Spotify playback, manage the library/queue, and build thematic playlists.

    Args:
        action: One of: search, play, pause, next, previous, get_current, build_playlist,
            save_track, remove_saved_track, get_saved_tracks, get_top_tracks, get_top_artists,
            queue_add, get_queue, set_volume, set_shuffle, set_repeat, seek
        query: Search query for 'search' (returns a list, no playback). For 'play', pass any
            free-form request here instead of a precise title — a specific song/artist
            ("Believer by Imagine Dragons") is searched and played directly; a mood/genre/activity
            request ("что-нибудь для вечерней пробежки") is turned into a themed playlist that gets
            saved and queued for immediate playback. If a specific-sounding request finds no match,
            it also falls back to a themed playlist instead of failing. Also used to pick a track
            for 'queue_add' if track_uri is absent.
        track_uri: Spotify track/context URI (optional for 'play' - omit to resume current context;
            required for 'save_track'/'remove_saved_track' if query is absent)
        theme: Playlist theme/mood description (required for 'build_playlist', e.g. "chill Sunday morning jazz")
        playlist_name: Explicit playlist title for 'build_playlist', if the user gave one
            (e.g. "Christmas Jazz & Classics"). Used as-is for the Spotify playlist name instead
            of the "Archie DJ: {theme}" default — keep it short, it becomes the literal name field.
        duration_minutes: Target playlist length in minutes for 'play' (theme fallback) and 'build_playlist' (default 30)
        volume_percent: Volume level 0-100 (required for 'set_volume')
        shuffle: True/False to enable/disable shuffle (required for 'set_shuffle')
        repeat_mode: One of 'track', 'context', 'off' (required for 'set_repeat')
        position_ms: Position in milliseconds to seek to (required for 'seek')
        time_range: One of 'short_term', 'medium_term', 'long_term' (for 'get_top_tracks'/'get_top_artists')

    Returns:
        Dict with playback/search/playlist/library data or error information
    """
    duration_minutes = coerce_int(duration_minutes)
    volume_percent = coerce_int(volume_percent)
    shuffle = coerce_bool(shuffle)
    position_ms = coerce_int(position_ms)

    logger.info(
        f"spotify_tool_001: Action requested: \033[36m{action}\033[0m, "
        f"demo_mode: \033[35m{demo_mode}\033[0m"
    )

    if demo_mode:
        return _demo_response(action, query, theme, playlist_name)

    request = SpotifyRequest(
        query=query,
        track_uri=track_uri,
        theme=theme,
        playlist_name=playlist_name,
        duration_minutes=duration_minutes,
        volume_percent=volume_percent,
        shuffle=shuffle,
        repeat_mode=repeat_mode,
        position_ms=position_ms,
        time_range=time_range,
    )
    try:
        handler = _ACTIONS.get(action)
        if handler is None:
            return {"success": False, "message": f"Unknown action: {action}"}
        return await handler(_get_spotify_client(), action, request)

    except SpotifyNoActiveDeviceError:
        logger.warning("spotify_tool_warn_003: \033[33mNo active Spotify device\033[0m")
        return {
            "success": False,
            "message": "No active Spotify device found. Open Spotify on a device and try again.",
        }
    except SpotifyAuthError as e:
        logger.error(f"spotify_tool_error_002: \033[31mAuth error: {e}\033[0m")
        return {"success": False, "message": f"Spotify authentication error: {e}"}
    except Exception as e:
        logger.error(f"spotify_tool_error_003: \033[31m{e!s}\033[0m")
        return {"success": False, "message": f"Spotify request failed: {e!s}"}
