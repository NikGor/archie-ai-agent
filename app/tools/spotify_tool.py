"""Spotify playback control and DJ-style thematic playlist tool."""

import asyncio
import logging
from typing import Any
from openai import OpenAI
from pydantic import BaseModel, Field
from app.backend.spotify_client import (
    SpotifyAuthError,
    SpotifyClient,
    SpotifyNoActiveDeviceError,
    get_spotify_client as _get_spotify_client,
)
from app.config import settings


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

    queries: list[str] = Field(description="5-15 Spotify search queries matching the theme")


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
            model="gpt-4.1-nano",
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
        logger.warning(f"spotify_tool_warn_001: DJ comment generation failed: \033[33m{e}\033[0m")
        return _FALLBACK_COMMENTS.get(action, "Enjoy the music.")


async def build_thematic_playlist(
    theme: str, duration_minutes: int, client: SpotifyClient
) -> dict[str, Any]:
    """Builds a themed playlist by generating search queries via LLM and collecting tracks."""
    query_count = min(
        MAX_PLAYLIST_QUERIES,
        max(MIN_PLAYLIST_QUERIES, duration_minutes // MINUTES_PER_QUERY),
    )
    try:
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": f"Generate {query_count} diverse Spotify search queries "
                    "(track or artist names) that fit the given playlist theme.",
                },
                {"role": "user", "content": f"Theme: {theme}"},
            ],
            response_format=PlaylistSeedQueries,
        )
        result = response.choices[0].message.parsed
        queries = result.queries if result else [theme]
    except Exception as e:
        logger.warning(f"spotify_tool_warn_002: Playlist query generation failed: \033[33m{e}\033[0m")
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

    user_id = await client.get_current_user_id()
    playlist = await client.create_playlist(user_id, name=f"Archie DJ: {theme}")
    await client.add_tracks_to_playlist(playlist["playlist_id"], [t["uri"] for t in tracks])

    return {
        "success": True,
        "action": "build_playlist",
        "playlist_id": playlist["playlist_id"],
        "playlist_name": playlist["playlist_name"],
        "playlist_url": playlist["playlist_url"],
        "tracks": tracks,
        "track_count": len(tracks),
    }


def _demo_response(action: str, query: str | None, theme: str | None) -> dict[str, Any]:
    """Returns canned demo data without calling the Spotify API."""
    demo_track = {
        "track_id": "demo_track_001",
        "uri": "spotify:track:demo_track_001",
        "title": "Demo Song",
        "artist": "Demo Artist",
        "album": "Demo Album",
        "duration_seconds": 210,
        "cover_url": None,
        "is_favorite": False,
    }
    if action == "search":
        return {
            "success": True,
            "action": "search",
            "query": query,
            "tracks": [demo_track],
        }
    if action == "get_current":
        return {
            "success": True,
            "action": "get_current",
            "is_playing": True,
            "current_track": demo_track,
            "progress_seconds": 42,
            "volume": 50,
            "shuffle": False,
            "repeat": "off",
        }
    if action == "build_playlist":
        return {
            "success": True,
            "action": "build_playlist",
            "playlist_id": "demo_playlist_001",
            "playlist_name": f"Archie DJ: {theme}",
            "playlist_url": None,
            "tracks": [demo_track],
            "track_count": 1,
        }
    if action in ("play", "pause", "next", "previous"):
        return {
            "success": True,
            "action": action,
            "message": f"[DEMO] Playback {action} would be executed",
            "current_track": demo_track,
            "dj_comment": _FALLBACK_COMMENTS.get(action, "Enjoy the music."),
        }
    if action in ("save_track", "remove_saved_track", "queue_add"):
        return {"success": True, "action": action, "message": f"[DEMO] {action} would be executed"}
    if action == "get_saved_tracks":
        return {"success": True, "action": action, "tracks": [demo_track]}
    if action in ("get_top_tracks",):
        return {"success": True, "action": action, "tracks": [demo_track]}
    if action == "get_top_artists":
        return {
            "success": True,
            "action": action,
            "artists": [
                {
                    "artist_id": "demo_artist_001",
                    "name": "Demo Artist",
                    "genres": ["demo"],
                    "popularity": 80,
                    "image_url": None,
                }
            ],
        }
    if action == "get_queue":
        return {"success": True, "action": action, "current_track": demo_track, "queue": [demo_track]}
    if action in ("set_volume", "set_shuffle", "set_repeat", "seek"):
        return {"success": True, "action": action, "message": f"[DEMO] {action} would be executed"}
    return {"success": False, "message": f"Unknown action: {action}"}


async def spotify_tool(  # noqa: PLR0911, PLR0912
    action: str,
    query: str | None = None,
    track_uri: str | None = None,
    theme: str | None = None,
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
        query: Search query (required for 'search'; used to pick a track for 'play'/'queue_add' if track_uri is absent)
        track_uri: Spotify track/context URI (optional for 'play' - omit to resume current context;
            required for 'save_track'/'remove_saved_track' if query is absent)
        theme: Playlist theme/mood description (required for 'build_playlist', e.g. "chill Sunday morning jazz")
        duration_minutes: Target playlist length in minutes for 'build_playlist' (default 30)
        volume_percent: Volume level 0-100 (required for 'set_volume')
        shuffle: True/False to enable/disable shuffle (required for 'set_shuffle')
        repeat_mode: One of 'track', 'context', 'off' (required for 'set_repeat')
        position_ms: Position in milliseconds to seek to (required for 'seek')
        time_range: One of 'short_term', 'medium_term', 'long_term' (for 'get_top_tracks'/'get_top_artists')

    Returns:
        Dict with playback/search/playlist/library data or error information
    """
    if isinstance(duration_minutes, str):
        duration_minutes = int(duration_minutes)
    if isinstance(volume_percent, str):
        volume_percent = int(volume_percent)
    if isinstance(shuffle, str):
        shuffle = shuffle.lower() == "true"
    if isinstance(position_ms, str):
        position_ms = int(position_ms)

    logger.info(
        f"spotify_tool_001: Action requested: \033[36m{action}\033[0m, "
        f"demo_mode: \033[35m{demo_mode}\033[0m"
    )

    if demo_mode:
        return _demo_response(action, query, theme)

    try:
        client = _get_spotify_client()

        if action == "search":
            if not query:
                return {"success": False, "message": "query is required for 'search'"}
            tracks = await client.search_tracks(query)
            return {"success": True, "action": "search", "query": query, "tracks": tracks}

        if action == "play":
            played_track = None
            if track_uri:
                await client.play(track_uris=[track_uri])
            elif query:
                found = await client.search_tracks(query, limit=1)
                if not found:
                    return {"success": False, "message": f"No tracks found for: {query}"}
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

        if action in ("pause", "next", "previous"):
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

        if action == "get_current":
            state = await client.get_playback_state()
            if state is None:
                return {"success": True, "action": "get_current", "is_playing": False}
            return {"success": True, "action": "get_current", **state}

        if action == "build_playlist":
            if not theme:
                return {"success": False, "message": "theme is required for 'build_playlist'"}
            return await build_thematic_playlist(theme, duration_minutes, client)

        if action in ("save_track", "remove_saved_track"):
            resolved_uri: str
            if track_uri:
                resolved_uri = track_uri
            elif query:
                found = await client.search_tracks(query, limit=1)
                if not found:
                    return {"success": False, "message": f"No tracks found for: {query}"}
                resolved_uri = found[0]["uri"]
            else:
                return {"success": False, "message": f"track_uri or query is required for '{action}'"}
            track_id = _track_id_from_uri(resolved_uri)
            if action == "save_track":
                await client.save_tracks([track_id])
            else:
                await client.remove_saved_tracks([track_id])
            return {"success": True, "action": action, "track_uri": resolved_uri}

        if action == "get_saved_tracks":
            tracks = await client.get_saved_tracks()
            return {"success": True, "action": action, "tracks": tracks}

        if action in ("get_top_tracks", "get_top_artists"):
            if action == "get_top_tracks":
                items = await client.get_top_tracks(time_range=time_range)
                return {"success": True, "action": action, "tracks": items}
            artists = await client.get_top_artists(time_range=time_range)
            return {"success": True, "action": action, "artists": artists}

        if action == "queue_add":
            queue_uri: str
            if track_uri:
                queue_uri = track_uri
            elif query:
                found = await client.search_tracks(query, limit=1)
                if not found:
                    return {"success": False, "message": f"No tracks found for: {query}"}
                queue_uri = found[0]["uri"]
            else:
                return {"success": False, "message": "track_uri or query is required for 'queue_add'"}
            await client.add_to_queue(queue_uri)
            return {"success": True, "action": action, "track_uri": queue_uri}

        if action == "get_queue":
            queue_state = await client.get_queue()
            return {"success": True, "action": action, **queue_state}

        if action == "set_volume":
            if volume_percent is None:
                return {"success": False, "message": "volume_percent is required for 'set_volume'"}
            await client.set_volume(volume_percent)
            return {"success": True, "action": action, "volume": volume_percent}

        if action == "set_shuffle":
            if shuffle is None:
                return {"success": False, "message": "shuffle is required for 'set_shuffle'"}
            await client.set_shuffle(shuffle)
            return {"success": True, "action": action, "shuffle": shuffle}

        if action == "set_repeat":
            if repeat_mode not in ("track", "context", "off"):
                return {
                    "success": False,
                    "message": "repeat_mode must be one of 'track', 'context', 'off'",
                }
            await client.set_repeat(repeat_mode)
            return {"success": True, "action": action, "repeat": repeat_mode}

        if action == "seek":
            if position_ms is None:
                return {"success": False, "message": "position_ms is required for 'seek'"}
            await client.seek(position_ms)
            return {"success": True, "action": action, "progress_seconds": position_ms // 1000}

        return {"success": False, "message": f"Unknown action: {action}"}

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
