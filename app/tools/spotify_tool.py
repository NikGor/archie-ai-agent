"""
Spotify DJ Agent Tool

=== IMPLEMENTATION TODO ===

PHASE 1: Frontend (do first - backend needs device_id from frontend)
─────────────────────────────────────────────────────────────────────
TODO [F1]: Create Spotify App at https://developer.spotify.com/dashboard
    - Get CLIENT_ID and CLIENT_SECRET
    - Set redirect URI (e.g., http://localhost:3000/callback)
    - Enable Web Playback SDK in app settings

TODO [F2]: Implement OAuth flow on frontend
    - Redirect to Spotify /authorize with scopes:
      user-read-playback-state, user-modify-playback-state,
      playlist-read-private, playlist-modify-public, playlist-modify-private
    - Handle callback, exchange code for access_token
    - Store tokens (localStorage or send to backend for refresh)

TODO [F3]: Initialize Web Playback SDK
    - Load SDK script: https://sdk.scdn.co/spotify-player.js
    - Create Spotify.Player instance with access_token
    - Listen for 'ready' event → save device_id
    - Pass device_id to archie requests

TODO [F4]: Build player UI component
    - Play/Pause button
    - Next/Previous buttons
    - Progress bar + current track info
    - Volume control


PHASE 2: Backend - Spotify Client
─────────────────────────────────────────────────────────────────────
TODO [B1]: Create spotify_client.py
    - httpx async client for Spotify Web API
    - Base URL: https://api.spotify.com/v1
    - Auth header: Bearer {access_token}
    - Methods: search(), play(), pause(), next(), previous(),
      get_playback_state(), create_playlist(), add_to_playlist()

TODO [B2]: Handle token from frontend
    - Receive access_token + device_id in request context
    - Option: implement token refresh on backend (needs CLIENT_SECRET)


PHASE 3: Backend - DJ Agent (OpenAI Agent SDK)
─────────────────────────────────────────────────────────────────────
TODO [B3]: Create dj_agent.py with OpenAI Agent SDK
    - System prompt: DJ personality, music knowledge
    - Tools: search_tracks, play_track, pause, next, previous,
      create_playlist, add_to_playlist, get_recommendations

TODO [B4]: Advanced DJ features
    - generate_dj_comment: LLM generates witty transition text
    - build_thematic_playlist: "2-hour party mix" → search + create
    - smooth_transition: gradual genre shift between tracks


PHASE 4: Integration
─────────────────────────────────────────────────────────────────────
TODO [B5]: Connect spotify_tool to DJ Agent
    - spotify_tool receives user request + context (device_id, token)
    - Passes to DJ Agent for multi-step reasoning
    - Returns structured response with track URIs for frontend

TODO [F5]: Frontend handles agent response
    - If response contains track_uri → call player.play({uris: [uri]})
    - If response contains playlist → show in UI
    - Display DJ comments as chat messages

===============================================================
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def spotify_tool(
    action: str,
    query: str | None = None,
    track_id: str | None = None,
) -> dict[str, Any]:
    """
    Spotify DJ Agent entry point.
    Currently a stub - will be replaced with DJ Agent in Phase 3.

    Args:
        action: Action to perform (search, play, pause, next, previous, get_current)
        query: Search query for finding music (required for 'search' action)
        track_id: Spotify track ID (required for 'play' action)

    Returns:
        Dict with action result or error information
    """
    # TODO [B5]: Replace this stub with DJ Agent call
    logger.info(f"spotify_001: Action requested: \033[36m{action}\033[0m")

    result = {"success": True, "action": action}

    if action == "search":
        result["message"] = f"Music search completed for: {query}"
        result["query"] = query
    elif action == "play":
        result["message"] = f"Now playing track: {track_id}"
        result["track_id"] = track_id
    elif action == "pause":
        result["message"] = "Playback paused successfully"
    elif action == "next":
        result["message"] = "Skipped to next track"
    elif action == "previous":
        result["message"] = "Returned to previous track"
    elif action == "get_current":
        result["message"] = "Current playback info retrieved successfully"
    else:
        result["message"] = f"Action '{action}' executed successfully"

    return result
