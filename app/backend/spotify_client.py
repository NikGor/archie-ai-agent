"""Spotify Web API client with automatic access token refresh."""

import datetime
import logging
from typing import Any
import httpx
from app.config import settings


logger = logging.getLogger(__name__)

TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE_URL = "https://api.spotify.com/v1"
TOKEN_EXPIRY_BUFFER_SECONDS = 60


class SpotifyAuthError(Exception):
    """Raised when Spotify credentials are missing or the token refresh fails."""


class SpotifyNoActiveDeviceError(Exception):
    """Raised when a playback command has no active Spotify device to target."""


class SpotifyClient:
    """Wrapper for Spotify Web API operations, backed by a refresh-token grant."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        refresh_token: str | None = None,
    ) -> None:
        self.client_id = client_id or settings.spotify_client_id
        self.client_secret = client_secret or settings.spotify_client_secret
        self.refresh_token = refresh_token or settings.spotify_refresh_token
        self._access_token: str | None = None
        self._token_expiry: datetime.datetime | None = None
        self._user_id: str | None = None

    async def _get_access_token(self) -> str:
        """Returns a cached access token, refreshing it if missing or expired."""
        now = datetime.datetime.now(datetime.UTC)
        if self._access_token and self._token_expiry and now < self._token_expiry:
            return self._access_token

        if not (self.client_id and self.client_secret and self.refresh_token):
            logger.error("spotify_client_error_001: \033[31mMissing Spotify credentials\033[0m")
            raise SpotifyAuthError("Spotify client_id/client_secret/refresh_token not configured")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    TOKEN_URL,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self.refresh_token,
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                    timeout=30.0,
                )
        except httpx.HTTPError as e:
            logger.error(f"spotify_client_error_002: \033[31mToken refresh request failed: {e}\033[0m")
            raise SpotifyAuthError(f"Token refresh request failed: {e}") from e

        if response.status_code != 200:
            logger.error(
                f"spotify_client_error_003: \033[31mToken refresh returned {response.status_code}\033[0m"
            )
            raise SpotifyAuthError(f"Token refresh failed: {response.status_code} {response.text}")

        data = response.json()
        access_token: str = data["access_token"]
        self._access_token = access_token
        expires_in = data.get("expires_in", 3600)
        self._token_expiry = now + datetime.timedelta(
            seconds=expires_in - TOKEN_EXPIRY_BUFFER_SECONDS
        )
        logger.info("spotify_client_001: Refreshed access token")
        return access_token

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Sends an authenticated request to the Spotify Web API."""
        token = await self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(base_url=API_BASE_URL) as client:
            response = await client.request(method, path, headers=headers, timeout=30.0, **kwargs)

        if response.status_code == 404:
            body = response.json() if response.content else {}
            if body.get("error", {}).get("reason") == "NO_ACTIVE_DEVICE":
                raise SpotifyNoActiveDeviceError("No active Spotify device found")

        return response

    async def search_tracks(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Searches Spotify for tracks matching the query."""
        response = await self._request(
            "GET", "/search", params={"q": query, "type": "track", "limit": limit}
        )
        response.raise_for_status()
        items = response.json().get("tracks", {}).get("items", [])
        return [_track_to_dict(item) for item in items]

    async def get_playback_state(self) -> dict[str, Any] | None:
        """Returns the current playback state, or None if nothing is playing."""
        response = await self._request("GET", "/me/player")
        if response.status_code == 204 or not response.content:
            return None
        response.raise_for_status()
        data = response.json()
        track = data.get("item")
        return {
            "is_playing": data.get("is_playing", False),
            "progress_seconds": (data.get("progress_ms") or 0) // 1000,
            "volume": (data.get("device") or {}).get("volume_percent", 50),
            "shuffle": data.get("shuffle_state", False),
            "repeat": data.get("repeat_state", "off"),
            "current_track": _track_to_dict(track) if track else None,
        }

    async def get_devices(self) -> list[dict[str, Any]]:
        """Returns the list of available Spotify devices."""
        response = await self._request("GET", "/me/player/devices")
        response.raise_for_status()
        return response.json().get("devices", [])

    async def play(
        self,
        track_uris: list[str] | None = None,
        context_uri: str | None = None,
    ) -> None:
        """Starts or resumes playback, optionally for the given track(s) or context."""
        body: dict[str, Any] = {}
        if track_uris:
            body["uris"] = track_uris
        if context_uri:
            body["context_uri"] = context_uri
        response = await self._request("PUT", "/me/player/play", json=body or None)
        if response.status_code not in (200, 202, 204):
            response.raise_for_status()

    async def pause(self) -> None:
        """Pauses playback."""
        response = await self._request("PUT", "/me/player/pause")
        if response.status_code not in (200, 202, 204):
            response.raise_for_status()

    async def next_track(self) -> None:
        """Skips to the next track."""
        response = await self._request("POST", "/me/player/next")
        if response.status_code not in (200, 202, 204):
            response.raise_for_status()

    async def previous_track(self) -> None:
        """Returns to the previous track."""
        response = await self._request("POST", "/me/player/previous")
        if response.status_code not in (200, 202, 204):
            response.raise_for_status()

    async def set_volume(self, volume_percent: int) -> None:
        """Sets playback volume (0-100)."""
        response = await self._request(
            "PUT", "/me/player/volume", params={"volume_percent": volume_percent}
        )
        if response.status_code not in (200, 202, 204):
            response.raise_for_status()

    async def set_shuffle(self, state: bool) -> None:
        """Toggles shuffle mode."""
        response = await self._request(
            "PUT", "/me/player/shuffle", params={"state": str(state).lower()}
        )
        if response.status_code not in (200, 202, 204):
            response.raise_for_status()

    async def set_repeat(self, state: str) -> None:
        """Sets repeat mode: 'track', 'context', or 'off'."""
        response = await self._request("PUT", "/me/player/repeat", params={"state": state})
        if response.status_code not in (200, 202, 204):
            response.raise_for_status()

    async def seek(self, position_ms: int) -> None:
        """Seeks to the given position (in milliseconds) in the current track."""
        response = await self._request(
            "PUT", "/me/player/seek", params={"position_ms": position_ms}
        )
        if response.status_code not in (200, 202, 204):
            response.raise_for_status()

    async def add_to_queue(self, track_uri: str) -> None:
        """Adds a track to the playback queue."""
        response = await self._request(
            "POST", "/me/player/queue", params={"uri": track_uri}
        )
        if response.status_code not in (200, 202, 204):
            response.raise_for_status()

    async def get_queue(self) -> dict[str, Any]:
        """Returns the currently playing track and the upcoming queue."""
        response = await self._request("GET", "/me/player/queue")
        response.raise_for_status()
        data = response.json()
        currently_playing = data.get("currently_playing")
        return {
            "current_track": _track_to_dict(currently_playing) if currently_playing else None,
            "queue": [_track_to_dict(t) for t in data.get("queue", [])],
        }

    async def get_saved_tracks(self, limit: int = 20) -> list[dict[str, Any]]:
        """Returns the user's saved ('liked') tracks."""
        response = await self._request("GET", "/me/tracks", params={"limit": limit})
        response.raise_for_status()
        tracks = []
        for item in response.json().get("items", []):
            track = _track_to_dict(item["track"])
            track["is_favorite"] = True
            tracks.append(track)
        return tracks

    async def save_tracks(self, track_ids: list[str]) -> None:
        """Adds tracks to the user's saved library."""
        response = await self._request("PUT", "/me/tracks", params={"ids": ",".join(track_ids)})
        response.raise_for_status()

    async def remove_saved_tracks(self, track_ids: list[str]) -> None:
        """Removes tracks from the user's saved library."""
        response = await self._request(
            "DELETE", "/me/tracks", params={"ids": ",".join(track_ids)}
        )
        response.raise_for_status()

    async def get_top_tracks(
        self, time_range: str = "medium_term", limit: int = 20
    ) -> list[dict[str, Any]]:
        """Returns the user's top tracks for the given time range."""
        response = await self._request(
            "GET", "/me/top/tracks", params={"time_range": time_range, "limit": limit}
        )
        response.raise_for_status()
        return [_track_to_dict(item) for item in response.json().get("items", [])]

    async def get_top_artists(
        self, time_range: str = "medium_term", limit: int = 20
    ) -> list[dict[str, Any]]:
        """Returns the user's top artists for the given time range."""
        response = await self._request(
            "GET", "/me/top/artists", params={"time_range": time_range, "limit": limit}
        )
        response.raise_for_status()
        return [_artist_to_dict(item) for item in response.json().get("items", [])]

    async def get_current_user_id(self) -> str:
        """Returns (and caches) the current user's Spotify ID."""
        if self._user_id:
            return self._user_id
        response = await self._request("GET", "/me")
        response.raise_for_status()
        user_id: str = response.json()["id"]
        self._user_id = user_id
        return user_id

    async def create_playlist(
        self, user_id: str, name: str, description: str = ""
    ) -> dict[str, Any]:
        """Creates a new playlist for the given user."""
        response = await self._request(
            "POST",
            f"/users/{user_id}/playlists",
            json={"name": name, "description": description, "public": True},
        )
        response.raise_for_status()
        data = response.json()
        return {
            "playlist_id": data["id"],
            "playlist_name": data["name"],
            "playlist_url": data.get("external_urls", {}).get("spotify"),
        }

    async def add_tracks_to_playlist(self, playlist_id: str, track_uris: list[str]) -> None:
        """Adds tracks to an existing playlist."""
        response = await self._request(
            "POST", f"/playlists/{playlist_id}/tracks", json={"uris": track_uris}
        )
        response.raise_for_status()


_client: SpotifyClient | None = None


def get_spotify_client() -> SpotifyClient:
    """Gets or creates the shared SpotifyClient singleton."""
    global _client
    if _client is None:
        _client = SpotifyClient()
    return _client


def _artist_to_dict(artist: dict[str, Any]) -> dict[str, Any]:
    """Maps a Spotify artist object onto a plain dict."""
    images = artist.get("images", [])
    return {
        "artist_id": artist["id"],
        "name": artist["name"],
        "genres": artist.get("genres", []),
        "popularity": artist.get("popularity"),
        "image_url": images[0]["url"] if images else None,
    }


def _track_to_dict(track: dict[str, Any]) -> dict[str, Any]:
    """Maps a Spotify track object onto MusicTrack-compatible fields."""
    images = (track.get("album") or {}).get("images", [])
    return {
        "track_id": track["id"],
        "uri": track["uri"],
        "title": track["name"],
        "artist": ", ".join(a["name"] for a in track.get("artists", [])),
        "album": (track.get("album") or {}).get("name"),
        "duration_seconds": (track.get("duration_ms") or 0) // 1000,
        "cover_url": images[0]["url"] if images else None,
        "is_favorite": False,
    }
