"""State service for managing user and application state."""

import asyncio
import json
import logging
from typing import Any
import redis
from ..config import DEFAULT_STATE_CONFIG, settings
from ..models.state_models import SpotifyPlaybackState, UserState
from .redis_factory import get_async_redis
from .spotify_client import get_spotify_client


logger = logging.getLogger(__name__)

SPOTIFY_CONTEXT_TIMEOUT_SECONDS = 3.0


class StateService:
    """Service for managing application and user state."""

    def __init__(self, user_name: str | None = None):
        self.user_name = user_name
        self.redis_client = get_async_redis()
        logger.info(
            f"state_service_001: Redis connected to "
            f"{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
        )

    def _get_default_state(self, user_name: str | None) -> dict[str, Any]:
        """Get default user state."""
        return {
            "user_name": user_name or "User",
            **DEFAULT_STATE_CONFIG,
            "measurement_units": "metric",
            "date_format": "DD Month YYYY",
            "time_format": "24h",
            "commercial_check_open_now": True,
        }

    async def _get_spotify_context(self) -> SpotifyPlaybackState | None:
        """Fetches ambient Spotify playback context for injection into every LLM call.

        Never raises: any failure (missing credentials, timeout, API error) degrades
        to None so a Spotify outage never blocks an unrelated chat turn.
        """
        try:
            client = get_spotify_client()
            state = await asyncio.wait_for(
                client.get_playback_state(), timeout=SPOTIFY_CONTEXT_TIMEOUT_SECONDS
            )
        except Exception as e:
            logger.warning(
                f"state_service_warn_001: Could not fetch Spotify context: \033[33m{e}\033[0m"
            )
            return None

        if not state:
            return SpotifyPlaybackState(is_playing=False)
        track = state.get("current_track") or {}
        return SpotifyPlaybackState(
            is_playing=state.get("is_playing", False),
            track_title=track.get("title"),
            track_artist=track.get("artist"),
            progress_seconds=state.get("progress_seconds", 0),
            volume=state.get("volume", 50),
            shuffle=state.get("shuffle", False),
            repeat=state.get("repeat", "off"),
        )

    async def get_user_state(
        self, user_name: str | None = None, demo_mode: bool = False
    ) -> UserState:
        """Get complete user state for prompt context, including ambient Spotify playback.

        `user_name` is request-scoped and takes precedence over the constructor
        default — safe to call concurrently on a shared StateService instance.
        """
        user_name = user_name or self.user_name
        spotify_context = None if demo_mode else await self._get_spotify_context()

        if not user_name:
            logger.info(
                "state_service_002: No user_name provided, returning default state"
            )
        else:
            merged = await self._load_merged_state(user_name)
            if merged is not None:
                return UserState(**merged, spotify=spotify_context)

        return UserState(**self._get_default_state(user_name), spotify=spotify_context)

    async def _load_merged_state(self, user_name: str) -> dict[str, Any] | None:
        """Fetch persisted state for `user_name`, merged over defaults.

        Returns None (caller falls back to defaults) on any cache miss or error.
        """
        redis_key = f"user_state:name:{user_name}"
        logger.info(
            f"state_service_003: Fetching state from Redis key: \033[36m{redis_key}\033[0m"
        )
        try:
            user_data_json = await self.redis_client.get(redis_key)
            if not user_data_json:
                logger.warning(
                    f"state_service_004: No data found in Redis for key: {redis_key}, using default"
                )
                return None

            user_data = json.loads(user_data_json)
            logger.info(
                f"state_service_005: Loaded user state for: \033[35m{user_data.get('user_name')}\033[0m"
            )
            default_data = self._get_default_state(user_name)
            return {
                **default_data,
                **{k: v for k, v in user_data.items() if v is not None},
            }

        except redis.RedisError as e:
            logger.error(f"state_service_error_001: Redis error: \033[31m{e}\033[0m")
            return None
        except json.JSONDecodeError as e:
            logger.error(
                f"state_service_error_002: JSON decode error: \033[31m{e}\033[0m"
            )
            return None
