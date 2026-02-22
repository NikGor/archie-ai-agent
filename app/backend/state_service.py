"""State service for managing user and application state."""

import json
import logging
import os
from datetime import datetime
from typing import Any

import redis

from ..config import DEFAULT_STATE_CONFIG
from ..models.state_models import UserState


logger = logging.getLogger(__name__)


class StateService:
    """Service for managing application and user state."""

    def __init__(self, user_name: str | None = None):
        self.user_name = user_name

        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))

        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
        )
        logger.info(
            f"state_service_001: Redis connected to {redis_host}:{redis_port}/{redis_db}"
        )

    def _get_datetime_info(self) -> dict[str, str]:
        """Get current date and time information."""
        now = datetime.now()
        return {
            "current_date": now.strftime("%d.%m.%Y"),
            "current_time": now.strftime("%H:%M"),
            "current_weekday": now.strftime("%A"),
        }

    def _get_default_state(self) -> dict[str, Any]:
        """Get default user state with datetime info."""
        datetime_info = self._get_datetime_info()
        return {
            "user_name": self.user_name or "User",
            **DEFAULT_STATE_CONFIG,
            **datetime_info,
            "measurement_units": "metric",
            "date_format": "DD Month YYYY",
            "time_format": "24h",
            "commercial_check_open_now": True,
        }

    def get_user_state(self) -> UserState:
        """Get complete user state for prompt context."""
        if not self.user_name:
            logger.info(
                "state_service_002: No user_name provided, returning default state"
            )
            data = self._get_default_state()
            return UserState(**data)

        redis_key = f"user_state:name:{self.user_name}"
        logger.info(
            f"state_service_003: Fetching state from Redis key: \033[36m{redis_key}\033[0m"
        )

        try:
            user_data_json = self.redis_client.get(redis_key)
            if not user_data_json:
                logger.warning(
                    f"state_service_004: No data found in Redis for key: {redis_key}, using default"
                )
                data = self._get_default_state()
                return UserState(**data)

            user_data = json.loads(user_data_json)
            logger.info(
                f"state_service_005: Loaded user state for: \033[35m{user_data.get('user_name')}\033[0m"
            )
            default_data = self._get_default_state()
            merged = {
                **default_data,
                **{k: v for k, v in user_data.items() if v is not None},
            }
            return UserState(**merged)

        except redis.RedisError as e:
            logger.error(f"state_service_error_001: Redis error: \033[31m{e}\033[0m")
            data = self._get_default_state()
            return UserState(**data)
        except json.JSONDecodeError as e:
            logger.error(
                f"state_service_error_002: JSON decode error: \033[31m{e}\033[0m"
            )
            data = self._get_default_state()
            return UserState(**data)
