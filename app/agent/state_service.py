"""State service for managing user and application state."""

from datetime import datetime
from typing import Any


class StateService:
    """Service for managing application and user state."""

    def __init__(
        self,
        user_name: str = "Николай",
        persona: str = "business",
        default_city: str = "Bad Mergentheim",
    ):
        self.user_name = user_name
        self.persona = persona
        self.default_city = default_city

    def _get_datetime_info(self) -> dict[str, str]:
        """Get current date and time information."""
        now = datetime.now()
        return {
            "current_date": now.strftime("%d.%m.%Y"),
            "current_time": now.strftime("%H:%M"),
            "current_weekday": now.strftime("%A"),
        }

    def get_user_state(self) -> dict[str, Any]:
        """Get complete user state for prompt context."""
        datetime_info = self._get_datetime_info()
        return {
            "user_name": self.user_name,
            "persona": self.persona,
            "default_city": self.default_city,
            "default_country": "Germany",
            **datetime_info,
            "user_timezone": "Europe/Berlin",
            "measurement_units": "metric",
            "language": "ru",
            "currency": "EUR",
            "date_format": "DD Month YYYY",
            "time_format": "24h",
            "commercial": {
                "holidays": "DE-BW",
                "check_open_now": True,
            },
            "user_preferences": {
                "transport": ["car", "bicycle"],
                "cuisine": ["italian", "russian", "ukrainian"],
            },
            "current_environment": {
                "weather": None,
                "sunrise": None,
                "sunset": None,
            },
        }
