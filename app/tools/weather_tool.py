"""Weather tool for getting weather information."""

import json
import logging
import os
from typing import Any
import httpx


logger = logging.getLogger(__name__)


async def get_weather(city_name: str) -> dict[str, Any]:
    """Get current weather information for a given city."""
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            logger.error("weather_error_001: OpenWeather API key not found")
            return {"error": "API key not configured"}
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city_name,
            "appid": api_key,
            "units": "metric",
            "lang": "ru",
        }
        logger.info(f"weather_001: Requesting weather for: \033[36m{city_name}\033[0m")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            weather_info = {
                "city": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
                "cloudiness": data["clouds"]["all"],
            }
            logger.info(f"weather_002: Successfully retrieved weather for {city_name}")
            return json.dumps(weather_info, ensure_ascii=False)
    except httpx.RequestError as e:
        logger.error(f"weather_error_002: Request error: \033[31m{e}\033[0m")
        return {"error": f"Failed to fetch weather data: {e!s}"}
    except KeyError as e:
        logger.error(f"weather_error_003: Invalid response format: \033[31m{e}\033[0m")
        return {"error": "Invalid response format from weather service"}
    except Exception as e:
        logger.error(f"weather_error_004: Unexpected error: \033[31m{e}\033[0m")
        return {"error": f"Unexpected error: {e!s}"}
