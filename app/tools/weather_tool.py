"""Weather tool: current weather and forecast from Open-Meteo (free, no API key).

Finds the place with the Open-Meteo geocoding API, then reads the current
conditions, the next hours and the daily forecast in the place's time zone.
"""

import logging
from datetime import datetime
from typing import Any
import httpx


logger = logging.getLogger(__name__)

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT_SECONDS = 10.0
_NEXT_HOURS = 12
_HOUR_STEP = 3
_MAX_DAYS = 7

# WMO weather interpretation codes (Open-Meteo docs)
_WEATHER_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "snow showers",
    95: "thunderstorm",
    96: "thunderstorm with light hail",
    99: "thunderstorm with hail",
}


def _condition(code: int | None) -> str:
    return _WEATHER_CODES.get(code, f"code {code}") if code is not None else "unknown"


async def _find_place(client: httpx.AsyncClient, city: str) -> dict[str, Any]:
    response = await client.get(
        _GEOCODING_URL, params={"name": city, "count": 1, "language": "en"}
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        raise ValueError(f"Place not found: {city}")
    return results[0]


def _next_hours(hourly: dict[str, list[Any]], now: str) -> list[dict[str, Any]]:
    start = next((i for i, t in enumerate(hourly["time"]) if t >= now[:13]), 0)
    return [
        {
            "time": hourly["time"][i][11:16],
            "temperature_c": hourly["temperature_2m"][i],
            "precipitation_probability": hourly["precipitation_probability"][i],
            "condition": _condition(hourly["weather_code"][i]),
        }
        for i in range(
            start, min(start + _NEXT_HOURS + 1, len(hourly["time"])), _HOUR_STEP
        )
    ]


def _days(daily: dict[str, list[Any]]) -> list[dict[str, Any]]:
    return [
        {
            "date": datetime.fromisoformat(daily["time"][i]).strftime("%d.%m.%Y"),
            "condition": _condition(daily["weather_code"][i]),
            "temperature_min_c": daily["temperature_2m_min"][i],
            "temperature_max_c": daily["temperature_2m_max"][i],
            "precipitation_mm": daily["precipitation_sum"][i],
            "precipitation_probability": daily["precipitation_probability_max"][i],
            "wind_max_ms": daily["wind_speed_10m_max"][i],
            "sunrise": daily["sunrise"][i][11:16],
            "sunset": daily["sunset"][i][11:16],
        }
        for i in range(len(daily["time"]))
    ]


async def weather_tool(
    city: str,
    days: int = 1,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """
    Get the current weather, the next hours and the daily forecast for a place.
    Free Open-Meteo data, no API key. Temperatures in °C, wind in m/s.

    Args:
        city: Place name, e.g. the user's city "Bad Mergentheim"
        days: Number of forecast days starting today (1-7)

    Returns:
        Dict with location, current conditions, next_hours (every 3 hours) and days
    """
    days = max(1, min(int(days), _MAX_DAYS))
    logger.info(f"weather_001: Weather for \033[36m{city}\033[0m, days={days}")
    if demo_mode:
        return {
            "location": city,
            "current": {"temperature_c": 14.0, "condition": "partly cloudy"},
            "message": "[DEMO] Weather data",
        }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            place = await _find_place(client, city)
            response = await client.get(
                _FORECAST_URL,
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code,cloud_cover,wind_speed_10m,wind_gusts_10m",
                    "hourly": "temperature_2m,precipitation_probability,weather_code",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_sum,precipitation_probability_max,wind_speed_10m_max,"
                    "sunrise,sunset",
                    "timezone": "auto",
                    "forecast_days": max(days, 2),
                    "wind_speed_unit": "ms",
                },
            )
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError) as error:
        logger.error(f"weather_error_001: \033[31m{error}\033[0m")
        return {"success": False, "error": str(error)}
    current = data["current"]
    return {
        "location": f"{place['name']}, {place.get('country', '')}".strip(", "),
        "current": {
            "time": current["time"][11:16],
            "temperature_c": current["temperature_2m"],
            "feels_like_c": current["apparent_temperature"],
            "humidity_percent": current["relative_humidity_2m"],
            "precipitation_mm": current["precipitation"],
            "cloud_cover_percent": current["cloud_cover"],
            "wind_ms": current["wind_speed_10m"],
            "wind_gusts_ms": current["wind_gusts_10m"],
            "condition": _condition(current["weather_code"]),
        },
        "next_hours": _next_hours(data["hourly"], current["time"]),
        "days": _days(data["daily"])[:days],
    }
