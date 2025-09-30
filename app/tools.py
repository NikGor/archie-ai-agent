import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


async def get_weather(city_name: str) -> dict[str, Any]:
    """
    Get current weather information for a given city.
    
    Args:
        city_name: Name of the city to get weather for
        
    Returns:
        Dictionary containing weather information or error message
    """
    try:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            logger.error("OpenWeather API key not found in environment variables")
            return {"error": "API key not configured"}

        # OpenWeather API endpoint for current weather
        url = "http://api.openweathermap.org/data/2.5/weather"

        params = {
            'q': city_name,
            'appid': api_key,
            'units': 'metric',  # Use Celsius
            'lang': 'ru'  # Russian language for descriptions
        }

        logger.info(f"Requesting weather data for city: {city_name}")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            # Extract relevant weather information
            weather_info = {
                "city": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
                "cloudiness": data["clouds"]["all"]
            }

            logger.info(f"Successfully retrieved weather data for {city_name}")
            return json.dumps(weather_info, ensure_ascii=False)

    except httpx.RequestError as e:
        logger.error(f"Error making request to OpenWeather API: {e}")
        return {"error": f"Failed to fetch weather data: {e!s}"}
    except KeyError as e:
        logger.error(f"Unexpected response format from OpenWeather API: {e}")
        return {"error": "Invalid response format from weather service"}
    except Exception as e:
        logger.error(f"Unexpected error getting weather for {city_name}: {e}")
        return {"error": f"Unexpected error: {e!s}"}

