import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


PLACES_FIELD_MASK = (
    "places.displayName,"
    "places.formattedAddress,"
    "places.location,"
    "places.rating,"
    "places.userRatingCount,"
    "places.websiteUri,"
    "places.regularOpeningHours,"
    "places.priceLevel,"
    "places.internationalPhoneNumber,"
    "places.googleMapsUri,"
    "places.editorialSummary,"
    "places.reviews,"
    "places.parkingOptions"
)


async def google_places_search_tool(
    query: str,
    max_results: int = 10,
    min_rating: float | None = None,
    price_levels: list[str] | None = None,
    open_now: bool | None = None,
    location_lat: float | None = None,
    location_lng: float | None = None,
    radius_meters: float = 5000.0,
    sort_by: str = "relevance",
) -> dict[str, Any]:
    """
    Search for physical locations, places, businesses, and addresses.
    ALWAYS use this tool FIRST when user asks about places, locations, or "where".

    Use cases: restaurants, cafes, parking, hotels, shops, banks, pharmacies,
    gas stations, attractions, airports, any "near me" or "in [city]" queries,
    finding addresses, business locations.

    Returns: name, address, coordinates, rating, reviews (count + texts),
    opening hours, price level, phone, description, parking options,
    website, Google Maps link.

    If you need more specific details not returned by this tool,
    use google_search_tool as fallback.

    Args:
        query: Location search query (e.g., "Parking near FRA", "German restaurants in Berlin")
        max_results: Maximum number of results to return (1-20, default 10)
        min_rating: Minimum rating filter (0.0-5.0, e.g., 4.0 for 4+ stars)
        price_levels: Filter by price, list of: PRICE_LEVEL_INEXPENSIVE, PRICE_LEVEL_MODERATE, PRICE_LEVEL_EXPENSIVE, PRICE_LEVEL_VERY_EXPENSIVE
        open_now: If True, only return places that are currently open
        location_lat: Latitude for location-based search (required for sort_by="distance")
        location_lng: Longitude for location-based search (required for sort_by="distance")
        radius_meters: Search radius in meters (default 5000m = 5km)
        sort_by: Sort results by "relevance" (default) or "distance" (requires location_lat/lng)

    Returns:
        Dict with comprehensive places data
    """
    logger.info(
        f"google_places_001: Search requested for query: \033[36m{query}\033[0m"
    )

    # Type coercion for LLM-provided string values
    if isinstance(max_results, str):
        max_results = int(max_results)
    if isinstance(min_rating, str):
        min_rating = float(min_rating)
    if isinstance(open_now, str):
        open_now = open_now.lower() == "true"
    if isinstance(location_lat, str):
        location_lat = float(location_lat)
    if isinstance(location_lng, str):
        location_lng = float(location_lng)
    if isinstance(radius_meters, str):
        radius_meters = float(radius_meters)

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error(
                "google_places_error_001: \033[31mGOOGLE_API_KEY not found\033[0m"
            )
            return {
                "success": False,
                "message": "GOOGLE_PLACES_API_KEY not configured",
                "query": query,
            }

        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": PLACES_FIELD_MASK,
        }
        payload = {"textQuery": query, "pageSize": min(max(1, max_results), 20)}
        if min_rating is not None:
            payload["minRating"] = min(max(0.0, min_rating), 5.0)
        if price_levels:
            payload["priceLevels"] = price_levels
        if open_now is not None:
            payload["openNow"] = open_now
        if location_lat is not None and location_lng is not None:
            payload["locationBias"] = {
                "circle": {
                    "center": {"latitude": location_lat, "longitude": location_lng},
                    "radius": radius_meters,
                }
            }
        if (
            sort_by == "distance"
            and location_lat is not None
            and location_lng is not None
        ):
            payload["rankPreference"] = "DISTANCE"
        elif sort_by == "relevance":
            payload["rankPreference"] = "RELEVANCE"

        logger.info("google_places_002: Calling Google Places API")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=30.0,
            )

        if response.status_code != 200:
            logger.error(
                f"google_places_error_002: API returned status \033[31m{response.status_code}\033[0m"
            )
            return {
                "success": False,
                "message": f"API error: {response.status_code}",
                "query": query,
                "details": response.text,
            }

        data = response.json()
        places = data.get("places", [])

        logger.info(f"google_places_003: Found \033[33m{len(places)}\033[0m places")

        results = []
        for place in places:
            place_data = {
                "name": place.get("displayName", {}).get("text", "Unknown"),
                "address": place.get("formattedAddress", ""),
            }
            if "location" in place:
                place_data["location"] = {
                    "latitude": place["location"].get("latitude"),
                    "longitude": place["location"].get("longitude"),
                }
            if "rating" in place:
                place_data["rating"] = place["rating"]
            if "userRatingCount" in place:
                place_data["reviews_count"] = place["userRatingCount"]
            if "websiteUri" in place:
                place_data["website"] = place["websiteUri"]
            if "googleMapsUri" in place:
                place_data["google_maps_url"] = place["googleMapsUri"]
            if "internationalPhoneNumber" in place:
                place_data["phone"] = place["internationalPhoneNumber"]
            if "editorialSummary" in place:
                place_data["description"] = place["editorialSummary"].get("text", "")
            if "regularOpeningHours" in place:
                hours = place["regularOpeningHours"]
                place_data["opening_hours"] = {
                    "open_now": hours.get("openNow"),
                    "weekday_text": hours.get("weekdayDescriptions", []),
                }
            if "priceLevel" in place:
                place_data["price_level"] = place["priceLevel"]
            if "parkingOptions" in place:
                place_data["parking"] = place["parkingOptions"]
            if "reviews" in place:
                place_data["reviews"] = [
                    {
                        "author": r.get("authorAttribution", {}).get("displayName", ""),
                        "rating": r.get("rating"),
                        "text": r.get("text", {}).get("text", ""),
                        "time": r.get("relativePublishTimeDescription", ""),
                    }
                    for r in place["reviews"][:3]
                ]
            results.append(place_data)

        return {
            "success": True,
            "query": query,
            "places": results,
            "count": len(results),
        }

    except httpx.TimeoutException:
        logger.error("google_places_error_003: \033[31mRequest timeout\033[0m")
        return {
            "success": False,
            "message": "Request timeout",
            "query": query,
        }
    except Exception as e:
        logger.error(f"google_places_error_004: \033[31m{e!s}\033[0m")
        return {
            "success": False,
            "message": f"Search failed: {e!s}",
            "query": query,
        }
