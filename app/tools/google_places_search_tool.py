import logging
from typing import Any
import httpx
from app.config import settings
from app.utils.arg_coercion import coerce_bool, coerce_float, coerce_int


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


def _build_payload(
    query: str,
    max_results: int,
    min_rating: float | None,
    price_levels: list[str] | None,
    open_now: bool | None,
    location_lat: float | None,
    location_lng: float | None,
    radius_meters: float | None,
    sort_by: str | None,
) -> dict[str, Any]:
    """Places API searchText request from the tool arguments."""
    payload: dict[str, Any] = {
        "textQuery": query,
        "pageSize": min(max(1, max_results), 20),
    }
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
    if sort_by == "distance" and location_lat is not None and location_lng is not None:
        payload["rankPreference"] = "DISTANCE"
    elif sort_by == "relevance":
        payload["rankPreference"] = "RELEVANCE"
    return payload


# Places API field -> result key, copied as is when present
_PLAIN_FIELDS = {
    "rating": "rating",
    "userRatingCount": "reviews_count",
    "websiteUri": "website",
    "googleMapsUri": "google_maps_url",
    "internationalPhoneNumber": "phone",
    "priceLevel": "price_level",
    "parkingOptions": "parking",
}


def _place_data(place: dict[str, Any]) -> dict[str, Any]:
    """One place of the API response in the shape the LLM reads."""
    place_data: dict[str, Any] = {
        "name": place.get("displayName", {}).get("text", "Unknown"),
        "address": place.get("formattedAddress", ""),
    }
    if "location" in place:
        place_data["location"] = {
            "latitude": place["location"].get("latitude"),
            "longitude": place["location"].get("longitude"),
        }
    for field, key in _PLAIN_FIELDS.items():
        if field in place and field not in ("priceLevel", "parkingOptions"):
            place_data[key] = place[field]
    if "editorialSummary" in place:
        place_data["description"] = place["editorialSummary"].get("text", "")
    if "regularOpeningHours" in place:
        hours = place["regularOpeningHours"]
        place_data["opening_hours"] = {
            "open_now": hours.get("openNow"),
            "weekday_text": hours.get("weekdayDescriptions", []),
        }
    for field in ("priceLevel", "parkingOptions"):
        if field in place:
            place_data[_PLAIN_FIELDS[field]] = place[field]
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
    return place_data


async def _post_search(api_key: str, payload: dict[str, Any]) -> httpx.Response:
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": PLACES_FIELD_MASK,
    }
    logger.info("google_places_002: Calling Google Places API")
    async with httpx.AsyncClient() as client:
        return await client.post(url, json=payload, headers=headers, timeout=30.0)


async def google_places_search_tool(
    query: str,
    max_results: int | str = 10,
    min_rating: float | str | None = None,
    price_levels: list[str] | None = None,
    open_now: bool | str | None = None,
    location_lat: float | str | None = None,
    location_lng: float | str | None = None,
    radius_meters: float | str = 5000.0,
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
    max_results = coerce_int(max_results)
    min_rating = coerce_float(min_rating)
    open_now = coerce_bool(open_now)
    location_lat = coerce_float(location_lat)
    location_lng = coerce_float(location_lng)
    radius_meters = coerce_float(radius_meters)

    try:
        api_key = settings.google_api_key
        if not api_key:
            logger.error(
                "google_places_error_001: \033[31mGOOGLE_API_KEY not found\033[0m"
            )
            return {
                "success": False,
                "message": "GOOGLE_PLACES_API_KEY not configured",
                "query": query,
            }

        payload = _build_payload(
            query,
            max_results,
            min_rating,
            price_levels,
            open_now,
            location_lat,
            location_lng,
            radius_meters,
            sort_by,
        )

        response = await _post_search(api_key, payload)

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

        results = [_place_data(place) for place in places]

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
