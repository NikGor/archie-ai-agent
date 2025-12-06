import logging
import os
from typing import Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
load_dotenv()


async def google_search_tool(query: str) -> dict[str, Any]:
    """
    Web search for general information, news, facts, events, articles.
    DO NOT use for finding physical locations, places, businesses, or addresses.
    
    Use cases: news, weather, sports scores, factual questions, current events,
    Wikipedia-style lookups, product info, how-to guides.
    
    FALLBACK: Use this tool if other tools returned errors or incomplete data
    (e.g., missing opening hours, prices, reviews from google_places_search_tool).
    
    For finding places (restaurants, parking, hotels, shops, addresses) ->
    use google_places_search_tool FIRST, then fallback to this tool if needed.

    Args:
        query: Search query string (e.g., "Bitcoin price", "Champions League results")

    Returns:
        Dict with search results including text, sources, and metadata
    """
    logger.info(
        f"google_search_001: Search requested for query: \033[36m{query}\033[0m"
    )

    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error(
                "google_search_error_001: \033[31mGEMINI_API_KEY not found\033[0m"
            )
            return {
                "success": False,
                "message": "GEMINI_API_KEY not configured",
                "query": query,
            }

        client = genai.Client(api_key=api_key)

        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

        logger.info("google_search_002: Calling Gemini with Google Search grounding")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
            config=config,
        )

        if not response or not response.candidates:
            logger.warning("google_search_warning_001: Empty response from Gemini")
            return {
                "success": False,
                "message": "Empty response from search API",
                "query": query,
            }

        candidate = response.candidates[0]
        text = response.text
        grounding_metadata = candidate.grounding_metadata

        logger.info(
            f"google_search_003: Response received, length: \033[33m{len(text)}\033[0m chars"
        )

        if not grounding_metadata:
            logger.info(
                "google_search_004: No grounding metadata, answered from model knowledge"
            )
            return {
                "success": True,
                "query": query,
                "answer": text,
                "grounded": False,
                "sources": [],
                "message": "Answer generated from model knowledge (no web search performed)",
            }

        sources = []
        if grounding_metadata.grounding_chunks:
            for idx, chunk in enumerate(grounding_metadata.grounding_chunks):
                if chunk.web:
                    sources.append(
                        {
                            "index": idx + 1,
                            "title": (
                                chunk.web.title
                                if hasattr(chunk.web, "title")
                                else "No title"
                            ),
                            "url": chunk.web.uri if hasattr(chunk.web, "uri") else "",
                        }
                    )

        web_queries = []
        if grounding_metadata.web_search_queries:
            web_queries = list(grounding_metadata.web_search_queries)

        logger.info(
            f"google_search_005: Found \033[33m{len(sources)}\033[0m sources, "
            f"\033[33m{len(web_queries)}\033[0m search queries"
        )

        return {
            "success": True,
            "query": query,
            "answer": text,
            "grounded": True,
            "sources": sources,
            "search_queries": web_queries,
            "grounding_supports": (
                len(grounding_metadata.grounding_supports)
                if grounding_metadata.grounding_supports
                else 0
            ),
        }

    except Exception as e:
        logger.error(f"google_search_error_002: \033[31m{e!s}\033[0m")
        return {
            "success": False,
            "message": f"Search failed: {e!s}",
            "query": query,
        }
