"""
Document search tool using OpenAI Vector Stores API.

Setup:
1. Go to https://platform.openai.com/storage/vector_stores
2. Create a new Vector Store (click "Create")
3. Upload files to the Vector Store (PDF, DOCX, TXT, MD, etc.)
4. Copy Vector Store ID (starts with "vs_...")
5. Add to .env: OPENAI_VECTOR_STORE_ID=vs_xxxxxxxxxxxx

Pricing (as of 2025):
- Search: $2.50 per 1,000 queries
- Storage: $0.10 per GB per day (first 1 GB free)
- For personal use (~1000 queries/month, <1GB): ~$2.50/month
"""

import logging
import os
from typing import Any
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")


async def document_search_tool(
    query: str,
    limit: int = 5,
) -> dict[str, Any]:
    """
    Searches through user's documents using OpenAI Vector Store semantic search.

    Args:
        query: Search query string
        limit: Maximum number of results to return (default: 5)

    Returns:
        Dict with search results or error information
    """
    limit = int(limit)
    logger.info(
        f"document_search_001: Search requested for query: \033[36m{query}\033[0m"
    )
    if not VECTOR_STORE_ID:
        logger.warning("document_search_002: OPENAI_VECTOR_STORE_ID not configured")
        return {
            "success": False,
            "error": "Document search not configured. Set OPENAI_VECTOR_STORE_ID.",
            "query": query,
            "results": [],
        }
    try:
        client = AsyncOpenAI()
        search_response = await client.vector_stores.search(
            vector_store_id=VECTOR_STORE_ID,
            query=query,
            max_num_results=limit,
        )
        results = []
        for item in search_response.data:
            content_parts = []
            for content in item.content:
                if content.type == "text":
                    content_parts.append(content.text)
            results.append(
                {
                    "file_id": item.file_id,
                    "filename": item.filename,
                    "score": item.score,
                    "content": "\n".join(content_parts),
                }
            )
        logger.info(f"document_search_003: Found \033[33m{len(results)}\033[0m results")
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
        }
    except Exception as e:
        logger.error(f"document_search_error_001: \033[31m{e}\033[0m")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "results": [],
        }
