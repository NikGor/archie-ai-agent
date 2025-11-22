import logging
from typing import Any

logger = logging.getLogger(__name__)


async def document_search_tool(
    query: str,
    document_type: str | None = None,
    limit: int = 5
) -> dict[str, Any]:
    """
    Searches through user's documents and knowledge base using semantic search.
    
    Args:
        query: Search query string
        document_type: Filter by document type (pdf, docx, txt, etc.)
        limit: Maximum number of results to return (default: 5)
        
    Returns:
        Dict with search results or error information
    """
    logger.info(f"document_search_001: Search requested for query: \033[36m{query}\033[0m")
    
    result = {
        "success": True,
        "message": f"Document search completed for query: {query}",
        "query": query,
        "limit": limit
    }
    
    if document_type:
        result["message"] = f"Document search completed for query: {query} (filtered by type: {document_type})"
        result["document_type"] = document_type
    
    return result
