"""
Unit tests for document_search_tool.

These tests require OPENAI_VECTOR_STORE_ID to be set in environment.
"""

import os
import pytest

import app.tools.document_search_tool as doc_search_module
from app.tools.document_search_tool import document_search_tool

doc_search_module.VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")


@pytest.mark.asyncio
async def test_document_search_costa_invoice():
    """Test search for Costa invoice date."""
    query = "когда мне был выставлен счет от Коста?"
    result = await document_search_tool(query=query, limit=5)
    assert result["success"] is True, f"Search failed: {result.get('error')}"
    assert result["query"] == query
    assert "results" in result
    assert result["count"] > 0, "Expected at least one result"
    for item in result["results"]:
        assert "file_id" in item
        assert "filename" in item
        assert "score" in item
        assert "content" in item
    print(f"\nFound {result['count']} results:")
    for i, item in enumerate(result["results"], 1):
        print(f"\n{i}. {item['filename']} (score: {item['score']:.3f})")
        print(f"   Content preview: {item['content'][:200]}...")
