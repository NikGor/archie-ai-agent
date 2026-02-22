"""
Shared fixtures and markers for archie-ai-agent tests.
"""

import os

# Load .env before any app module is imported — app clients initialize at module level.
from dotenv import load_dotenv

load_dotenv()

import pytest

from app.agent.agent_factory import AgentFactory


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llm: tests that require a real LLM API key (may be slow)"
    )


def pytest_collection_modifyitems(items):
    """Auto-skip @pytest.mark.llm tests when OPENAI_API_KEY is not set."""
    if os.getenv("OPENAI_API_KEY"):
        return
    skip = pytest.mark.skip(reason="OPENAI_API_KEY not set")
    for item in items:
        if item.get_closest_marker("llm"):
            item.add_marker(skip)


@pytest.fixture
def agent_factory():
    """AgentFactory with demo_mode=True so smarthome tools don't need real devices."""
    return AgentFactory(demo_mode=True)


@pytest.fixture
def simple_messages():
    """A neutral user message that does not require any external tool calls."""
    return [{"role": "user", "content": "Какой сейчас день недели?"}]
