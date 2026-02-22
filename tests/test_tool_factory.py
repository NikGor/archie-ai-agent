"""
Unit tests for ToolFactory.get_tool_schemas().

No LLM calls, no network — only schema generation from tool function signatures.
Tests verify the factory's contract: schemas are valid dicts with required fields,
and format-based filtering works correctly.
"""

import pytest

from app.tools.tool_factory import ToolFactory


@pytest.fixture(scope="module")
def factory():
    return ToolFactory(demo_mode=True)


@pytest.fixture(scope="module")
def plain_schemas(factory):
    return factory.get_tool_schemas("gpt-4.1", "plain")


@pytest.fixture(scope="module")
def dashboard_schemas(factory):
    return factory.get_tool_schemas("gpt-4.1", "dashboard")


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------


def test_plain_schemas_is_nonempty_list(plain_schemas):
    assert isinstance(plain_schemas, list)
    assert len(plain_schemas) > 0


def test_each_schema_has_name(plain_schemas):
    for schema in plain_schemas:
        assert "name" in schema, f"Schema missing 'name': {schema}"


def test_each_schema_has_description(plain_schemas):
    for schema in plain_schemas:
        assert "description" in schema, f"Schema missing 'description': {schema}"


def test_each_schema_has_parameters(plain_schemas):
    for schema in plain_schemas:
        assert "parameters" in schema, f"Schema missing 'parameters': {schema}"


def test_schema_names_are_strings(plain_schemas):
    for schema in plain_schemas:
        assert isinstance(schema["name"], str)
        assert len(schema["name"]) > 0


# ---------------------------------------------------------------------------
# Format-based filtering
# ---------------------------------------------------------------------------


def test_dashboard_format_returns_fewer_schemas_than_plain(
    plain_schemas, dashboard_schemas
):
    """Dashboard/widget formats only expose smarthome tools."""
    assert len(dashboard_schemas) < len(plain_schemas)


def test_dashboard_schemas_is_nonempty(dashboard_schemas):
    assert len(dashboard_schemas) > 0


def test_widget_format_same_as_dashboard(factory):
    widget_schemas = factory.get_tool_schemas("gpt-4.1", "widget")
    dashboard_schemas = factory.get_tool_schemas("gpt-4.1", "dashboard")
    widget_names = {s["name"] for s in widget_schemas}
    dashboard_names = {s["name"] for s in dashboard_schemas}
    assert widget_names == dashboard_names


# ---------------------------------------------------------------------------
# Schema names match known tool registry
# ---------------------------------------------------------------------------


def test_plain_schemas_include_search_tools(plain_schemas):
    names = {s["name"] for s in plain_schemas}
    assert "google_search_tool" in names
    assert "google_places_search_tool" in names
