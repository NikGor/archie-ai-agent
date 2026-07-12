"""Unit tests for app/utils/tools_utils.py — Pydantic-based tool schema generation."""

from typing import Literal
from app.utils.tools_utils import (
    gemini_parse,
    openai_parse,
    openai_responses_parse,
    oss_parse,
)


def sample_tool(
    query: str,  # noqa: ARG001
    max_results: int = 10,  # noqa: ARG001
    min_rating: float | str | None = None,  # noqa: ARG001
    is_active: bool | str | None = None,  # noqa: ARG001
    tags: list[str] | None = None,  # noqa: ARG001
    mode: Literal["fast", "slow"] = "fast",  # noqa: ARG001
) -> dict:
    """
    Search for something.

    Longer description of the tool.

    Args:
        query: Search text
        max_results: Max number of results
        min_rating: Minimum rating, allowed values: 1, 2, 3
        is_active: Whether the item is active
        tags: List of tags to filter on
        mode: Speed mode. Allowed values: fast, slow
    """
    return {}


def context_tool(context: str, name: str) -> dict:  # noqa: ARG001
    """
    Tool with a context parameter.

    Args:
        context: Execution context (skipped by gemini/oss parsers)
        name: Name to use
    """
    return {}


class TestOpenaiParse:
    def test_basic_shape(self):
        schema = openai_parse(sample_tool)
        assert schema["name"] == "sample_tool"
        assert "Search for something." in schema["description"]
        assert "Longer description" in schema["description"]

        props = schema["parameters"]["properties"]
        assert props["query"] == {"type": "string", "description": "Search text"}
        assert props["max_results"]["type"] == "integer"
        assert props["min_rating"]["type"] == "number"
        assert props["is_active"]["type"] == "boolean"
        # Literal type resolves to string type + enum
        assert props["mode"]["type"] == "string"
        assert props["mode"]["enum"] == ["fast", "slow"]

    def test_list_type_falls_back_to_string(self):
        """`list[str] | None` has no JSON-schema-native primitive; keep legacy 'string' fallback."""
        schema = openai_parse(sample_tool)
        assert schema["parameters"]["properties"]["tags"]["type"] == "string"

    def test_required_params(self):
        schema = openai_parse(sample_tool)
        assert schema["parameters"]["required"] == ["query"]


class TestGeminiParse:
    def test_skips_context_param(self):
        schema = gemini_parse(context_tool)
        assert "context" not in schema["parameters"]["properties"]
        assert "name" in schema["parameters"]["properties"]

    def test_allowed_values_enum_from_description(self):
        schema = gemini_parse(sample_tool)
        prop = schema["parameters"]["properties"]["mode"]
        assert prop["enum"] == ["fast", "slow"]

    def test_date_format_hint(self):
        def tool_with_date(when: str) -> dict:  # noqa: ARG001
            """
            Tool.

            Args:
                when: Date to use, e.g. 2024-01-01
            """
            return {}

        schema = gemini_parse(tool_with_date)
        assert schema["parameters"]["properties"]["when"]["format"] == "date"


class TestOssParse:
    def test_no_descriptions_or_required(self):
        schema = oss_parse(sample_tool)
        assert "required" not in schema["parameters"]
        for prop in schema["parameters"]["properties"].values():
            assert "description" not in prop

    def test_types_present(self):
        schema = oss_parse(sample_tool)
        assert schema["parameters"]["properties"]["query"]["type"] == "string"
        assert schema["parameters"]["properties"]["max_results"]["type"] == "integer"

    def test_skips_context_param(self):
        schema = oss_parse(context_tool)
        assert "context" not in schema["parameters"]["properties"]


class TestOpenaiResponsesParse:
    def test_shape_and_strict_mode(self):
        schema = openai_responses_parse(sample_tool)
        assert schema["type"] == "function"
        assert schema["strict"] is True
        assert schema["parameters"]["additionalProperties"] is False
        # strict mode requires ALL properties in required, regardless of defaults
        assert set(schema["parameters"]["required"]) == set(
            schema["parameters"]["properties"].keys()
        )

    def test_literal_enum(self):
        schema = openai_responses_parse(sample_tool)
        assert schema["parameters"]["properties"]["mode"]["enum"] == ["fast", "slow"]

    def test_does_not_skip_context_param(self):
        """Unlike gemini/oss parsers, the Responses API parser keeps 'context'."""
        schema = openai_responses_parse(context_tool)
        assert "context" in schema["parameters"]["properties"]
