import inspect
import logging
import types
from collections.abc import Callable
from typing import Any, Union, get_args, get_origin, get_type_hints
import docstring_parser
from pydantic import TypeAdapter


logger = logging.getLogger(__name__)

# JSON schema "type" values a provider tool schema can express directly.
# A resolved type that maps to anything outside this set (e.g. array, object)
# falls back to "string", preserving the pre-existing (permissive) parser
# behavior for parameter annotations like `list[str] | None` that don't have
# a JSON-schema-native primitive representation in these tool signatures.
_PRIMITIVE_JSON_TYPES = {"string", "integer", "number", "boolean"}


def _full_description(doc: docstring_parser.Docstring) -> str:
    """Return combined short + long description from a parsed docstring."""
    parts = [doc.short_description or ""]
    if doc.long_description:
        parts.append(doc.long_description)
    return " ".join(p for p in parts if p)


def _resolve_hint(t: type) -> type:
    """
    Resolve a type hint to the single type a JSON schema should be built from.

    Tool parameters are frequently declared as unions (e.g. `int | str | None`)
    because arguments arrive from the LLM as strings and get coerced by the
    tool itself. For schema-generation purposes we take the first non-None
    member of the union, matching the previous hand-rolled implementation.
    """
    origin = get_origin(t)
    if origin is Union or origin is types.UnionType:
        args = [arg for arg in get_args(t) if arg is not type(None)]
        if args:
            return args[0]
    return t


def _schema_type_and_enum(t: type) -> tuple[str, list[str] | None]:
    """
    Derive JSON schema "type" and optional "enum" for a parameter type via Pydantic.

    Args:
        t (type): Python type annotation

    Returns:
        tuple[str, list[str] | None]: JSON schema type string and enum values (if any)
    """
    resolved = _resolve_hint(t)
    try:
        schema = TypeAdapter(resolved).json_schema()
    except Exception:
        logger.warning(f"tools_utils_warning_001: Failed to build schema for {t!r}")
        return "string", None

    json_type = schema.get("type")
    enum = schema.get("enum")
    enum_values = list(enum) if isinstance(enum, list) else None

    if json_type not in _PRIMITIVE_JSON_TYPES:
        return "string", enum_values
    return json_type, enum_values


def _is_required(func: Callable, param_name: str) -> bool:
    """
    Check if parameter is required by inspecting function signature.

    Args:
        func (Callable): Function to inspect
        param_name (str): Parameter name

    Returns:
        bool: True if parameter has no default value
    """
    sig = inspect.signature(func)
    param = sig.parameters.get(param_name)
    if param is None:
        return False
    return param.default == inspect.Parameter.empty


def openai_parse(func: Callable) -> dict[str, Any]:
    """
    Parse docstring and type hints into OpenAI JSON schema.

    Args:
        func (Callable): Function to parse

    Returns:
        dict[str, Any]: OpenAI function calling schema
    """
    doc = docstring_parser.parse(func.__doc__ or "")
    type_hints = get_type_hints(func)
    properties = {}
    required = []

    for param in doc.params:
        hint = type_hints.get(param.arg_name, str)
        json_type, enum_values = _schema_type_and_enum(hint)
        prop = {
            "type": json_type,
            "description": param.description or "",
        }
        if enum_values:
            prop["enum"] = enum_values
        properties[param.arg_name] = prop
        if _is_required(func, param.arg_name):
            required.append(param.arg_name)

    return {
        "name": func.__name__,
        "description": _full_description(doc),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def gemini_parse(func: Callable) -> dict[str, Any]:
    """
    Parse docstring and type hints into Gemini JSON schema.

    Supports enums via 'Allowed values: x, y, z' in description.
    Supports format hints for dates.

    Args:
        func (Callable): Function to parse

    Returns:
        dict[str, Any]: Gemini function calling schema
    """
    doc = docstring_parser.parse(func.__doc__ or "")
    type_hints = get_type_hints(func)
    properties = {}
    required = []

    for param in doc.params:
        if param.arg_name == "context":
            continue
        hint = type_hints.get(param.arg_name, str)
        json_type, _ = _schema_type_and_enum(hint)
        prop = {"type": json_type, "description": param.description or ""}

        if "allowed values:" in (param.description or "").lower():
            allowed = (
                (param.description or "").lower().split("allowed values:")[1].strip()
            )
            prop["enum"] = [v.strip() for v in allowed.split(",")]

        if "date" in (param.description or "").lower():
            prop["format"] = "date"

        properties[param.arg_name] = prop
        if _is_required(func, param.arg_name):
            required.append(param.arg_name)

    return {
        "name": func.__name__,
        "description": _full_description(doc),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def oss_parse(func: Callable) -> dict[str, Any]:
    """
    Parse docstring and type hints into OSS-compatible JSON schema.

    Simplified schema without required fields and descriptions.
    Compatible with llama.cpp, Mistral, and other OSS solutions.

    Args:
        func (Callable): Function to parse

    Returns:
        dict[str, Any]: OSS function calling schema
    """
    doc = docstring_parser.parse(func.__doc__ or "")
    type_hints = get_type_hints(func)
    properties = {}

    for param in doc.params:
        if param.arg_name == "context":
            continue
        hint = type_hints.get(param.arg_name, str)
        json_type, _ = _schema_type_and_enum(hint)
        properties[param.arg_name] = {"type": json_type}

    return {
        "name": func.__name__,
        "description": _full_description(doc),
        "parameters": {"type": "object", "properties": properties},
    }


def openai_responses_parse(func: Callable) -> dict[str, Any]:
    """
    Parse function into OpenAI Responses API tool format.

    Returns dict ready for responses.parse() tools parameter.
    Format: {"type": "function", "name": ..., "parameters": {...}, "strict": True}
    Note: strict mode requires ALL properties in required array.

    Args:
        func (Callable): Function to parse

    Returns:
        dict[str, Any]: OpenAI Responses API tool schema
    """
    doc = docstring_parser.parse(func.__doc__ or "")
    type_hints = get_type_hints(func)
    properties = {}

    for param in doc.params:
        hint = type_hints.get(param.arg_name, str)
        json_type, enum_values = _schema_type_and_enum(hint)
        prop = {
            "type": json_type,
            "description": param.description or "",
        }
        if enum_values:
            prop["enum"] = enum_values
        properties[param.arg_name] = prop

    return {
        "type": "function",
        "name": func.__name__,
        "description": _full_description(doc),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
            "additionalProperties": False,
        },
        "strict": True,
    }
