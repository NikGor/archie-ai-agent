import inspect
import logging
from typing import Any, Callable, get_type_hints, get_origin, get_args
import docstring_parser


logger = logging.getLogger(__name__)


def _map_type(t: type) -> str:
    """
    Map Python type to JSON schema type.
    
    Handles Union types (e.g., int | None) by extracting the non-None type.
    
    Args:
        t (type): Python type annotation
        
    Returns:
        str: JSON schema type string
    """
    origin = get_origin(t)
    if origin is not None:
        args = get_args(t)
        non_none_types = [arg for arg in args if arg is not type(None)]
        if non_none_types:
            t = non_none_types[0]
    
    if t in [str]:
        return "string"
    if t in [int]:
        return "integer"
    if t in [float]:
        return "number"
    if t in [bool]:
        return "boolean"
    return "string"


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
        properties[param.arg_name] = {
            "type": _map_type(hint),
            "description": param.description or ""
        }
        if _is_required(func, param.arg_name):
            required.append(param.arg_name)
    
    return {
        "name": func.__name__,
        "description": doc.short_description or "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
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
        prop = {
            "type": _map_type(hint),
            "description": param.description or ""
        }
        
        if "allowed values:" in (param.description or "").lower():
            allowed = param.description.lower().split("allowed values:")[1].strip()
            prop["enum"] = [v.strip() for v in allowed.split(",")]
        
        if "date" in (param.description or "").lower():
            prop["format"] = "date"
        
        properties[param.arg_name] = prop
        if _is_required(func, param.arg_name):
            required.append(param.arg_name)
    
    return {
        "name": func.__name__,
        "description": doc.short_description or "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
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
        properties[param.arg_name] = {
            "type": _map_type(hint)
        }
    
    return {
        "name": func.__name__,
        "parameters": {
            "type": "object",
            "properties": properties
        }
    }
