"""Type coercion for LLM-provided tool arguments.

LLM function-calling APIs sometimes send numeric/boolean parameters as
strings; each tool used to reimplement the same isinstance-guarded
conversion independently.
"""


def coerce_int(value: int | str | None) -> int | None:
    return int(value) if isinstance(value, str) else value


def coerce_float(value: float | str | None) -> float | None:
    return float(value) if isinstance(value, str) else value


def coerce_bool(value: bool | str | None) -> bool | None:
    return value.lower() == "true" if isinstance(value, str) else value
