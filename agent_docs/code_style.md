# Code Style

## Python

- **Python 3.11+**, PEP8, type annotations on all function signatures
- `|` union syntax (`str | None`, not `Optional[str]`)
- **f-strings** for formatting; `"""` for docstrings, `"` for single-line strings
- `async/await` for all I/O; `asyncio.gather` for concurrent calls
- Run `make format` after writing code

## Pydantic Models

- `BaseModel` for all data structures with `Field(description=...)` on every field
- Access fields directly — don't guard with `hasattr` or `.get()`
- Use `model_dump()` / `model_validate()`, not deprecated `.dict()` / `.parse_obj()`

## Architecture Rules

- One responsibility per function/class
- No global state; use dependency injection or module-level singletons for clients
- Prefer dict dispatch over long `if/elif` chains
- No `print()` — use `logger` only
- No hardcoded values — use `config.py`, env vars, or constants
- No wildcard imports (`from x import *`)
- No code in `__init__.py`

## Import Order

```python
import os                          # 1. stdlib
from openai import OpenAI          # 2. third-party
from app.models.state_models import UserState  # 3. local
```

## Naming

- Functions: `get_`, `create_`, `update_`, `execute_`, `parse_` prefixes
- Modules: `*_tool.py`, `*_service.py`, `*_utils.py`, `*_client.py`
- No abbreviations in names

## What NOT to Do

- Don't use `print()` for logging
- Don't hardcode user-specific defaults (city, timezone, preferences) in source code
- Don't catch bare `except Exception` without logging the specific error
- Don't create new utility functions that duplicate existing ones — check `app/utils/` first
- Don't divide code with blank lines into visual "sections" inside functions
