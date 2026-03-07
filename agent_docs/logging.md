# Logging Convention

Use `logger` (not `print()`). Color-code by data type:

```python
logger.info(f"module_001: Description with \033[36m{id}\033[0m")    # Cyan — IDs, URLs
logger.info(f"module_002: Count: \033[33m{count}\033[0m")           # Yellow — numbers
logger.info(f"module_003: Name: \033[35m{name}\033[0m")             # Magenta — names
logger.error(f"module_error_001: \033[31m{error}\033[0m")           # Red — errors
```

Prefix format: `module_NNN:` where module is the short name (e.g. `agent`, `parser`, `executor`).
