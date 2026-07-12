# Contract: `app/utils/url_validator.py`

This is an internal module contract (function signature and behavior), not
a network API — the project is a single FastAPI service and this feature has
no external-facing endpoint. The contract below is what `create_output_tool.py`
and unit tests rely on.

## Public function

```python
async def validate_and_fix_urls(content: Content) -> Content
```

- **Input**: `content` — the object returned by
  `app/utils/llm_parser.build_content_from_parsed()` (the parsed Stage 3
  content tree containing buttons, cards, and text items).
- **Output**: the same `Content` object, with any broken URLs rewritten or
  removed in place (or an equivalent new instance — the caller only relies
  on the returned value, not on mutation semantics).
- **Side effects**: emits `logger.warning` calls prefixed `url_validator_NNN:`
  for every replacement, conversion, or removal (per FR-007 / Constitution
  Principle VI).
- **Failure mode**: MUST NOT raise for network errors, timeouts, or
  malformed URLs — those are treated as "invalid, no replacement found" and
  handled via the normal fallback path (convert/null/strip). Only
  programming errors (e.g. unexpected type) propagate as exceptions.
- **Concurrency**: all URL checks for a single `content` tree run
  concurrently via a single `asyncio.gather` call; the function's own
  wall-clock cost is bounded by the slowest individual check (≤ 3s HEAD
  timeout), not by the number of links.

## Call site contract

`app/tools/create_output_tool.py` calls `validate_and_fix_urls` exactly once,
immediately after `build_content_from_parsed()` returns, before the content
is passed further into response assembly:

```python
content = build_content_from_parsed(...)
content = await validate_and_fix_urls(content)
```

## Behavioral contract per element type (from data-model.md)

| Element | Broken + replacement found | Broken + no replacement |
|---|---|---|
| `FrontendButton` (in-scope command) | `url` replaced with fallback | converted to `AssistantButton(assistant_request=text)` |
| `LocationCard.open_map_url` | n/a (no fallback attempted) | set to `None` |
| Markdown link in `TextAnswer.text` | link URL replaced with fallback | `[label](url)` reduced to plain `label` |

Any URL that passes the format + reachability check is left completely
unchanged, and elements with no URL are never touched (contract:
zero-link responses are byte-for-byte unaffected, per FR-008).
