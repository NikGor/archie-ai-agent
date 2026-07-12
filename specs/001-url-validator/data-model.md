# Phase 1 Data Model: Response URL Validation

This feature does not introduce new persisted entities — it validates and,
where necessary, rewrites fields on existing `archie_shared.ui.models`
objects in-memory during Stage 3 output assembly. No new Pydantic models are
required; one internal (non-exported) result type is introduced to carry
validation outcomes between the checker and the rewriting logic.

## Existing entities touched (from `archie_shared.ui.models`)

### FrontendButton
- `command`: one of `navigate_to`, `url_to`, `open_on_youtube_video`,
  `open_on_youtube_music`, `check_amazon`, `open_map`, `call`, `email`,
  `message`, `show_details`, `export_to_notes`, `export_to_calendar`
- `url`: optional string — the field this feature validates
- **Validation rule**: only buttons whose `command` is one of `navigate_to`,
  `url_to`, `open_on_youtube_video`, `open_on_youtube_music`,
  `check_amazon` are in scope (per ARCHIE-123); other commands (`call`,
  `email`, `open_map`, etc.) do not carry a hallucination-prone URL and are
  left untouched.
- **State transition**: `FrontendButton` (broken `url`, no replacement
  found) → `AssistantButton(text=<original text>, assistant_request=<original text>)`

### AssistantButton
- `assistant_request`: natural-language request — set to the original
  button's `text` when a `FrontendButton` is converted (FR-003).
- No fields of this model are themselves validated (it carries no URL).

### LocationCard
- `open_map_url`: optional string — validated; set to `None` if broken and
  unrecoverable (FR-004). No fallback search is attempted for map links
  (out of scope per the ticket — map links are Google Maps links, not
  general web/video/shopping targets).

### TextAnswer
- `text`: string, may contain markdown-style links `[label](url)` when
  `type == "markdown"`.
- **Validation rule**: each markdown link's `url` is extracted, validated,
  and — depending on outcome — left unchanged, replaced with a fallback
  URL, or stripped down to plain `label` text with the `[...]()` markdown
  removed (FR-005). Non-markdown `TextAnswer.type` values are left
  untouched since they cannot contain markdown links.

## Internal types introduced (not part of `archie_shared`)

### `UrlCheckResult` (internal dataclass/BaseModel in `app/utils/url_validator.py`)
- `url: str` — the original URL checked
- `is_valid: bool` — whether the URL passed format + reachability checks
- `replacement_url: str | None` — fallback URL if one was found via
  `google_search_tool`, else `None`

This type exists only to pass results from the concurrent check phase to the
rewrite phase within `url_validator.py`; it is not persisted, not returned
from the module's public function, and not shared with other modules.

## Relationships

```
create_output_tool.build_content_from_parsed()
        │
        ▼
url_validator.validate_and_fix(content)   # new entry point
        │
        ├── walks FrontendButton.url in all buttons  ──► may become AssistantButton
        ├── walks LocationCard.open_map_url           ──► may become None
        └── walks TextAnswer.text markdown links      ──► may lose link, keep label
        │
        ▼
   (possibly mutated) content returned to create_output_tool
```

No new database tables, Redis keys, or cross-service contracts are
introduced by this feature.
