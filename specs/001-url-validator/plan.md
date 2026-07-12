# Implementation Plan: Response URL Validation

**Branch**: `001-url-validator` | **Date**: 2026-07-12 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-url-validator/spec.md`

## Summary

Stage 3 output can contain hallucinated URLs in three places — `FrontendButton.url`,
`LocationCard.open_map_url`, and markdown links inside `TextAnswer.text`. Add a new
`app/utils/url_validator.py` module invoked in `create_output_tool.py` right after
`build_content_from_parsed()`. It checks every URL in parallel
(`asyncio.gather`): format check (http/https) → `httpx` `HEAD` request (3s timeout) →
for command types tied to YouTube/Amazon/web navigation, fall back to
`google_search_tool` for a replacement. Unrecoverable buttons become
`AssistantButton(assistant_request=...)`, unrecoverable map links are set to `None`,
unrecoverable markdown links are stripped to plain text. Every replacement is logged
at WARNING with the `url_validator_` prefix.

## Technical Context

**Language/Version**: Python 3.11+ (project standard, `|` union syntax)

**Primary Dependencies**: `httpx` (already a project dependency, async HEAD requests),
existing `app/tools/google_search_tool.py` (`async def google_search_tool(query: str) -> dict[str, Any]`),
`archie_shared.ui.models` (`FrontendButton`, `AssistantButton`, `LocationCard`, `TextAnswer`)

**Storage**: N/A — stateless, per-response validation, no persistence

**Testing**: `pytest` via `./execute_tests.sh -m "not llm"` (unit tests must not require
network access — HTTP calls mocked with `httpx` transport mocking/`respx`-style mocking
or `unittest.mock`)

**Target Platform**: Existing FastAPI service (Linux/Docker), Stage 3 of the request pipeline

**Project Type**: Single project (existing monolithic FastAPI service, `app/` layout)

**Performance Goals**: Per FR-006/SC-003 — URL checks must not noticeably add to
response latency; achieved by checking all URLs in a response concurrently via
`asyncio.gather` with a per-request 3s HEAD timeout (bounded worst case, not
proportional to link count)

**Constraints**: 3s timeout per HTTP HEAD check (from ARCHIE-123); no new external
service dependency beyond the existing `google_search_tool`; must not break any
existing non-LLM test in `./execute_tests.sh -m "not llm"`

**Scale/Scope**: Applies to every Stage 3 response containing 0–N links (buttons, one
location card, inline text); N is small (single-digit) per response in practice

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Status |
|---|---|---|
| I. Staged Pipeline Integrity | Validator runs strictly inside Stage 3 (`create_output_tool.py`, after `build_content_from_parsed()`); no Stage 1/2 boundary is touched | PASS |
| II. Verification Over Assertion | Feature ships with unit tests (`tests/test_url_validator.py`) covering valid/404/markdown/button-conversion/search-fallback; `./execute_tests.sh` is the completion gate | PASS |
| III. Test-First Discipline | `tests/test_url_validator.py` planned as part of this feature, covering all branches named in the spec/ticket | PASS |
| IV. Reuse Before Creation | Reuses existing `httpx` dependency and existing `google_search_tool`; no duplicate HTTP client or search logic introduced; `archie_shared` models (`FrontendButton`, `AssistantButton`, `LocationCard`, `TextAnswer`) are consumed as-is, not modified | PASS |
| V. Typed, Minimal, SOLID Code | New module has one responsibility (URL validation + fallback resolution); Pydantic models already typed; no speculative abstraction beyond what the 3 call sites need | PASS |
| VI. Structured Observability | Every replacement logged at WARNING with `url_validator_NNN:` prefix, per `agent_docs/logging.md` convention | PASS |

No violations — Complexity Tracking section not needed.

## Project Structure

### Documentation (this feature)

```text
specs/001-url-validator/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
app/
├── tools/
│   └── create_output_tool.py     # MODIFY: call url_validator after build_content_from_parsed()
└── utils/
    └── url_validator.py          # NEW: async URL validation + fallback module

tests/
└── test_url_validator.py         # NEW: unit tests (valid URL, 404, markdown link,
                                   #      button conversion, Google Search fallback)
```

**Structure Decision**: Single project (Option 1) — this is a targeted addition to the
existing `app/utils/` and `app/tools/` layout already used by the pipeline; no new
top-level project or service boundary is introduced.

## Complexity Tracking

*No violations — table intentionally omitted.*
