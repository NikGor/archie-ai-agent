# Phase 0 Research: Response URL Validation

No `NEEDS CLARIFICATION` markers remained in the Technical Context after
inspecting the existing codebase, so this phase documents the concrete
decisions made rather than open unknowns.

## Decision: HTTP client for reachability checks

**Decision**: Use the existing `httpx` async client (already a project
dependency, `^0.28.0`) to issue `HEAD` requests with a 3-second timeout.

**Rationale**: `httpx` is already used elsewhere in the codebase for async
HTTP; adding a second HTTP library (e.g. `aiohttp`) would violate the Reuse
Before Creation principle. `HEAD` avoids downloading response bodies, keeping
checks cheap.

**Alternatives considered**:
- `aiohttp` — rejected, would add a duplicate dependency for the same need.
- `GET` instead of `HEAD` — rejected, unnecessarily downloads full response
  bodies just to check existence; some URLs (video pages) could be large.
- `requests` (sync) — rejected, would block the event loop and defeats the
  `asyncio.gather` parallelism required by FR-006.

## Decision: Fallback resolution via existing `google_search_tool`

**Decision**: When a URL is broken and belongs to a YouTube/Amazon/web
navigation command (`navigate_to`, `url_to`, `open_on_youtube_video`,
`open_on_youtube_music`, `check_amazon`), call the existing
`app/tools/google_search_tool.google_search_tool(query: str) -> dict[str, Any]`
with a query derived from the button's `text`/label to find a replacement
result.

**Rationale**: The tool already exists and is used elsewhere in the pipeline
for web search; reusing it avoids a second search integration (Reuse Before
Creation). No signature change is needed — it is called with a plain string
query.

**Alternatives considered**:
- New dedicated search client scoped to `url_validator.py` — rejected as
  unnecessary duplication.
- Skipping fallback entirely and always converting to `AssistantButton` —
  rejected, contradicts FR-002 and the ticket's explicit fallback
  requirement.

## Decision: Concurrency model

**Decision**: Collect every URL-bearing element from the parsed content in
one pass, then check all of them concurrently with a single
`asyncio.gather(*checks)` call before returning the (possibly mutated)
content to the rest of `create_output_tool.py`.

**Rationale**: Matches FR-006/SC-003 (response latency must not scale with
link count) and the project's existing pattern of parallel tool execution via
`asyncio.gather` (`app/utils/tool_executor.py`).

**Alternatives considered**:
- Sequential per-URL checks — rejected, latency would scale linearly with
  link count, violating SC-003.

## Decision: Test isolation from real network calls

**Decision**: Unit tests in `tests/test_url_validator.py` mock the `httpx`
transport (or the calling function) so no real HTTP requests are made,
keeping the tests runnable under `./execute_tests.sh -m "not llm"` (no
network/API keys required).

**Rationale**: Constitution Principle III (Test-First Discipline) requires
non-`llm`-marked tests to pass without network access.

**Alternatives considered**:
- Hitting real URLs in tests — rejected, flaky and violates the no-network
  requirement for the default test run.

## Open questions resolved

None remain — all Technical Context fields were filled from direct
inspection of `pyproject.toml`, `app/tools/create_output_tool.py`,
`app/tools/google_search_tool.py`, and `archie_shared.ui.models`.
