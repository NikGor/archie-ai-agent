---

description: "Task list template for feature implementation"
---

# Tasks: Response URL Validation

**Input**: Design documents from `/specs/001-url-validator/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/url_validator.md, quickstart.md

**Tests**: Explicitly required — ARCHIE-123 and the plan mandate unit tests covering all
validator branches (`tests/test_url_validator.py`), so test tasks are included per story.

**Organization**: Tasks are grouped by user story (P1 button, P2 map, P3 text link) so
each can be implemented and validated independently on top of a shared foundation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/independent logic, no unmet dependencies)
- **[Story]**: US1 = broken button, US2 = broken map link, US3 = broken text link
- File paths are exact per plan.md's Project Structure section

---

## Phase 1: Setup

**Purpose**: Create the module and test file skeletons that all later work fills in.

- [X] T001 Create `app/utils/url_validator.py` with the public async stub
      `async def validate_and_fix_urls(content: Content) -> Content` that returns
      `content` unchanged, plus module-level `logger = logging.getLogger(__name__)`
- [X] T002 [P] Create `tests/test_url_validator.py` with shared pytest fixtures: a
      mocked `httpx.AsyncClient` (via `httpx.MockTransport`) for reachability checks
      and a mocked `app.tools.google_search_tool.google_search_tool` for fallback
      resolution, per research.md's "Test isolation from real network calls" decision

**Checkpoint**: Module and test file exist; no behavior yet.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared URL-checking core that every user story (button/map/text) reuses.

**⚠️ CRITICAL**: No user story task may start until this phase is complete.

- [X] T003 Implement URL format validation (must start with `http://`/`https://`,
      else immediately invalid) in `app/utils/url_validator.py`
- [X] T004 Implement async reachability check using the project's existing `httpx`
      dependency — `HEAD` request with a 3-second timeout, treating any network
      error, timeout, or non-2xx status as "invalid" without raising — in
      `app/utils/url_validator.py` (depends on: T003)
- [X] T005 Implement the concurrent check runner: collect every candidate URL from a
      `Content` tree and check them all in one `asyncio.gather` call, returning a
      `url -> UrlCheckResult` mapping (per data-model.md) in
      `app/utils/url_validator.py` (depends on: T004)
- [X] T006 [P] Implement the Google Search fallback helper — given a broken URL's
      button label/text, call `google_search_tool(query)` and extract a candidate
      replacement URL — in `app/utils/url_validator.py` (depends on: T003)
- [X] T007 [P] Implement the `url_validator_NNN:` WARNING logging helper (per
      `agent_docs/logging.md` convention) used for every replacement/conversion/
      removal, in `app/utils/url_validator.py`
- [X] T008 Wire `content = await validate_and_fix_urls(content)` into
      `app/tools/create_output_tool.py` immediately after
      `build_content_from_parsed(...)` returns, with an inline `# ARCHIE-123`
      comment per the constitution's non-obvious-fix rule (depends on: T001)

**Checkpoint**: Core checking/fallback/logging primitives exist and are wired into
Stage 3; no element type is validated yet (stub still passes content through
unchanged in effect, since no story has registered any checks).

---

## Phase 3: User Story 1 - Broken chat button is fixed or converted (Priority: P1) 🎯 MVP

**Goal**: A `FrontendButton` with an unreachable URL for an in-scope command
(`navigate_to`, `url_to`, `open_on_youtube_video`, `open_on_youtube_music`,
`check_amazon`) is never shown to the user as a dead link — it is fixed via
Google Search fallback or converted to an `AssistantButton`.

**Independent Test**: Call `validate_and_fix_urls` with a `Content` tree containing
one `FrontendButton` with a mocked-broken URL for `open_on_youtube_video`; assert
the returned button is either the fallback URL or an `AssistantButton` with
`assistant_request` equal to the original button text — no other code path needed.

### Tests for User Story 1 ⚠️

> Write these tests FIRST; they must FAIL until the Implementation tasks below land.

- [X] T009 [P] [US1] Unit test: `FrontendButton` with a valid, reachable URL is
      returned unchanged, in `tests/test_url_validator.py`
- [X] T010 [P] [US1] Unit test: `FrontendButton` with a broken URL and a mocked
      Google Search fallback result has its `url` replaced with the fallback, in
      `tests/test_url_validator.py`
- [X] T011 [P] [US1] Unit test: `FrontendButton` with a broken URL and no fallback
      found is converted to `AssistantButton(assistant_request=<original text>)`, in
      `tests/test_url_validator.py`

### Implementation for User Story 1

- [X] T012 [US1] Implement the in-scope command filter for `FrontendButton`
      (`navigate_to`, `url_to`, `open_on_youtube_video`, `open_on_youtube_music`,
      `check_amazon` only, per data-model.md) in `app/utils/url_validator.py`
      (depends on: T005)
- [X] T013 [US1] Implement `FrontendButton` → `AssistantButton` conversion (copies
      `text`, sets `assistant_request=text`) for the no-fallback-found case, in
      `app/utils/url_validator.py` (depends on: T012)
- [X] T014 [US1] Wire button walking (filter → check → fallback via T006 → replace
      or convert via T013) into `validate_and_fix_urls`'s content tree traversal, in
      `app/utils/url_validator.py` (depends on: T012, T013)

**Checkpoint**: User Story 1 is fully functional and independently testable — broken
buttons are fixed or safely converted. This alone is a deployable MVP.

---

## Phase 4: User Story 2 - Broken map link is omitted (Priority: P2)

**Goal**: A `LocationCard.open_map_url` that is broken is set to `None` instead of
being shown as a dead map link, while the rest of the location card is preserved.

**Independent Test**: Call `validate_and_fix_urls` with a `Content` tree containing
one `LocationCard` with a mocked-broken `open_map_url`; assert the returned card has
`open_map_url is None` and all other fields unchanged.

### Tests for User Story 2 ⚠️

- [X] T015 [P] [US2] Unit test: `LocationCard` with a valid, reachable
      `open_map_url` is returned unchanged, in `tests/test_url_validator.py`
- [X] T016 [P] [US2] Unit test: `LocationCard` with a broken `open_map_url` has it
      set to `None`, with `title`/`address`/`description` untouched, in
      `tests/test_url_validator.py`

### Implementation for User Story 2

- [X] T017 [US2] Implement `LocationCard.open_map_url` validation (no fallback
      search attempted, per data-model.md) and wire it into `validate_and_fix_urls`'s
      content tree traversal, in `app/utils/url_validator.py` (depends on: T005)

**Checkpoint**: User Stories 1 and 2 both work independently — buttons and map links
are both handled.

---

## Phase 5: User Story 3 - Broken text link does not mislead the user (Priority: P3)

**Goal**: A broken markdown link `[label](url)` inside `TextAnswer.text` (when
`type == "markdown"`) is replaced with a working link or reduced to plain `label`
text — never left as a dead clickable link.

**Independent Test**: Call `validate_and_fix_urls` with a `Content` tree containing
one `TextAnswer(type="markdown", text="see [this](broken-url)")`; assert the
returned text either contains a replacement URL or reads `"see this"` with no
markdown link syntax.

### Tests for User Story 3 ⚠️

- [X] T018 [P] [US3] Unit test: markdown text with a valid, reachable link is
      returned unchanged, in `tests/test_url_validator.py`
- [X] T019 [P] [US3] Unit test: markdown text with a broken link and a mocked
      Google Search fallback has the link URL replaced, in
      `tests/test_url_validator.py`
- [X] T020 [P] [US3] Unit test: markdown text with a broken link and no fallback
      found is reduced to plain label text with no markdown link syntax, in
      `tests/test_url_validator.py`

### Implementation for User Story 3

- [X] T021 [US3] Implement markdown link extraction (`[label](url)` regex) for
      `TextAnswer.text` when `type == "markdown"`, in `app/utils/url_validator.py`
      (depends on: T005)
- [X] T022 [US3] Implement the rewrite (replace URL) / strip (drop to plain label)
      logic and wire it into `validate_and_fix_urls`'s content tree traversal, in
      `app/utils/url_validator.py` (depends on: T021, T006)

**Checkpoint**: All three user stories are independently functional — buttons, map
links, and text links are all validated per the spec.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Confirm the full feature holds together and leaves no regressions.

- [X] T023 [P] Run `./execute_tests.sh -m "not llm"` and confirm every existing test
      still passes unchanged (FR-008 / Constitution Principle II)
- [X] T024 [P] Execute the manual smoke steps in `specs/001-url-validator/quickstart.md`
      (latency check + `url_validator_NNN:` log line confirmation)
- [X] T025 Review `app/utils/url_validator.py` end-to-end for dead code and
      single-responsibility per the constitution's mandatory post-implementation
      review step

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup (T001) — BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational (T005, T006) completion
- **User Story 2 (Phase 4)**: Depends on Foundational (T005) completion — independent
  of User Story 1
- **User Story 3 (Phase 5)**: Depends on Foundational (T005, T006) completion —
  independent of User Story 1 and 2
- **Polish (Phase 6)**: Depends on all three user stories being complete

### User Story Dependencies

- **US1 (P1)**: No dependency on US2/US3 — buttons are handled entirely within
  `FrontendButton`/`AssistantButton`
- **US2 (P2)**: No dependency on US1/US3 — map links have no fallback path at all
- **US3 (P3)**: No dependency on US1/US2 — text links are handled entirely within
  `TextAnswer`

All three stories touch the same file (`app/utils/url_validator.py`) but different,
non-overlapping functions/branches, so they are logically independent even though
not physically parallelizable by file lock.

### Parallel Opportunities

- T002 (test file skeleton) can run in parallel with T001 (module skeleton)
- T006 and T007 (foundational) can run in parallel with each other, after T003/T004
- All three Tests sub-phases within a story (T009-T011, T015-T016, T018-T020) can
  run in parallel with each other (independent test functions in the same file)
- Once Phase 2 is complete, US1/US2/US3 implementation work can proceed in any
  order or in parallel if staffed by different people

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Unit test: valid FrontendButton URL passes unchanged in tests/test_url_validator.py"
Task: "Unit test: broken FrontendButton URL + fallback replaces url in tests/test_url_validator.py"
Task: "Unit test: broken FrontendButton URL + no fallback converts to AssistantButton in tests/test_url_validator.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T002)
2. Complete Phase 2: Foundational (T003-T008) — CRITICAL, blocks all stories
3. Complete Phase 3: User Story 1 (T009-T014)
4. **STOP and VALIDATE**: run T009-T011 tests independently; confirm broken buttons
   are fixed or converted
5. This is a deployable MVP per ARCHIE-123's most visible failure mode

### Incremental Delivery

1. Setup + Foundational → foundation ready (button/map/text checking core exists)
2. Add User Story 1 → validate independently → MVP ships (P1: buttons)
3. Add User Story 2 → validate independently → ships (P2: map links)
4. Add User Story 3 → validate independently → ships (P3: text links)
5. Polish phase confirms no regression across the whole non-`llm` test suite

---

## Notes

- [P] tasks touch independent logic; when they land in the same file
  (`app/utils/url_validator.py`), apply them as separate, focused commits per the
  constitution's one-file-at-a-time commit rule rather than literal concurrent edits
- Tests MUST be written and confirmed failing before their story's implementation
  tasks (T012-T014, T017, T021-T022) land, per Constitution Principle III
- Commit after each task or logical group; run `./execute_tests.sh -m "not llm"`
  before moving to the next story
- Stop at any checkpoint (end of Phase 3, 4, or 5) to validate that story
  independently before continuing
