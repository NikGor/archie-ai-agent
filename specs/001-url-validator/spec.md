# Feature Specification: Response URL Validation

**Feature Branch**: `001-url-validator`

**Created**: 2026-07-12

**Status**: Draft

**Input**: User description: "ARCHIE-123: LLM генерирует несуществующие (галлюцинированные) URL в финальных ответах на Stage 3 пайплайна, без какой-либо валидации. Невалидные ссылки появляются в трёх местах: FrontendButton.url, LocationCard.open_map_url, TextAnswer.text (markdown-ссылки). Нужно добавить асинхронный валидатор URL, подключаемый в create_output_tool.py после build_content_from_parsed(). Логика: regex-проверка формата → HTTP HEAD-запрос → если невалиден и это YouTube/Amazon/Web-команда, попытаться найти замену через google_search_tool → иначе конвертировать/обнулить/убрать ссылку. Параллельно через asyncio.gather, WARNING-логирование, unit-тесты."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Broken link is removed from a chat button (Priority: P1)

A user asks the assistant to open a YouTube video, website, or Amazon product.
The assistant's underlying model hallucinates a URL that does not actually
exist. Today the user sees a working-looking button that leads to a dead
link when tapped. Instead, the user should either land on a real, working
resource (found via a fallback search) or see a button that lets them ask
the assistant to help find the right thing — never a dead link presented as
if it works.

**Why this priority**: This is the most visible and most damaging failure
mode — a user actively taps a button expecting content and hits a broken
page, which directly erodes trust in the assistant.

**Independent Test**: Trigger a response containing a button with a
deliberately broken/non-existent URL and command type in scope (e.g. "open
this YouTube video"); verify the user never receives a button pointing to a
dead link — either the link is replaced with a working one, or the button is
turned into a request the user can send back to the assistant.

**Acceptance Scenarios**:

1. **Given** the assistant generated a button with a valid, reachable URL,
   **When** the response is finalized, **Then** the button and its URL are
   shown to the user unchanged.
2. **Given** the assistant generated a button with a URL that does not exist
   or is unreachable, **When** a working alternative can be found, **Then**
   the user sees the button pointing to the alternative instead.
3. **Given** the assistant generated a button with a broken URL and no
   alternative can be found, **When** the response is finalized, **Then** the
   user sees a button that lets them ask the assistant about their original
   request, instead of a dead link.

---

### User Story 2 - Broken map link is not shown (Priority: P2)

A user asks about a place (restaurant, venue, address) and the assistant
attaches a "view on map" link that turns out to be hallucinated or broken.
Instead of a link that fails to open a map, the user should simply not see a
map link at all for that place.

**Why this priority**: Map links are secondary to the core information (name,
address, description) already present in the response; a missing map link is
a minor inconvenience, not a broken promise, unlike a dead button (P1).

**Independent Test**: Trigger a location-type response with a deliberately
broken map URL; verify the map link is absent from what the user sees while
the rest of the location information remains intact.

**Acceptance Scenarios**:

1. **Given** a location response with a valid, reachable map URL, **When**
   the response is finalized, **Then** the user sees the map link unchanged.
2. **Given** a location response with a broken or non-existent map URL,
   **When** the response is finalized, **Then** the user sees the location
   information without any map link.

---

### User Story 3 - Broken link inside message text does not mislead the user (Priority: P3)

A user receives a chat message where the assistant's text includes an inline
link (e.g. "check out [this article](url)") and that link is hallucinated.
Instead of a clickable link that fails, the user should see the surrounding
text without a misleading clickable link, or with a link that has been
replaced with a real, working one.

**Why this priority**: Inline text links are the least common of the three
occurrences and least likely to be tapped compared to a dedicated button, so
the risk to user trust is lower.

**Independent Test**: Trigger a text response containing a deliberately
broken markdown link; verify the user sees the surrounding sentence intact,
either with the link removed (plain text) or replaced with a working
alternative.

**Acceptance Scenarios**:

1. **Given** message text with a valid, reachable link, **When** the response
   is finalized, **Then** the user sees the text with the link unchanged.
2. **Given** message text with a broken or non-existent link, **When** a
   working alternative can be found, **Then** the user sees the text with the
   link pointing to the alternative.
3. **Given** message text with a broken link and no alternative found,
   **When** the response is finalized, **Then** the user sees the plain text
   without a clickable link.

---

### Edge Cases

- What happens when the link-checking service itself is slow or unreachable
  (network issue, target host down)? The response MUST still reach the user
  without unacceptable delay — a single broken/unreachable link check cannot
  block or noticeably slow down the whole response.
- What happens when a response contains multiple links to check? All checks
  happen without materially adding to the time the user waits for a
  response, regardless of how many links are present.
- What happens when a fallback search finds a result that is itself
  unreachable? The user must never receive a still-broken link — only a
  verified-reachable link or no link at all.
- What happens when the response contains no links at all? The response is
  unaffected and delivered exactly as generated.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST verify that every URL presented to the user in
  a chat button, map link, or inline message link actually leads to an
  existing, reachable resource before the response is delivered.
- **FR-002**: When a button's URL is broken and the button's intent is to
  open a video, a shopping item, or a general website/navigation target, the
  system MUST attempt to find a working replacement resource matching the
  user's original intent before giving up.
- **FR-003**: When no working replacement can be found for a broken button
  URL, the system MUST turn the button into one that lets the user ask the
  assistant about the original request, rather than presenting a dead link.
- **FR-004**: When a map link is broken, the system MUST omit the map link
  from the response while preserving the rest of the location information.
- **FR-005**: When an inline text link is broken and no working replacement
  is found, the system MUST present the surrounding text without a clickable
  link, rather than a link that fails.
- **FR-006**: The system MUST perform link verification in a way that does
  not noticeably increase how long a user waits for a response, regardless of
  how many links a given response contains.
- **FR-007**: The system MUST record every case where a link was replaced,
  removed, or converted, so the behavior can be reviewed after the fact.
- **FR-008**: Responses that contain no links, or only links that are already
  valid and reachable, MUST be delivered unchanged.

### Key Entities

- **Chat Button**: An interactive element in a response that either opens an
  external resource (video, shopping item, website/map) or, when no valid
  target exists, sends a follow-up request back to the assistant on the
  user's behalf.
- **Location Reference**: A place mentioned in a response (name, address,
  description) that may optionally include a link to view it on a map.
- **Message Link**: A clickable reference embedded directly in the
  assistant's written response text.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of links shown to users in buttons, map references, or
  message text lead to a reachable resource at the time the response is
  delivered.
- **SC-002**: 0% of user-facing buttons point to a dead link — every broken
  button is either replaced with a working link or converted into a
  follow-up request.
- **SC-003**: Adding link verification changes total response time by no
  more than a small, unnoticeable margin for the user, regardless of how many
  links are present in a response.
- **SC-004**: Every link replacement, removal, or conversion is traceable
  after the fact for review and debugging.

## Assumptions

- Determining whether a URL "exists" is based on whether the resource is
  currently reachable at response time; a resource that later goes offline is
  out of scope for this feature.
- A "working replacement" for a broken video/shopping/website link is found
  using the assistant's existing web search capability; no new external
  search integration is introduced.
- This feature applies only to links generated for the user-facing final
  response (Stage 3 output); links appearing elsewhere (e.g. debug/internal
  logs) are out of scope.
- Existing responses that contain no links, or only already-valid links, see
  no behavior change.
