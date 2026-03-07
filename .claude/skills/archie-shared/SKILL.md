---
name: archie-shared
description: Work with the archie-shared package — understand its models, find what to import, request changes via JIRA.
argument-hint: "what you need — model name, feature, or change description"
---

`archie-shared` is a shared Pydantic models library used across the Archie ecosystem.

**Repo:** `https://github.com/NikGor/homeassistant.git`, subdirectory `archie-shared`
**Current version:** see `pyproject.toml` tag (e.g. `tag = "v0.1.60"`)
**Installed at:** `.venv/lib/python3.11/site-packages/archie_shared/`

---

## Package structure

### `archie_shared.chat.models`
Core chat and pipeline models:
| Class | Purpose |
|---|---|
| `ChatRequest` | Incoming request (input, response_format, user_name, models, etc.) |
| `ChatMessage` | Single message (role, content, llm_trace, pipeline_trace) |
| `Conversation` | Full conversation with messages and aggregated stats |
| `Content` | Message content — holds text or structured payload (see below) |
| `LllmTrace` | LLM usage: input/output tokens, cost, model |
| `StepTrace` | Timing + LLM trace for one pipeline stage |
| `PipelineTrace` | Aggregated trace for all 3 stages + total_ms |
| `PipelineStep` | Named step with status and duration_ms |

**`response_format` values** (on `ChatRequest` / `Content.content_format`):
`plain`, `markdown`, `html`, `ssml`, `json`, `csv`, `xml`, `yaml`, `prompt`, `python`, `bash`, `sql`, `regex`, `dockerfile`, `makefile`, `level2_answer`, `level3_answer`, `ui_answer`, `dashboard`, `widget`

### `archie_shared.ui.models`
UI component models (used in Content):
| Class | Purpose |
|---|---|
| `Content` | Top-level content wrapper (also exported from chat.models) |
| `Level2Answer` | Text + quick action buttons |
| `Level3Answer` | Text + widgets + quick actions |
| `UIAnswer` | Full UI elements |
| `Dashboard` | Tiles layout for smarthome dashboard |
| `Widget`, `LightWidget`, `ClimateWidget`, `FootballWidget`, `MusicWidget`, `DocumentsWidget` | Standalone widgets |
| `Card`, `LocationCard`, `ProductCard`, `MovieCard`, etc. | Rich content cards |
| `Table`, `Chart`, `Image` | Data display components |
| `FrontendButton`, `AssistantButton` | Action buttons |

### `archie_shared.user.models`
| Class | Purpose |
|---|---|
| `UserState` | User context for Redis — city, timezone, preferences, smarthome state |

---

## Where it's used in this project

| File | Imports |
|---|---|
| `app/api_controller.py` | `ChatMessage`, `ChatRequest` |
| `app/endpoints.py` | `ChatMessage`, `ChatRequest`, `Content` |
| `app/agent/agent_factory.py` | `LllmTrace`, `PipelineTrace` |
| `app/models/output_models.py` | `LllmTrace`, `PipelineTrace`, `Content`, UI models |
| `app/tools/create_output_tool.py` | `Content` |
| `app/utils/llm_parser.py` | `LllmTrace`, `StepTrace`, etc. |
| `app/utils/openai_utils.py` | `InputTokensDetails`, `LllmTrace`, `OutputTokensDetails` |
| `app/utils/trace_utils.py` | `LllmTrace`, `StepTrace` |

---

## Upgrading the version

After a new tag is released in the external repo, update `pyproject.toml`:
```
archie-shared = {git = "https://github.com/NikGor/homeassistant.git", subdirectory = "archie-shared", tag = "vX.Y.Z"}
```
Then run:
```bash
poetry update archie-shared
```

---

## Requesting changes to archie-shared

**Do NOT edit the package directly.** It lives in an external repo.

To request a change:
1. Use `mcp__jira-mcp__jira_post` to create a JIRA task in project **ARCHIE**
2. Task must include:
   - **Summary:** what needs to change in archie-shared
   - **Description:** exact model/field/class, why it's needed, what the change should be
   - **Affected modules** in this project that will use the new/changed model
   - **Acceptance criteria:** what the new version should export/expose
3. Set issue type: `task` or `story`

Example situations requiring a JIRA task:
- Adding a new field to `ChatMessage`, `UserState`, or any model
- Adding a new UI component class to `ui/models.py`
- Adding a new `response_format` value
- Fixing a typo in a class name (e.g. `LllmTrace` → `LlmTrace`)
- Adding new utility functions to `chat/utils.py`
