# Archie AI Agent

[![CI](https://github.com/NikGor/archie-ai-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/NikGor/archie-ai-agent/actions/workflows/ci.yml)

The AI core of Archie, a personal home assistant. A FastAPI service that answers in plain text,
voice-ready text, or generative UI (cards, tables, charts, maps, forms, smart-home widgets),
controls the home, and runs a multi-agent swarm. It runs in Docker on a Raspberry Pi next to the
Django web app ([NikGor/archie-web](https://github.com/NikGor/archie-web)), which proxies
requests, stores conversations, and renders the UI.

## Features

- **3-stage SGR pipeline**: Schema-Guided Reasoning with typed Pydantic outputs at each stage
  (decide → run tools in parallel → format the answer).
- **Generative UI**: answers as `level2_answer` (text + quick actions), `level3_answer`
  (text + widget), `ui_answer` (ordered items: cards, tables, Chart.js charts, maps, images,
  email/event/note forms, light and climate cards), `dashboard` and `widget`.
  The `ui_answer` schema is trimmed per request by the intents found in Stage 1 to save tokens.
- **Streaming over WebSocket**: pipeline status events, text chunks, UI deltas, final message.
- **Multiple LLM providers**: OpenAI (GPT-4.1, GPT-5.x) and OpenRouter (Gemini, Claude, Grok);
  the model can be set per stage per request. Token cost tracking, prompt caching.
- **Multimodal input**: attached images (vision models) and documents (PDF, DOCX).
- **Personas**: `business`, `bro`, `butler`, `flirty`, `futurebot`.
- **Voice**: `gemini_tts` format with TTS-friendly text; `announce_tool` speaks messages at home.
- **Smart home**: lights, climate and scenes via the Home Assistant MCP server; Spotify; Android TV
  with IPTV channels.
- **Productivity**: Google Calendar events, tasks, notes, read-only Gmail, Google web and places
  search, document search (OpenAI vector store).
- **Scheduler**: `cron_tool` defers any tool call (one-shot or cron expression); jobs live in Redis.
- **Runtime skills**: instruction packs the LLM loads on demand (`football`, `homeassistant-mcp`,
  `spotify`, `start-my-day`).
- **Agent swarm**: a group chat of specialist agents (`@light`, `@climate`, `@football`,
  `@search`, `@voice`) led by a council of personas that discusses, decides and delegates.
- **Evals**: case catalog with assertions and an LLM-as-judge, run from the Django eval UI with
  live progress.
- **Chat monitoring**: an LLM judge grades stored chat turns (red/orange/green) against the rules
  of their prompt specs; results go to Django and Langfuse.
- **Tracing**: one Langfuse trace per chat turn (Stage 1 generation, tool observations, output).
- **Declarative prompts**: YAML prompt specs compiled from JSON rule catalogs.

## Architecture

```
Browser ──WS /ws_chat──────────────────────────► archie-ai-agent :8005
Browser ──HTTP──► Django ──POST /chat──────────► archie-ai-agent
                                                    │
                     Redis (UserState, cron jobs, swarm) ◄─┤
                     Home Assistant MCP, Google, Spotify, TV ◄─┤
                     OpenAI / OpenRouter ◄─┤
```

Request flow: `main.py → app/endpoints.py → app/api_controller.py → AgentFactory.arun()`

1. **Stage 1 — Command** (`app/agent/agent_factory.py`, `command_loop.py`): the LLM returns a
   `DecisionResponse` with `function_call`, `parameters_request` or `final_response`, plus
   `intents`. Up to `MAX_COMMAND_ITERATIONS` rounds.
2. **Stage 2 — Execution** (`app/utils/tool_executor.py`): tools run in parallel via
   `asyncio.gather` and return `ToolResult`.
3. **Stage 3 — Output** (`app/tools/output/`): the answer is built in the requested
   `response_format` (`output_models.py`), then post-processed and repaired.

`dashboard` and `widget` skip Stage 1.

Shared request/response models (`ChatRequest`, `ChatMessage`, `Content`, UI elements, `UserState`)
come from the `archie-shared` package, installed from the Django repo by git tag.

### Project layout

| Path | Contents |
|------|----------|
| `app/agent/` | Orchestration, prompt builder and compiler, prompt specs (`prompts/specs`) and rule catalogs (`prompts/rules`) |
| `app/tools/` | Tools (one file per tool); `output/` — Stage 3 |
| `app/backend/` | Clients: OpenAI, OpenRouter, Redis, Google Calendar, Gmail, Spotify, Android TV, HA MCP; cron scheduler |
| `app/models/` | Stage 1/3 models, `ToolResult`, WS events |
| `app/utils/` | LLM parser, provider routing, intent config and schema filter, tracing, status messages |
| `app/skills/` | Runtime skills for the LLM |
| `app/swarm/` | Agent swarm: bus, dispatcher, council, runner, policy |
| `app/evals/` | Eval engine and API |
| `app/monitoring/` | Chat monitoring judge and API |
| `tests/` | Unit and smoke tests |
| `scripts/` | Developer scripts (not production code) |
| `docs/` | Guides, e.g. `tools-guide.md` |

## API

| Endpoint | Purpose |
|----------|---------|
| `POST /chat` | One chat turn (`ChatRequest` → `ChatMessage`); used by Django and system calls |
| `WS /ws_chat` | Streaming chat: one connection per message (send `ChatRequest`, receive `status`, `stream_*`, `final` or `error` events) |
| `GET /swarm/agents`, `POST /swarm/messages`, `WS /ws_swarm` | Agent swarm chat |
| `/evals/*` | Eval schema, cases validation, prompts, models, runs (+ WS progress) |
| `/monitoring/runs` | Start and read chat monitoring runs |

Docs while the server runs:
- Swagger: `http://localhost:8005/docs`
- WebSocket protocol: `http://localhost:8005/ws_docs`

Example:

```bash
curl -X POST http://localhost:8005/chat \
  -H "Content-Type: application/json" \
  -d '{"input": "What is the weather today?", "user_name": "niko", "response_format": "ui_answer"}'
```

## Tools

Registered in `TOOLS_CONFIG` (`app/config.py`) by group; the group decides which tools a response
format can use.

| Group | Tools |
|-------|-------|
| `smarthome` | `smarthome_control_tool`, `smarthome_status_tool`, `spotify_tool` |
| `tv` | `tv_tool` |
| `internet_search` | `google_search_tool`, `google_places_search_tool` |
| `productivity` | `task_tool`, `notes_tool`, `events_tool`, `gmail_tool`, `cron_tool`, `announce_tool` |
| `knowledge` | `document_search_tool` |
| `skills` | `skill_loader_tool` |

How to add a tool: `docs/tools-guide.md`.

## Getting started

Requirements: Python 3.11, Poetry, Redis, Docker (for deployment).

```bash
make install            # poetry install
cp .env.example .env    # fill in the keys
make dev                # uvicorn with reload
```

> `make run` / `make dev` listen on port 8000; Docker and `python main.py` use 8005.

Docker (needs the external `shared_network`):

```bash
docker network create shared_network   # once
make docker-build
make docker-run
```

### Environment variables

See `.env.example`. Main groups:

```
OPENAI_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY
REDIS_HOST / REDIS_PORT / REDIS_DB
HOMEASSISTANT_MCP_URL / HOMEASSISTANT_MCP_TOKEN
OPENAI_VECTOR_STORE_ID
TV_HOST / TV_CERT_FILE / TV_KEY_FILE / TV_PLAYLIST_PATH / SHARAVOZ_TOKEN
CRON_POLL_INTERVAL / CRON_MAX_JOBS_PER_TICK
DEFAULT_PERSONA / DEFAULT_USER_NAME / DEFAULT_CITY / DEFAULT_LANGUAGE / ...
LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL / LANGFUSE_TRACING_ENVIRONMENT
EVAL_DJANGO_URL / EVAL_API_TOKEN
MONITOR_DJANGO_URL / MONITOR_JUDGE_MODEL
```

Gmail uses `token_gmail.json`; Spotify auth: `scripts/spotify_auth_setup.py`.

## Development

```bash
./execute_tests.sh                   # all tests
./execute_tests.sh -m "not llm"      # unit tests, no API keys
./execute_tests.sh -m llm            # smoke tests, needs .env
make check                           # black --check, ruff, mypy
make fix                             # ruff --fix, format
```

### Prompts and personas

Prompts are YAML specs in `app/prompts/specs/` (`cmd_prompt`, `assistant_prompt`,
`formats/format_*`, `personas/persona_*`) that list rules from JSON catalogs in
`app/prompts/rules/`; `prompt_compiler.py` builds them. Swarm prompts are Jinja2
(`prompts/swarm/`).

To add a persona: add `yourname_*` rules to `rules/personas.json` and a spec
`specs/personas/persona_yourname.yaml`. Run evals after any prompt change.

## Roadmap

- [ ] Authentication and authorization
- [ ] Plugin system for custom tools
- [ ] Conversation export/import
