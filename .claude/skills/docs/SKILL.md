---
name: docs
description: Look up current library documentation via context7 MCP before implementing features that depend on external libraries
argument-hint: [library name]
---

Use context7 MCP to get up-to-date docs for `$ARGUMENTS` (or the relevant library).

**Usage pattern:**
1. `mcp__context7__resolve-library-id` — get context7 ID for the library
2. `mcp__context7__query-docs` — retrieve docs for a specific topic

**Key libraries in this project:**
- `fastapi` — web framework (`main.py`, `endpoints.py`)
- `pydantic` v2 — data models (`app/models/`)
- `openai` — OpenAI SDK (`app/utils/llm_parser.py`)
- `httpx` — async HTTP client (`app/backend/`)
- `redis` — state storage (`app/backend/state_service.py`)
- `jinja2` — prompt templates (`app/agent/prompts/`)
- `google-genai` — Gemini LLM
- `google-api-python-client` — Google API interface
