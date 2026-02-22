---
description: Look up current library documentation via context7 MCP
---

Use context7 MCP tools to get up-to-date documentation for any library used in this project.

**When to use**: before implementing features that depend on an external library — verify current API,
method signatures, and examples rather than relying on training data cutoff.

**Key libraries in this project:**
- `fastapi` — web framework (`main.py`, `endpoints.py`)
- `pydantic` v2 — data models (`app/models/`)
- `openai` — OpenAI SDK (`app/utils/llm_parser.py`, `app/utils/provider_utils.py`)
- `httpx` — async HTTP client (`app/backend/`)
- `redis` — state storage (`app/backend/state_service.py`)
- `jinja2` — prompt templates (`app/agent/prompts/`)
- `google-genai` —  gemini LLM 
- `google-api-python-client` — google api interface


**Usage pattern:**
1. Call `mcp__context7__resolve-library-id` with the library name to get its context7 ID
2. Call `mcp__context7__get-library-docs` with the ID and a topic to retrieve relevant docs
3. Use the returned documentation to inform implementation
