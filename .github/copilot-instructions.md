# Archie AI Agent - Copilot Instructions

## Project Overview
This is a sophisticated AI agent service built with FastAPI that implements **Schema-Guided Reasoning (SGR)** for structured, verifiable responses. The agent supports multiple personas and integrates with an external backend for conversation persistence.

## Architecture Components

### Core Request Flow
1. **FastAPI Entry** (`main.py` → `endpoints.py`) - Single `/chat` endpoint
2. **API Controller** (`api_controller.py`) - Manages backend integration and conversation history  
3. **Agent Builder** (`agent_builder.py`) - Constructs prompts using Jinja2 templates
4. **OpenAI Client** (`openai_client.py`) - Handles structured outputs with SGR trace
5. **External Backend** - Separate service at `BACKEND_API_URL` for data persistence

### Key Architectural Patterns

**SGR (Schema-Guided Reasoning)**: Every response includes structured reasoning trace with routing decisions, evidence mapping, and verification status. See `SGRTrace` model in `response_understand.py`.

**Persona System**: Dynamic personality injection via Jinja2 templates in `app/prompts/persona_*.jinja2`. Currently supports: `business`, `flirty`, `futurebot`, `bro`.

**Rich Metadata Response**: Responses include structured UI elements (cards, buttons, tables) alongside plain text. Never duplicate metadata content in response text.

**Microservice Integration**: This agent service communicates with external backend via HTTP for conversation persistence and history retrieval.

## Development Workflow

### Essential Commands (use Poetry)
```bash
make dev          # Development server with reload
make chat         # Console interface for testing  
make api-test     # Quick API validation
make format       # Black + Ruff auto-formatting
```

### Configuration 
Environment variables in `.env`:
- `OPENAI_API_KEY` - Required for AI responses
- `DEFAULT_PERSONA` - Controls agent personality (business/flirty/futurebot/bro)  
- `BACKEND_API_URL` - External backend service URL (default: http://localhost:8002)

### Logging Convention
Follow structured logging pattern with module prefixes and ANSI colors:
```python
logger.info(f"module_001: Description with \033[36mvariable\033[0m")  # Cyan for IDs/URLs
logger.info(f"module_002: Count: \033[33m{count}\033[0m")            # Yellow for numbers
logger.error(f"module_error_001: \033[31m{error}\033[0m")            # Red for errors
```

## Code Patterns & Conventions

### Model Design
- **Pydantic BaseModel** for all data structures with Field descriptions
- **Type unions** with `|` syntax (Python 3.10+)
- **Structured responses** via `AgentResponse` with mandatory `sgr` trace
- **UI metadata** separate from response text to avoid duplication

### File Organization
- `app/models.py` - Shared Pydantic models (ChatMessage, Metadata, UI components)
- `app/prompts/` - Jinja2 templates for personas and system prompts
- `app/tools.py` - Async tool functions (currently weather API)

### Key Integration Points
**Backend Communication**: Uses httpx async client with detailed request/response logging. Handles conversation creation, history loading, and message persistence.

**OpenAI Integration**: Uses structured outputs with `client.responses.parse()` and `text_format=AgentResponse`. Supports tool calling and web search.

**State Management**: `app/state.py` provides user context (timezone, preferences, current date) injected into prompts via `assistant_prompt.jinja2`.

## Testing & Debugging

### Quick Testing
- `make chat` - Interactive console testing with full agent pipeline
- `make api-test` - Curl-based API validation  
- Check logs for detailed request/response traces with color coding

### Common Issues
- **Persona not loading**: Check `app/prompts/persona_*.jinja2` exists and matches `DEFAULT_PERSONA`
- **Backend connection**: Verify external backend service is running at `BACKEND_API_URL`
- **OpenAI errors**: Ensure `OPENAI_API_KEY` is set and model name is correct (currently `gpt-4.1`)

## Extension Points

### Adding New Personas
1. Create `app/prompts/persona_name.jinja2` with personality description
2. Update persona logic in `main_agent_prompt.jinja2` template  
3. Test with `DEFAULT_PERSONA=name` in `.env`

### UI Components
Use structured metadata models from `app/models.py`:
- `Card` with `ButtonOption` for interactive elements
- `NavigationCard` for locations (fixed buttons: show_on_map, route)  
- `ContactCard` for people (fixed buttons: call, email, message)
- `Table`/`Elements` for structured data display

Keep responses mobile-friendly and action-oriented with minimal text in main response field.

## Code Style Guidelines

### Python Standards & Formatting
- Use **PEP8** for all Python code (formatting, naming, imports, etc.)
- Apply code checks from the Makefile immediately after writing code
- Always use **type annotations** for function signatures and variables (except where type is None)
- When formatting a Class, function, or method definition with multiple parameters, place each parameter on a new line for readability
- The same applies to function calls with multiple arguments

### String Formatting & Documentation
- Use **f-strings** for string formatting
- Use `'''` and `"""` for multi-line string literals and docstrings; use `"` for single-line strings only
- Keep docstrings and comments concise: one-liners or short comments only

### Data Models & Type Safety
- Use **Pydantic** (`BaseModel`) for all data models and API schemas
- Avoid using `Any`, `None`, or `Dict` types in models
- Don't check if fields exist in Pydantic models; access them directly
- Use variables of primitive or Pydantic types as parameters instead of Classes, instances, or functions

### Async Programming & Performance
- Prefer **async/await** for all I/O and database operations
- Use parallelism (e.g., `asyncio.gather`) for concurrent tasks

### Code Organization & Architecture
- Organize code by responsibility: endpoints → business logic → data/models
- Prefer minimalistic, simple solutions over complex ones
- Avoid if-elif-elif-else chains; use dictionaries or polymorphism instead
- Don't overengineer: avoid unnecessary abstractions, files, or classes for small tasks
- Use Single Responsibility Principle: each function/class should have one clear purpose
- Do not not follow the DRY principle too strictly; if it's a standard and readable code, it's okay to repeat it
- Use standard, simple verbs for function names (e.g., `get_`, `create_`, `update_`, etc.)
- Use simple english for variable and function names; avoid abbreviations or acronyms
- I prefer following namespace for module names:
    - `app` as the main directory for all application code
    - `main.py` as the entrypoint: a few code lines to start the FastAPI app only.
    - `endpoints.py` for all API routes: facade and minimal architecture.
    - `api_controller.py` for logic that connects endpoints to services (optional). Facade and minimal architecture.
    - `..._tool.py` for utility modules (e.g., `text_tool.py`, `db_tool.py`)
    - `..._service.py` for business logic and service layer (e.g., `user_service.py`, `message_service.py`)
    - `..._utils.py` for helper functions (e.g., `string_utils.py`, `date_utils.py`). Divide utils into classes by responsibility.

### Import Management
- Don't use wildcard imports (`from x import *`)
- Follow import hierarchy: standard library → third-party → local modules:
    ```python
    import os
    from openai import OpenAI
    from .models import User, Message
    ```

### What NOT to Do
- **Don't** use global variables for state or configuration
- **Don't** use print statements for logging (use `logger` only, per logging guidelines)
- **Don't** hardcode values; use config, env vars, or constants
- **Don't** overuse if-checks; prefer exceptions for error handling
- **Don't** divide code with empty lines into sections; keep related code together
- **Don't** use __init__.py files to import modules or to another code
