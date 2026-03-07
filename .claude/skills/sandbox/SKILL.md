---
name: sandbox
description: Run sandbox scripts to test LLM provider API syntax, response structure, and parser compatibility. Use when debugging provider integration, testing a new model, or validating parser output.
argument-hint: "provider or script (openai | openrouter [model_id] | gemini | qwen | pipeline | models)"
disable-model-invocation: true
allowed-tools: Bash(PYTHONPATH=. poetry run python *)
---

Sandbox scripts for testing provider APIs, internal clients, and the full agent pipeline.
All scripts live in `${CLAUDE_SKILL_DIR}/scripts/`.

## Scripts

| Script | Tests | Args |
|--------|-------|------|
| `openai_sample.py` | OpenAI Responses API (`client.responses.parse`) + `DecisionResponse` + `parse_llm_response` | — |
| `openai_client_sample.py` | Internal `OpenAIClient.create_completion`: text / structured / tools / mixed | — |
| `openrouter_sample.py` | Raw OpenRouter API, multi-model, full response struct dump + `parse_openrouter_response` | `[model_id]` |
| `openrouter_client_sample.py` | Internal `OpenRouterClient.create_completion`: text / structured / tools | `[model_id]` |
| `gemini_sample.py` | Gemini native SDK + `parse_llm_response` | — |
| `full_pipeline_test.py` | E2E `AgentFactory.arun()`: OpenAI + Gemini (standard + thinking) | — |
| `list_models.py` | Available OpenAI models (gpt-4/o-series) | — |
| `list_gemini_models.py` | Available Gemini models | — |
| `qwen_sample.py` | Qwen3 via HuggingFace router + instructor | — |

## Pick the right script based on $ARGUMENTS

- **`openai`** → run `openai_sample.py`, then `openai_client_sample.py`
- **`openrouter`** or **`openrouter <model_id>`** → run `openrouter_sample.py` (optionally with model arg), then `openrouter_client_sample.py`
- **`gemini`** → run `gemini_sample.py`
- **`qwen`** → run `qwen_sample.py`
- **`pipeline`** → run `full_pipeline_test.py`
- **`models`** → run `list_models.py` and `list_gemini_models.py`
- **no args or `all`** → run all provider samples + full pipeline

## Run commands

```bash
# OpenAI
PYTHONPATH=. poetry run python ${CLAUDE_SKILL_DIR}/scripts/openai_sample.py
PYTHONPATH=. poetry run python ${CLAUDE_SKILL_DIR}/scripts/openai_client_sample.py

# OpenRouter — all models
PYTHONPATH=. poetry run python ${CLAUDE_SKILL_DIR}/scripts/openrouter_sample.py
# OpenRouter — one model
PYTHONPATH=. poetry run python ${CLAUDE_SKILL_DIR}/scripts/openrouter_sample.py anthropic/claude-sonnet-4

# Gemini
PYTHONPATH=. poetry run python ${CLAUDE_SKILL_DIR}/scripts/gemini_sample.py

# Full pipeline E2E
PYTHONPATH=. poetry run python ${CLAUDE_SKILL_DIR}/scripts/full_pipeline_test.py

# List available models
PYTHONPATH=. poetry run python ${CLAUDE_SKILL_DIR}/scripts/list_models.py
PYTHONPATH=. poetry run python ${CLAUDE_SKILL_DIR}/scripts/list_gemini_models.py
```

## Output interpretation

- `✅ model — SUCCESS` / `❌ model — FAILED: ...` — per-model in openrouter_sample
- `Parser OK: {parsed_type, model, input_tokens, output_tokens}` — parser returned correct types
- `LLM Trace: model=..., in=..., out=...` — verify token accounting
- Any `❌ Error:` — print full traceback and investigate root cause

Report: what was tested, pass/fail per script/model, any errors with full context.
