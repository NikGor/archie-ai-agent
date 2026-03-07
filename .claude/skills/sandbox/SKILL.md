---
name: sandbox
description: Run sandbox scripts to test LLM provider API syntax, response structure, and parser compatibility. Use when debugging provider integration, testing a new model, or validating parser output. Scripts live in scripts/.
argument-hint: "provider or script name (openai | openrouter | gemini | qwen | pipeline | models)"
allowed-tools: Bash(PYTHONPATH=. poetry run python scripts/* *)
---

Sandbox scripts for testing provider APIs and internal clients.
All run from repo root with: `PYTHONPATH=. poetry run python scripts/<script>.py`

## Scripts

| Script | What it tests | Key args |
|--------|--------------|----------|
| `openai_sample.py` | OpenAI Responses API (`client.responses.parse`) + `DecisionResponse` + `parse_llm_response` | — |
| `openai_client_sample.py` | Internal `OpenAIClient.create_completion`: text / structured / tools / mixed | — |
| `openrouter_sample.py` | Raw OpenRouter API, multi-model, full response struct dump + `parse_openrouter_response` | `[model_id]` to test one model |
| `openrouter_client_sample.py` | Internal `OpenRouterClient.create_completion`: text / structured / tools | `[model_id]` to test one model |
| `gemini_sample.py` | Gemini native SDK + `parse_llm_response` | — |
| `full_pipeline_test.py` | E2E `AgentFactory.arun()` for OpenAI + Gemini (standard + thinking) | — |
| `list_models.py` | List available OpenAI models (gpt-4/o-series) | — |
| `list_gemini_models.py` | List available Gemini models | — |
| `qwen_sample.py` | Qwen3 via HuggingFace router + instructor | — |

## Picking the right script

- **New model on OpenRouter** → `openrouter_sample.py <model_id>` — shows raw response structure
- **Internal client broken** → `openai_client_sample.py` or `openrouter_client_sample.py`
- **Parser mismatch** → run provider-specific sample; output shows `parse_*_response` result
- **Full pipeline smoke** → `full_pipeline_test.py`
- **What models are available** → `list_models.py` (OpenAI) or `list_gemini_models.py`

## Run examples

```bash
# Test all OpenRouter models
PYTHONPATH=. poetry run python scripts/openrouter_sample.py

# Test specific OpenRouter model
PYTHONPATH=. poetry run python scripts/openrouter_sample.py anthropic/claude-sonnet-4

# Test OpenAI client wrapper (all 4 scenarios)
PYTHONPATH=. poetry run python scripts/openai_client_sample.py

# Full pipeline E2E
PYTHONPATH=. poetry run python scripts/full_pipeline_test.py
```

## Output interpretation

- `✅ model — SUCCESS` / `❌ model — FAILED: ...` — per-model result in openrouter_sample
- `Parser OK: {...}` — parser returned correct types and token counts
- `LLM Trace: model=..., in=..., out=...` — verify token accounting
- Any `❌ Error:` — print full traceback and investigate

Based on `$ARGUMENTS`, select and run the appropriate script(s). Report: what was tested, pass/fail per model, any errors with context.
