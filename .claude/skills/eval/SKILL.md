---
name: eval
description: Run the Archie agent eval suite to measure routing accuracy, tool selection, and output quality. Use to benchmark a model, compare two models (A/B), or check for regressions against a previous run.
argument-hint: "suite name, model, or comparison target (e.g. 'routing', 'gpt-4.1', 'routing vs gpt-4.1')"
disable-model-invocation: true
allowed-tools: Bash(.venv/bin/python -m scripts.eval.eval_runner *)
---

Run `.venv/bin/python -m scripts.eval.eval_runner` with appropriate flags.

## Suites

| Suite | What it tests | Stage |
|---|---|---|
| `routing` | `action_type` classification (final_response vs function_call) | stage1 |
| `tool_calls` | Correct tool selected for request | stage1 |
| `output_format` | Content non-empty, token/latency budgets | full pipeline |

## Common commands

**Run all suites (default model gpt-4.1-mini):**
```
.venv/bin/python -m scripts.eval.eval_runner
```

**Run one suite:**
```
.venv/bin/python -m scripts.eval.eval_runner --suite routing
```

**Use a specific model:**
```
.venv/bin/python -m scripts.eval.eval_runner --suite routing --model gpt-4.1
```

**A/B compare two models:**
```
.venv/bin/python -m scripts.eval.eval_runner --suite routing --model gpt-4.1-mini --compare gpt-4.1
```

**Diff against a previous run:**
```
.venv/bin/python -m scripts.eval.eval_runner --suite routing --diff eval_results/<previous>.json
```

## Output

- Rich table in terminal: pass/fail per case, tokens, latency per stage
- JSON saved to `eval_results/YYYY-MM-DD_HH-MM_<suite>_<model>.json`

## Interpreting results

- **PASS** — all assertions in `expect:` block satisfied
- **FAIL** — shows which assertion failed and expected vs actual value
- **ERROR** — exception during case execution (check API key, model name)
- Failures on `action_type: final_response` expected `function_call` → model requests parameters instead of calling tool → improve system prompt

## Adding test cases

Edit files in `scripts/eval/cases/`. YAML format:
```yaml
- name: "descriptive name"
  input: "User message"
  response_format: plain          # plain | level2_answer | level3_answer | ui_answer | dashboard | widget
  stage: stage1                   # stage1 | full
  expect:
    action_type: function_call    # stage1 only
    tools_called: [tool_name]     # stage1 only
    content_not_empty: true       # full only
    max_total_tokens: 1000        # full only
    max_latency_ms: 15000         # full only
    text_contains: "substring"    # full only
```
