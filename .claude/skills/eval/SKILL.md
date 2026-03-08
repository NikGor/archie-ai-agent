---
name: eval
description: Run the Archie agent eval suite to measure routing accuracy, tool selection, and output quality. Use to benchmark a model, compare two models (A/B), check for regressions, or run LLM-as-judge quality scoring.
argument-hint: "suite name, model, comparison target, or judge flag (e.g. 'ui_answer --judge', 'routing', 'gpt-4.1', 'routing vs gpt-4.1')"
disable-model-invocation: true
---

Run `.venv/bin/python -m scripts.eval.eval_runner` with appropriate flags.

## Suites

| Suite | What it tests | Stage |
|---|---|---|
| `routing` | `action_type` classification (final_response vs function_call) | stage1 |
| `tool_calls` | Correct tool selected for request | stage1 |
| `output_format` | Content non-empty, token/latency budgets | full pipeline |
| `ui_answer` | Real-world ui_answer scenarios: UI components, forms, quick actions + LLM judge | full pipeline |

## Common commands

**Run all suites (default model gpt-4.1-mini):**
```
.venv/bin/python -m scripts.eval.eval_runner
```

**Run one suite:**
```
.venv/bin/python -m scripts.eval.eval_runner --suite routing
```

**Run ui_answer suite with LLM judge:**
```
.venv/bin/python -m scripts.eval.eval_runner --suite ui_answer --judge
```

**Use a specific model:**
```
.venv/bin/python -m scripts.eval.eval_runner --suite routing --model gpt-4.1
```

**Use a stronger judge model:**
```
.venv/bin/python -m scripts.eval.eval_runner --suite ui_answer --judge --judge-model gpt-4.1
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

- Rich table in terminal: pass/fail per case, tokens, latency per stage, UI item count, judge scores (UI / Inf / Prm)
- Summary line with mean judge scores and per-case reasoning (when `--judge` enabled)
- JSON saved to `eval_results/YYYY-MM-DD_HH-MM_<suite>_<model>.json`

## Interpreting results

- **PASS** — all assertions in `expect:` block satisfied
- **FAIL** — shows which assertion failed and expected vs actual value
- **ERROR** — exception during case execution (check API key, model name)
- Failures on `action_type: final_response` expected `function_call` → model requests parameters instead of calling tool → improve system prompt

### Judge scores (0–10, only with `--judge`)

| Column | Metric | What it measures |
|--------|--------|-----------------|
| UI | `ui_optimality` | Right component types chosen? (weather_card, location_card, forms vs generic) |
| Inf | `content_informativeness` | Is the content useful and complete? |
| Prm | `prompt_following` | quick_actions present, forms rendered for creation, no walls of text |

Scores are **per-case** in the table and **averaged** in the summary line.
Per-case judge reasoning is printed below the summary for debugging.

## Adding test cases

Edit files in `scripts/eval/cases/`. YAML format:
```yaml
- name: "descriptive name"
  input: "User message"
  response_format: plain          # plain | level2_answer | level3_answer | ui_answer | dashboard | widget
  stage: stage1                   # stage1 | full
  judge: true                     # optional: enable LLM-as-judge (ui_answer only)
  expect:
    action_type: function_call    # stage1 only
    tools_called: [tool_name]     # stage1 only
    content_not_empty: true       # full only
    max_total_tokens: 1000        # full only
    max_latency_ms: 15000         # full only
    text_contains: "substring"    # full only
    # ui_answer quality checks (full only):
    min_ui_items: 2               # minimum number of UI components in response
    expected_item_types: [weather_card, card_grid]  # component types that must appear
    has_quick_actions: true       # response must include quick_action_buttons
    has_form: true                # response must include event_form / note_form / email_form
```
