---
name: prompt-edit
description: Edit Archie agent prompts safely — follow hierarchy, stay token-conscious, use general formulations. Invoke when asked to improve, fix, or tune any prompt file.
argument-hint: "symptom or goal (e.g. 'agent asks too many questions', 'ui_answer lacks quick actions', 'butler persona too formal')"
---

## Prompt Hierarchy (most specific → least specific)

Edit the **lowest-level** prompt that owns the problem. Do not touch higher-level prompts unless the fix genuinely belongs there.

| Priority | File | Owns |
|----------|------|------|
| 1st | `format_prompts/format_<X>.jinja2` | Format-specific output rules (ui_answer components, plain text style, etc.) |
| 2nd | `persona_prompts/persona_<X>.jinja2` | Tone, vocabulary, personality quirks |
| 3rd | `assistant_prompt.jinja2` | User context injection (state variables, response_format hint) |
| 4th | `main_agent_prompt.jinja2` | Stage 3 system prompt — SGR workflow, safety rails, output contract |
| **Last** | `cmd_prompt.jinja2` | Stage 1 routing — only if routing/tool-selection is broken |

**Rule:** if the symptom is "wrong UI component chosen" → edit `format_ui_answer.jinja2`. If it's "butler sounds too casual" → edit `persona_butler.jinja2`. If it's "agent calls the same tool twice" → that's `cmd_prompt.jinja2` (edit last, after ruling out everything else).

---

## Before Editing

1. **Identify the stage** — is the problem in Stage 1 (routing, tool calls) or Stage 3 (output content, format, tone)?
2. **Read the target prompt** in full before proposing changes.
3. **Locate the existing rule** that should cover the behavior — maybe it's missing, maybe it's ambiguous, maybe it contradicts another rule.
4. **Estimate token impact**: count added vs removed lines. A fix should not increase prompt length by more than ~20% unless the entire section is missing.
5. **Check for schema locks** — if the word you are changing also appears as a Pydantic `Literal` or `archie_shared` field name, it is schema-locked (see "Schema-Locked Strings" below).

---

## Editing Rules

### General formulations only
- Write instructions that apply to **any** request of that type, not just the one that triggered the edit.
- BAD: "When the user asks about the weather in Moscow, always use weather_card."
- GOOD: "For weather queries, always use `weather_card` component — never plain text or generic cards."

### Isolation
- Keep the fix in the prompt that owns it. Do not duplicate a rule across multiple prompts.
- If `cmd_prompt.jinja2` already says "don't repeat tool calls," don't add the same rule to `main_agent_prompt.jinja2`.

### Token proportionality
- A missing comma or wrong tone → 1–3 lines max.
- A missing behavioral rule → 1 concise bullet or sentence.
- A new section → only if the topic is entirely absent and genuinely needed.
- Never add examples inline unless behavior is complex and examples are the only clear way — prefer abstract rules.

### Primacy + recency (GPT-4.1 pattern)
- Critical constraints belong near the **top** of their section (primacy) and optionally echoed at the bottom as a one-liner (recency).
- Don't bury important rules in the middle of a long list.

### Explicit over implicit
- GPT-4.1 follows instructions literally. Write what you mean exactly.
- Replace vague words ("try to", "consider", "may") with direct verbs ("always", "never", "must", "do not").
- Use **MUST** / **NEVER** for hard constraints. Use plain verbs for soft guidance.

### Structured format
- Use numbered lists for ordered workflows, bullet lists for rules, `**bold**` for emphasis on key terms.
- Avoid prose paragraphs for rules — they get skipped. Use lists.

---

## Schema-Locked Strings

Some strings in prompts are bound to Pydantic `Literal` types or `archie_shared` schema field names. Changing them breaks validation silently — the LLM returns a value the parser rejects.

**Never rename or rephrase:**
- `function_call`, `parameters_request`, `final_response` — exact `action_type` literals in `DecisionResponse` (`app/models/orchestration_sgr.py`). They appear in `cmd_prompt.jinja2` and must stay verbatim.
- `quick_action_buttons`, `intro_text`, `items` — actual field names in the `ui_answer` schema. Do not paraphrase in format prompts.
- UI component type names: `weather_card`, `location_card`, `EventForm`, `NoteForm`, `EmailForm` — defined in `archie_shared`. Never invent new type names; request additions via `archie-shared` JIRA task.

**How to identify a locked string:** if a word in a prompt also appears as a Pydantic field name or `Literal` value in `app/models/` or `archie_shared/`, treat it as locked.

---

## Workflow

1. Read the relevant prompt(s) — never edit blind.
2. Identify the root cause: missing rule / ambiguous rule / wrong prompt level.
3. Write the minimal change that fixes the general case.
4. Check: does this change add tokens proportionally? Does it duplicate an existing rule?
5. Apply the edit with `Edit` tool (prefer targeted edits over rewrites).
6. State clearly: which file was changed, what was added/removed, and why this level.

---

## cmd_prompt.jinja2 — Special Rules

Edit this file **only** when:
- Stage 1 is choosing the wrong `action_type` (e.g., `final_response` when tool needed)
- Stage 1 is calling the wrong tool
- Stage 1 is repeating a tool call that already ran
- Stage 1 is asking for parameters it should infer

Do NOT edit `cmd_prompt.jinja2` for:
- Output format issues (→ format prompt)
- Tone/personality issues (→ persona prompt)
- Wrong UI components (→ format prompt)
- Missing content in response (→ main_agent_prompt or format prompt)

---

## assistant_prompt.jinja2 — Hard Constraint

This file is injected as an **assistant-role message**, not a system prompt. The LLM treats it as its own previous utterance — not as an instruction source.

**Never add behavioral instructions here.** They will be inconsistently followed or ignored. This file is a pure state dump (user name, date, persona, format hint, preferences). Keep it that way.

- If you need to add a rule → put it in `main_agent_prompt.jinja2` or the relevant format prompt.
- If you need to expose a new state variable → add the `{{ state.field }}` line here, no instructions attached.

---

## Eval Coverage — Do Not Weaken

The eval suite (`scripts/eval/cases/`) asserts specific behaviors. Weakening the following prompt rules will cause eval failures:

| Eval assertion | Owning prompt rule | File |
|---|---|---|
| `has_form: true` | "ALWAYS render Form for creation intents" | `format_ui_answer.jinja2` |
| `has_quick_actions: true` | "2-3 quick_action_buttons" | `format_ui_answer.jinja2` |
| `expected_item_types: [weather_card]` | weather → `weather_card` component rule | `format_ui_answer.jinja2` |
| `action_type: function_call` (smart home) | smart home inference rule + math examples | `cmd_prompt.jinja2` |
| `action_type: final_response` (greeting) | no-tool cases routing | `cmd_prompt.jinja2` |

**Rule:** if you are softening a "ALWAYS" / "NEVER" / "MUST" instruction, check whether an eval case asserts that behavior. Run `./execute_tests.sh -m llm` after any prompt edit that touches these areas.

The smart home math examples in `cmd_prompt.jinja2` ("decrease color_temp by 500K", "increase brightness by 20%") are load-bearing for inference quality — do not remove them.

---

## Token Budget Checklist

Before saving:
- [ ] Lines added ≤ lines removed + reasonable new coverage
- [ ] No duplicate rules across files
- [ ] No inline examples unless strictly necessary
- [ ] No vague qualifiers ("try", "consider", "may") — replace with direct verbs
- [ ] Critical rules are near the top of their section
