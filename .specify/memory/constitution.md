<!--
Sync Impact Report
- Version change: [TEMPLATE] → 1.0.0 (initial ratification)
- Modified principles: n/a (first fill from template placeholders)
- Added sections: Core Principles (I–VI), Quality & Workflow Constraints, Development Workflow, Governance
- Removed sections: none (template placeholders replaced)
- Templates requiring updates:
  - ✅ .specify/templates/plan-template.md — no agent-specific conflicts found, generic Constitution Check gate applies
  - ✅ .specify/templates/spec-template.md — no changes required, requirements format already generic
  - ✅ .specify/templates/tasks-template.md — task categorization already covers test-first and observability
  - ✅ .claude/skills/speckit-*/*.md — no outdated agent-name references found
  - ⚠ README.md — high-level only, no principle-specific claims to sync; left as-is
- Follow-up TODOs: TODO(RATIFICATION_DATE) resolved to today (first adoption); none deferred
-->

# Archie AI Agent Constitution

## Core Principles

### I. Staged Pipeline Integrity
The agent's request flow (`main.py → endpoints.py → api_controller.py →
AgentFactory.arun()`) MUST preserve its 3-stage structure: Stage 1 Command
(intent/decision via `DecisionResponse`), Stage 2 Execution (parallel tool
calls via `asyncio.gather`), Stage 3 Output (format-specific response
generation). Dashboard and widget response formats MAY bypass Stage 1 and go
directly to Stage 3, per existing design. Any change that blurs stage
boundaries (e.g., tool execution triggered from Stage 3, or output formatting
performed inside Stage 1) MUST be justified in the PR/commit description or
rejected.
**Rationale**: The staged design keeps intent parsing, side effects, and
presentation independently testable and swappable per LLM provider; collapsing
stages reintroduces coupling the architecture was built to avoid.

### II. Verification Over Assertion (NON-NEGOTIABLE)
Every diagnostic claim (bug cause, behavior explanation, fix confirmation)
MUST be grounded in a cited source: a specific log line, a DB/Redis record, an
API response field, an eval/test result, or an exact code path — never
reasoning alone. After any change, success MUST be confirmed by running
`./execute_tests.sh`, `make api-test`, an eval, or by reading logs; code
inspection alone is not a valid success claim.
**Rationale**: LLM-orchestration bugs are frequently non-obvious (provider
quirks, prompt drift, Redis state); unverified claims have historically led to
false fixes.

### III. Test-First Discipline
New features and bug fixes MUST include necessary and sufficient unit or smoke
tests (temporary or permanent) before the task is considered done. Tests run
via `./execute_tests.sh -m "not llm"` (no API keys) MUST pass without network
access; tests marked `llm` MAY require `.env` credentials and are run
separately. A task is not complete until its acceptance criteria are covered
by a passing test or an explicit, documented reason why one is impractical.
**Rationale**: The 7-step task workflow treats "Test" as a distinct, mandatory
stage separate from "Implement" — skipping it has historically hidden
regressions across LLM providers.

### IV. Reuse Before Creation
Before writing new logic, the codebase (especially `app/utils/`,
`app/tools/`, and the external `archie-shared` package) MUST be searched for
an existing implementation to reuse or extend. `archie-shared` is
externally versioned (git-tagged) and MUST NEVER be edited directly from this
repository — required changes are requested via a JIRA task against that
package and consumed after a version bump in `pyproject.toml`.
**Rationale**: Duplicated utility logic across providers/tools has been a
recurring source of drift; treating `archie-shared` as external-only keeps
its versioning and consumers in sync.

### V. Typed, Minimal, SOLID Code
All function signatures MUST carry type annotations (Python 3.11+ `|` union
syntax, not `Optional`), and data structures MUST be Pydantic `BaseModel`s
with `Field(description=...)` rather than bare `dict`/`Any`. Code MUST follow
single-responsibility and clean-code principles without overengineering:
no speculative abstractions, no handling for scenarios that cannot occur, no
feature flags or compatibility shims where the code can simply be changed.
Exceptions are used in low-level functions and handled at higher levels;
bare `except Exception` without logging the specific error is prohibited.
**Rationale**: Strict typing and Pydantic validation catch provider-response
drift early; minimalism keeps a multi-provider, multi-persona codebase
reviewable.

### VI. Structured Observability
All runtime diagnostics MUST go through `logger` — never `print()`. Log
messages MUST use the project's `module_NNN:` prefix and ANSI color
convention (cyan IDs/URLs, yellow counts, magenta names, red errors) as
defined in `agent_docs/logging.md`. Every non-obvious fix or workaround MUST
carry a short inline comment referencing its JIRA ticket (`# ARCHIE-XXXX`).
**Rationale**: Consistent, greppable logs are the primary tool for diagnosing
issues across Stage 1–3 and across LLM providers in production.

## Quality & Workflow Constraints

- **Provider parity**: any change to `app/utils/llm_parser.py` or provider
  routing (`app/utils/provider_utils.py`) MUST be validated against every
  active provider listed in `PROJECT_STATE.md`, not just the one under active
  development.
- **State safety**: Redis/DB mutations (`app/backend/state_service.py`,
  `UserState`) MUST be safe to retry or replay; no mutation may assume
  exactly-once delivery.
- **Prompt changes**: any edit to files under `app/agent/prompts/` MUST be
  validated with the `eval` skill before merging.
- **Security**: no secrets in code or commits; all credentials come from
  environment variables listed in `CLAUDE.md`; access follows least privilege.
- **Commit granularity**: changed/created modules are committed one file at a
  time with a short, focused commit message — never a bulk `git add .`/`git
  add -A`.

## Development Workflow

Every task follows the 7-step process defined in `CLAUDE.md`: Analyze → JIRA
→ Implement → Review → Test → Git → JIRA update. A task MUST have a JIRA
entry (description, affected modules, acceptance criteria) before
implementation begins. Implementation is not "done" until:
1. `./execute_tests.sh` passes (and any task-specific temporary tests),
2. changed modules have been re-reviewed for dead code,
3. commits are pushed only on explicit user request, and
4. the JIRA task is transitioned to `Готово` (id `41`) with a comment noting
   issues encountered and a brief final report.

Before any non-trivial change (new feature, refactor, architectural
decision), the approach MUST be checked against: SOLID/clean architecture,
security, observability, idempotency/state safety, and — for prompt changes —
`eval` validation. Deviations from best practice MUST be documented explicitly
in the commit message or a code comment.

## Governance

This constitution supersedes ad-hoc conventions and prior informal practice
for this repository. `CLAUDE.md` and `agent_docs/*.md` are the operational
detail layer beneath these principles; where they conflict with this
document, this document wins and the conflicting file MUST be updated in the
same change.

**Amendment procedure**: amendments are proposed via a documented change to
this file (PR or direct commit) including a Sync Impact Report (as prepended
above) describing version bump rationale and any dependent templates or docs
requiring updates. Amendments affecting `.specify/templates/*` or
`.claude/skills/speckit-*` MUST update those files in the same change.

**Versioning policy** (semantic versioning for governance):
- MAJOR: backward-incompatible principle removal or redefinition.
- MINOR: new principle or materially expanded section added.
- PATCH: wording clarifications, typo fixes, non-semantic refinements.

**Compliance review**: all PRs/reviews MUST verify compliance with the Core
Principles above; unresolved complexity or deviation MUST be justified
in-line per the Development Workflow section.

**Version**: 1.0.0 | **Ratified**: 2026-07-12 | **Last Amended**: 2026-07-12
