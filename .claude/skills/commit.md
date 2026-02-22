---
description: Create a git commit following project conventions
---

Stage and commit following CLAUDE.md step 6:

1. `git status` — identify changed files
2. `git diff` — review all changes
3. `git add <file>` — stage files **one by one** (never `git add -A` or `git add .`)
4. Commit with message: `<type>(<ticket>): <description>`
   - Types: `feat`, `fix`, `refactor`, `test`, `docs`
   - Ticket: JIRA key if applicable (e.g. `ARCHIE-27`)
   - Example: `feat(ARCHIE-29): add PostToolUse auto-format hook`

Never use `--no-verify`. Do not push unless explicitly asked.
