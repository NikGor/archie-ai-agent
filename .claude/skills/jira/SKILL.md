---
name: jira
description: Manage JIRA tasks — search, create, transition, comment. Use during task workflow steps 2 and 8.
argument-hint: [action] [key or keyword]
---

Use the Jira MCP tools (`mcp__jira-mcp__*`).

**Step 2 — check/create JIRA task:**
1. Search: `mcp__jira-mcp__jira_get` with JQL `project=ARCHIE AND text~"$ARGUMENTS"`
2. If exists: use it (update description/acceptance criteria if needed)
3. If not: create via `mcp__jira-mcp__jira_post` with fields:
   - `summary`: short task title
   - `description`: problem, affected modules, acceptance criteria (Atlassian Document Format)
   - `issuetype`: `task` | `bug` | `story` | `subtask` | `epic`
   - `parent`: epic key for subtasks

**Step 3 — create branch:**
`git checkout -b <JIRA-KEY>-<short-english-description>`

**Step 7 — git push + PR:**
1. `git push` the branch
2. `gh pr create --title "<JIRA-KEY>: <summary>" --body "..."`

**Step 8 — update after PR is open:**
1. Transition to PR OPEN: `mcp__jira-mcp__jira_post` to `/rest/api/3/issue/{key}/transitions` with `{"transition": {"id": "2"}}`
2. Add comment with brief report: what was done, key decisions, issues encountered

JIRA project: **ARCHIE** on `badich.atlassian.net`
Transition IDs: `2` (PR OPEN), `11` (К выполнению), `21` (В работе), `31` (Postponed), `41` (Готово)
