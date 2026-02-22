---
description: Manage JIRA tasks following the project workflow
---

Use the Jira MCP tools (`mcp__jira-mcp__*`) or `python scripts/jira_tool.py` CLI.

**Step 2 of task workflow — check/create JIRA task:**
1. Search for existing tasks: `mcp__jira-mcp__jira_get` with JQL `project=MBA AND text~"<keyword>"`
2. If task exists: use it (update description/acceptance criteria if needed)
3. If no match: create via `mcp__jira-mcp__jira_post` with fields:
   - `summary`: short task title
   - `description`: problem, affected modules, acceptance criteria (Atlassian Document Format)
   - `issuetype`: `task` | `bug` | `story` | `subtask` | `epic`
   - `parent`: epic key for subtasks

**Step 7 — update after implementation:**
1. Transition: POST `/rest/api/3/issue/{key}/transitions` with body `{"transition": {"id": "41"}}`
2. Comment: add brief report — what was done, key decisions, issues encountered

JIRA project: **MBA** on `badich.atlassian.net`
Transition IDs: `11` (К выполнению), `21` (В работе), `31` (Postponed), `41` (Готово)
