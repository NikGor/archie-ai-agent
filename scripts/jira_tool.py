"""JIRA CLI tool for managing issues via REST API."""

import argparse
import asyncio
import logging
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

ISSUE_TYPES = {
    "bug": "Bug",
    "feature": "Story",
    "story": "Story",
    "task": "Задача",
    "subtask": "Subtask",
    "epic": "Эпик",
}


class JiraTool:
    """Async JIRA REST API client."""

    def __init__(
        self,
        host: str | None = None,
        email: str | None = None,
        api_token: str | None = None,
        project: str | None = None,
    ):
        self.host = host or os.getenv("JIRA_HOST", "")
        self.email = email or os.getenv("JIRA_EMAIL", "")
        self.api_token = api_token or os.getenv("JIRA_API_TOKEN", "")
        self.project = project or os.getenv("JIRA_PROJECT", "ARCHIE")
        self.base_url = f"https://{self.host}/rest/api/3"
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate that JIRA credentials are set."""
        if not self.host or not self.email or not self.api_token:
            logger.error(
                "\033[31mMissing JIRA_HOST, JIRA_EMAIL, or JIRA_API_TOKEN in .env\033[0m"
            )
            sys.exit(1)

    def _get_auth(self) -> tuple[str, str]:
        """Return Basic Auth tuple."""
        return (self.email, self.api_token)

    def _resolve_issue_type(self, type_name: str) -> str:
        """Resolve issue type alias to JIRA issue type name."""
        normalized = type_name.lower().strip()
        if normalized not in ISSUE_TYPES:
            logger.error(
                f"\033[31mUnknown issue type: {type_name}. "
                f"Valid: {', '.join(ISSUE_TYPES.keys())}\033[0m"
            )
            sys.exit(1)
        return ISSUE_TYPES[normalized]

    def _build_adf_document(
        self,
        description: str,
        acceptance_criteria: list[str] | None = None,
    ) -> dict:
        """Build Atlassian Document Format JSON from plain text."""
        content = []
        if description:
            content.append(
                {
                    "type": "heading",
                    "attrs": {"level": 3},
                    "content": [{"type": "text", "text": "Описание"}],
                }
            )
            content.append(
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": description}],
                }
            )
        if acceptance_criteria:
            content.append(
                {
                    "type": "heading",
                    "attrs": {"level": 3},
                    "content": [{"type": "text", "text": "Критерии приёмки"}],
                }
            )
            items = [
                {
                    "type": "listItem",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": c}],
                        }
                    ],
                }
                for c in acceptance_criteria
            ]
            content.append({"type": "bulletList", "content": items})
        return {"type": "doc", "version": 1, "content": content}

    def _browse_url(self, key: str) -> str:
        """Return browse URL for an issue."""
        return f"https://{self.host}/browse/{key}"

    async def create_issue(
        self,
        issue_type: str,
        summary: str,
        parent: str | None = None,
        description: str | None = None,
        acceptance_criteria: list[str] | None = None,
        project: str | None = None,
    ) -> dict:
        """Create a JIRA issue."""
        jira_type = self._resolve_issue_type(issue_type)
        fields: dict = {
            "project": {"key": project or self.project},
            "issuetype": {"name": jira_type},
            "summary": summary,
        }
        if parent:
            fields["parent"] = {"key": parent}
        if description or acceptance_criteria:
            fields["description"] = self._build_adf_document(
                description=description or "",
                acceptance_criteria=acceptance_criteria,
            )
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/issue",
                auth=self._get_auth(),
                headers={"Content-Type": "application/json"},
                json={"fields": fields},
                timeout=15.0,
            )
        if response.status_code == 201:
            data = response.json()
            key = data["key"]
            print(
                f"\033[32m✓\033[0m Created {jira_type}: "
                f"\033[36m{key}\033[0m — {summary}"
            )
            print(f"  \033[34m{self._browse_url(key)}\033[0m")
            return data
        logger.error(
            f"\033[31m✗ Failed ({response.status_code}): {response.text}\033[0m"
        )
        sys.exit(1)

    async def update_issue(
        self,
        issue_key: str,
        summary: str | None = None,
        description: str | None = None,
        acceptance_criteria: list[str] | None = None,
    ) -> None:
        """Update an existing JIRA issue."""
        fields: dict = {}
        if summary:
            fields["summary"] = summary
        if description or acceptance_criteria:
            fields["description"] = self._build_adf_document(
                description=description or "",
                acceptance_criteria=acceptance_criteria,
            )
        if not fields:
            logger.error(
                "\033[31mNothing to update. Use --summary, --description, or --ac.\033[0m"
            )
            sys.exit(1)
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.base_url}/issue/{issue_key}",
                auth=self._get_auth(),
                headers={"Content-Type": "application/json"},
                json={"fields": fields},
                timeout=15.0,
            )
        if response.status_code == 204:
            print(f"\033[32m✓\033[0m Updated: \033[36m{issue_key}\033[0m")
            return
        logger.error(
            f"\033[31m✗ Failed ({response.status_code}): {response.text}\033[0m"
        )
        sys.exit(1)

    async def list_issues(
        self,
        jql: str,
        max_results: int = 50,
    ) -> list[dict]:
        """Search JIRA issues by JQL."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/search/jql",
                auth=self._get_auth(),
                params={
                    "jql": jql,
                    "maxResults": max_results,
                    "fields": "summary,status,issuetype,parent,priority",
                },
                timeout=15.0,
            )
        if response.status_code != 200:
            logger.error(
                f"\033[31m✗ Search failed ({response.status_code}): {response.text}\033[0m"
            )
            sys.exit(1)
        data = response.json()
        issues = data.get("issues", [])
        print(f"\033[33m{len(issues)}\033[0m issues found:\n")
        for issue in issues:
            fields = issue["fields"]
            key = issue["key"]
            issue_summary = fields["summary"]
            status = fields["status"]["name"]
            issue_type = fields["issuetype"]["name"]
            parent_key = fields["parent"]["key"] if fields.get("parent") else "—"
            print(
                f"  \033[36m{key}\033[0m [{issue_type}] {issue_summary}  "
                f"(\033[33m{status}\033[0m, parent: {parent_key})"
            )
        return issues

    async def add_comment(self, issue_key: str, text: str) -> None:
        """Add a comment to a JIRA issue."""
        body = {
            "type": "doc",
            "version": 1,
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": text}]}
            ],
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/issue/{issue_key}/comment",
                auth=self._get_auth(),
                headers={"Content-Type": "application/json"},
                json={"body": body},
                timeout=15.0,
            )
        if response.status_code == 201:
            print(f"\033[32m✓\033[0m Comment added to \033[36m{issue_key}\033[0m")
            return
        logger.error(
            f"\033[31m✗ Failed ({response.status_code}): {response.text}\033[0m"
        )
        sys.exit(1)

    async def transition_issue(self, issue_key: str, transition_id: str) -> None:
        """Transition a JIRA issue to a new status."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/issue/{issue_key}/transitions",
                auth=self._get_auth(),
                headers={"Content-Type": "application/json"},
                json={"transition": {"id": transition_id}},
                timeout=15.0,
            )
        if response.status_code == 204:
            print(
                f"\033[32m✓\033[0m Transitioned: \033[36m{issue_key}\033[0m → {transition_id}"
            )
            return
        logger.error(
            f"\033[31m✗ Failed ({response.status_code}): {response.text}\033[0m"
        )
        sys.exit(1)

    async def get_issue(self, issue_key: str) -> dict:
        """Get a single JIRA issue details."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/issue/{issue_key}",
                auth=self._get_auth(),
                params={
                    "fields": "summary,status,issuetype,parent,priority,description,subtasks",
                },
                timeout=15.0,
            )
        if response.status_code != 200:
            logger.error(
                f"\033[31m✗ Failed ({response.status_code}): {response.text}\033[0m"
            )
            sys.exit(1)
        data = response.json()
        fields = data["fields"]
        print(f"\033[36m{issue_key}\033[0m — {fields['summary']}")
        print(
            f"  Type: {fields['issuetype']['name']}  "
            f"Status: \033[33m{fields['status']['name']}\033[0m"
        )
        if fields.get("parent"):
            print(f"  Parent: {fields['parent']['key']}")
        if fields.get("subtasks"):
            print("  Subtasks:")
            for st in fields["subtasks"]:
                print(
                    f"    \033[36m{st['key']}\033[0m — {st['fields']['summary']} "
                    f"(\033[33m{st['fields']['status']['name']}\033[0m)"
                )
        return data


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="JIRA CLI tool for archie-ai-agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/jira_tool.py create story ARCHIE-8 "New feature"
  python scripts/jira_tool.py create subtask ARCHIE-10 "Fix bug" -d "Details" --ac "Tests pass" --ac "No errors"
  python scripts/jira_tool.py create bug ARCHIE-8 "Login broken" -d "Cannot login"
  python scripts/jira_tool.py update ARCHIE-17 -d "Updated desc" --ac "Criterion 1"
  python scripts/jira_tool.py list "parent = ARCHIE-10"
  python scripts/jira_tool.py get ARCHIE-10
  python scripts/jira_tool.py transition ARCHIE-21 41
  python scripts/jira_tool.py comment ARCHIE-21 "Реализовано. Issues: none."
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a new issue")
    create_parser.add_argument(
        "type",
        help=f"Issue type: {', '.join(ISSUE_TYPES.keys())}",
    )
    create_parser.add_argument(
        "parent",
        help="Parent issue key (e.g. ARCHIE-8)",
    )
    create_parser.add_argument("summary", help="Issue title")
    create_parser.add_argument("--description", "-d", help="Description text")
    create_parser.add_argument(
        "--ac",
        action="append",
        default=[],
        help="Acceptance criterion (repeatable)",
    )
    create_parser.add_argument(
        "--project",
        default=None,
        help="Project key override",
    )

    update_parser = subparsers.add_parser("update", help="Update an existing issue")
    update_parser.add_argument("key", help="Issue key (e.g. ARCHIE-17)")
    update_parser.add_argument("--summary", "-s", help="New summary")
    update_parser.add_argument("--description", "-d", help="New description text")
    update_parser.add_argument(
        "--ac",
        action="append",
        default=[],
        help="Acceptance criterion (repeatable)",
    )

    list_parser = subparsers.add_parser("list", help="Search issues by JQL")
    list_parser.add_argument(
        "jql",
        help='JQL query (e.g. "parent = ARCHIE-10")',
    )
    list_parser.add_argument(
        "--max",
        type=int,
        default=50,
        help="Max results",
    )

    get_parser = subparsers.add_parser("get", help="Get issue details")
    get_parser.add_argument("key", help="Issue key (e.g. ARCHIE-10)")

    comment_parser = subparsers.add_parser("comment", help="Add a comment to an issue")
    comment_parser.add_argument("key", help="Issue key (e.g. ARCHIE-21)")
    comment_parser.add_argument("text", help="Comment text")

    transition_parser = subparsers.add_parser(
        "transition", help="Transition issue status"
    )
    transition_parser.add_argument("key", help="Issue key (e.g. ARCHIE-21)")
    transition_parser.add_argument(
        "transition_id",
        help="Transition ID: 2=PR OPEN, 11=К выполнению, 21=В работе, 31=Postponed, 41=Готово",
    )

    return parser


async def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    jira = JiraTool()

    if args.command == "create":
        await jira.create_issue(
            issue_type=args.type,
            summary=args.summary,
            parent=args.parent,
            description=args.description,
            acceptance_criteria=args.ac if args.ac else None,
            project=args.project,
        )
    elif args.command == "update":
        await jira.update_issue(
            issue_key=args.key,
            summary=args.summary,
            description=args.description,
            acceptance_criteria=args.ac if args.ac else None,
        )
    elif args.command == "list":
        await jira.list_issues(
            jql=args.jql,
            max_results=args.max,
        )
    elif args.command == "get":
        await jira.get_issue(issue_key=args.key)
    elif args.command == "comment":
        await jira.add_comment(issue_key=args.key, text=args.text)
    elif args.command == "transition":
        await jira.transition_issue(
            issue_key=args.key, transition_id=args.transition_id
        )


if __name__ == "__main__":
    asyncio.run(main())
