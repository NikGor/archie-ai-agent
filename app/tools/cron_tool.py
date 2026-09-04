"""Tool for scheduling persistent future calls to other Archie tools."""

import json
from typing import Any
from redis.exceptions import RedisError
from app.backend.cron_scheduler import cron_scheduler


def _parse_arguments(value: str | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("arguments_json must be a valid JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError("arguments_json must contain a JSON object")
    return parsed


def _normalize_schedule_name(
    schedule_name: str | None,
    legacy_name: str | None,
    extra_arguments: dict[str, Any],
) -> str | None:
    """Recover the two malformed name encodings seen in structured LLM output."""
    normalized = schedule_name or legacy_name
    extras = dict(extra_arguments)
    if normalized is None:
        value_alias = extras.pop("value", None)
        if isinstance(value_alias, str):
            normalized = value_alias
    if normalized is None and len(extras) == 1:
        key, value = extras.popitem()
        if isinstance(value, str) and key == value:
            normalized = value
        else:
            extras[key] = value
    if extras:
        unknown = ", ".join(sorted(extras))
        raise ValueError(f"Unknown cron arguments: {unknown}")
    return normalized


async def cron_tool(  # noqa: PLR0912
    action: str = "create",
    job_id: str | None = None,
    schedule_name: str | None = None,
    tool_name: str | None = None,
    arguments_json: str | dict[str, Any] | None = None,
    run_at: str | None = None,
    cron_expression: str | None = None,
    timezone: str = "UTC",
    demo_mode: bool = False,
    name: str | None = None,
    **extra_arguments: Any,
) -> dict[str, Any]:
    """
    Schedule existing Archie tools for later execution and manage saved schedules.
    Use it for future actions such as turning on a light or opening a football channel.
    For a new job, translate natural-language dates into an ISO 8601 run_at using the
    current date/time context, or use a standard five-field cron_expression. Always pass
    the user's IANA timezone. arguments_json is the target tool's arguments as JSON.
    Creating a schedule authorizes that exact future tool call; cron_tool itself cannot
    be scheduled. Jobs persist across Archie restarts.

    Examples:
        One time: schedule_name="Living room light", tool_name="smarthome_control_tool",
        arguments_json='{"action":"turn_on","area":"Living Room"}',
        run_at="2026-08-30T20:00:00", timezone="Europe/Berlin".
        Recurring: schedule_name="Saturday football", tool_name="tv_tool",
        arguments_json='{"action":"play_channel","query":"Матч! Футбол 1"}',
        cron_expression="0 20 * * 6", timezone="Europe/Berlin".

    Args:
        action: One of: create, list, get, pause, resume, delete, run_now; defaults to create
        job_id: Scheduled job ID, required for get, pause, resume, delete, run_now
        schedule_name: Human-readable schedule name, required for create. Always use
            the literal parameter key schedule_name, never use its value as a key.
        tool_name: Existing Archie tool to call, required for create
        arguments_json: JSON object with arguments for the target tool, required for create
        run_at: One-time ISO 8601 local or offset date/time; mutually exclusive with cron_expression
        cron_expression: Standard five-field recurring cron expression; mutually exclusive with run_at
        timezone: IANA timezone such as Europe/Berlin; always use the user's timezone

    Returns:
        dict[str, Any]: Created job, saved jobs, management result, or execution result
    """
    action = action.strip().casefold()
    try:
        schedule_name = _normalize_schedule_name(schedule_name, name, extra_arguments)
    except ValueError as exc:
        return {"status": "error", "action": action, "message": str(exc)}
    if demo_mode:
        return {
            "status": "demo",
            "action": action,
            "message": f"[DEMO] Scheduled-task action {action} would be executed",
        }

    try:
        if action == "create":
            if not schedule_name or not tool_name:
                raise ValueError("schedule_name and tool_name are required for create")
            job = await cron_scheduler.create_job(
                name=schedule_name,
                tool_name=tool_name,
                arguments=_parse_arguments(arguments_json),
                timezone=timezone,
                run_at=run_at,
                cron_expression=cron_expression,
            )
            return {"status": "success", "action": action, "job": job}
        if action == "list":
            return {
                "status": "success",
                "action": action,
                "jobs": await cron_scheduler.list_jobs(),
            }
        if not job_id:
            raise ValueError(f"job_id is required for {action}")
        if action == "get":
            result = await cron_scheduler.get_job(job_id)
        elif action == "pause":
            result = await cron_scheduler.set_enabled(job_id, False)
        elif action == "resume":
            result = await cron_scheduler.set_enabled(job_id, True)
        elif action == "delete":
            result = await cron_scheduler.delete_job(job_id)
        elif action == "run_now":
            result = await cron_scheduler.run_job_now(job_id)
        else:
            raise ValueError(f"Unknown cron action: {action}")
        return {"status": "success", "action": action, "result": result}
    except (ValueError, RedisError) as exc:
        return {"status": "error", "action": action, "message": str(exc)}
