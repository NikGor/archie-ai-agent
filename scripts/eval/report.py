"""Terminal reporting and JSON export for eval runs."""

import json
import statistics
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich import box

from .eval_types import EvalResult, RunSummary


_console = Console()


def build_run_summary(
    results: list[EvalResult],
    model: str,
    suite: str,
) -> RunSummary:
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    pass_rate = passed / len(results) if results else 0.0
    total_ms_values = [r.total_ms for r in results if r.total_ms is not None]
    stage1_ms_values = [r.stage1_ms for r in results if r.stage1_ms is not None]
    stage3_ms_values = [r.stage3_ms for r in results if r.stage3_ms is not None]
    token_values = [r.total_tokens for r in results if r.total_tokens is not None]
    return RunSummary(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        model=model,
        suite=suite,
        total=len(results),
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
        median_total_ms=statistics.median(total_ms_values) if total_ms_values else None,
        mean_total_tokens=statistics.mean(token_values) if token_values else None,
        stage1_mean_ms=statistics.mean(stage1_ms_values) if stage1_ms_values else None,
        stage3_mean_ms=statistics.mean(stage3_ms_values) if stage3_ms_values else None,
        cases=results,
    )


def print_summary(summary: RunSummary) -> None:
    _console.print()
    _console.rule(
        f"[bold]Eval: {summary.suite}[/bold]  model=[cyan]{summary.model}[/cyan]"
    )
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Case", style="white", min_width=28)
    table.add_column("Result", justify="center", min_width=8)
    table.add_column("Tokens", justify="right", min_width=8)
    table.add_column("Total ms", justify="right", min_width=9)
    table.add_column("S1 ms", justify="right", min_width=7)
    table.add_column("S3 ms", justify="right", min_width=7)
    table.add_column("Failures", style="dim")
    for r in summary.cases:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        if r.error:
            status = "[red]ERROR[/red]"
        failure_text = "; ".join(str(f) for f in r.failures) if r.failures else ""
        if r.error:
            failure_text = r.error[:80]
        table.add_row(
            r.name,
            status,
            str(r.total_tokens) if r.total_tokens is not None else "-",
            str(r.total_ms) if r.total_ms is not None else "-",
            str(r.stage1_ms) if r.stage1_ms is not None else "-",
            str(r.stage3_ms) if r.stage3_ms is not None else "-",
            failure_text,
        )
    _console.print(table)
    pass_color = (
        "green"
        if summary.pass_rate == 1.0
        else ("yellow" if summary.pass_rate >= 0.5 else "red")
    )
    _console.print(
        f"  [bold]Results:[/bold] [{pass_color}]{summary.passed}/{summary.total} passed[/{pass_color}] "
        f"({summary.pass_rate:.0%})"
    )
    if summary.median_total_ms is not None:
        _console.print(
            f"  [bold]Latency:[/bold] median={summary.median_total_ms:.0f}ms", end=""
        )
        if summary.stage1_mean_ms is not None:
            _console.print(f"  stage1_mean={summary.stage1_mean_ms:.0f}ms", end="")
        if summary.stage3_mean_ms is not None:
            _console.print(f"  stage3_mean={summary.stage3_mean_ms:.0f}ms", end="")
        _console.print()
    if summary.mean_total_tokens is not None:
        _console.print(f"  [bold]Tokens:[/bold] mean={summary.mean_total_tokens:.0f}")
    _console.print()


def print_diff(current: RunSummary, previous: RunSummary) -> None:
    _console.rule("[bold]Diff vs previous run[/bold]")
    delta_pass = current.passed - previous.passed
    delta_color = "green" if delta_pass >= 0 else "red"
    _console.print(
        f"  Pass rate: {previous.pass_rate:.0%} → [{delta_color}]{current.pass_rate:.0%}[/{delta_color}]"
        f"  ({delta_pass:+d} cases)"
    )
    if current.median_total_ms is not None and previous.median_total_ms is not None:
        delta_ms = current.median_total_ms - previous.median_total_ms
        ms_color = "green" if delta_ms <= 0 else "yellow"
        _console.print(
            f"  Latency: {previous.median_total_ms:.0f}ms → [{ms_color}]{current.median_total_ms:.0f}ms[/{ms_color}]"
            f"  ({delta_ms:+.0f}ms)"
        )
    if current.mean_total_tokens is not None and previous.mean_total_tokens is not None:
        delta_tok = current.mean_total_tokens - previous.mean_total_tokens
        tok_color = "green" if delta_tok <= 0 else "yellow"
        _console.print(
            f"  Tokens:  {previous.mean_total_tokens:.0f} → [{tok_color}]{current.mean_total_tokens:.0f}[/{tok_color}]"
            f"  ({delta_tok:+.0f})"
        )
    _console.print()


def export_json(summary: RunSummary, path: str) -> None:
    data = {
        "timestamp": summary.timestamp,
        "model": summary.model,
        "suite": summary.suite,
        "summary": {
            "total": summary.total,
            "passed": summary.passed,
            "failed": summary.failed,
            "pass_rate": round(summary.pass_rate, 4),
        },
        "metrics": {
            "median_total_ms": summary.median_total_ms,
            "mean_total_tokens": summary.mean_total_tokens,
            "stage1_mean_ms": summary.stage1_mean_ms,
            "stage3_mean_ms": summary.stage3_mean_ms,
        },
        "cases": [
            {
                "name": r.name,
                "passed": r.passed,
                "failures": [str(f) for f in r.failures],
                "error": r.error,
                "stage1_ms": r.stage1_ms,
                "stage2_ms": r.stage2_ms,
                "stage3_ms": r.stage3_ms,
                "total_ms": r.total_ms,
                "total_tokens": r.total_tokens,
            }
            for r in summary.cases
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json_summary(path: str) -> RunSummary:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    from .eval_types import AssertionFailure, EvalResult

    cases = [
        EvalResult(
            name=c["name"],
            passed=c["passed"],
            failures=[],
            error=c.get("error"),
            stage1_ms=c.get("stage1_ms"),
            stage2_ms=c.get("stage2_ms"),
            stage3_ms=c.get("stage3_ms"),
            total_ms=c.get("total_ms"),
            total_tokens=c.get("total_tokens"),
        )
        for c in data["cases"]
    ]
    return RunSummary(
        timestamp=data["timestamp"],
        model=data["model"],
        suite=data["suite"],
        total=data["summary"]["total"],
        passed=data["summary"]["passed"],
        failed=data["summary"]["failed"],
        pass_rate=data["summary"]["pass_rate"],
        median_total_ms=data["metrics"].get("median_total_ms"),
        mean_total_tokens=data["metrics"].get("mean_total_tokens"),
        stage1_mean_ms=data["metrics"].get("stage1_mean_ms"),
        stage3_mean_ms=data["metrics"].get("stage3_mean_ms"),
        cases=cases,
    )
