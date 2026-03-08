"""
Eval runner for the Archie AI agent pipeline.

Usage:
    python -m scripts.eval.eval_runner
    python -m scripts.eval.eval_runner --suite routing
    python -m scripts.eval.eval_runner --model gpt-4.1-mini --compare gpt-4.1
    python -m scripts.eval.eval_runner --suite routing --diff eval_results/2026-01-01_10-00.json
    python -m scripts.eval.eval_runner --suite ui_answer --judge
    python -m scripts.eval.eval_runner --suite ui_answer --judge --judge-model gpt-4.1
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agent.agent_factory import AgentFactory
from app.utils.provider_utils import get_provider_for_model

from .assertion_engine import check_full, check_stage1
from .eval_types import EvalCase, Expect, EvalResult
from .judge import evaluate_response
from .report import (
    build_run_summary,
    export_json,
    load_json_summary,
    print_diff,
    print_summary,
)

CASES_DIR = Path(__file__).parent / "cases"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "eval_results"
DEFAULT_MODEL = "gpt-4.1-mini"


def load_cases(suite: str) -> list[EvalCase]:
    path = CASES_DIR / f"{suite}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Suite file not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cases = []
    for item in raw:
        expect_data = item.get("expect", {})
        expect = Expect(
            action_type=expect_data.get("action_type"),
            tools_called=expect_data.get("tools_called"),
            content_not_empty=expect_data.get("content_not_empty"),
            max_total_tokens=expect_data.get("max_total_tokens"),
            max_latency_ms=expect_data.get("max_latency_ms"),
            text_contains=expect_data.get("text_contains"),
            min_ui_items=expect_data.get("min_ui_items"),
            has_quick_actions=expect_data.get("has_quick_actions"),
            has_form=expect_data.get("has_form"),
            expected_item_types=expect_data.get("expected_item_types"),
        )
        cases.append(
            EvalCase(
                name=item["name"],
                input=item["input"],
                response_format=item.get("response_format", "plain"),
                stage=item.get("stage", "full"),
                judge=item.get("judge", False),
                expect=expect,
            )
        )
    return cases


async def run_case(
    case: EvalCase,
    agent_factory: AgentFactory,
    model: str,
    judge_model: str | None = None,
) -> EvalResult:
    try:
        if case.stage == "stage1":
            provider = get_provider_for_model(model)
            user_state = agent_factory.state_service.get_user_state()
            decision, _ = await agent_factory._make_command_call(
                user_input=case.input,
                model=model,
                provider=provider,
                user_state=user_state,
                response_format=case.response_format,
            )
            return check_stage1(case, decision)
        else:
            messages = [{"role": "user", "content": case.input}]
            response = await agent_factory.arun(
                messages=messages,
                command_model=model,
                final_output_model=model,
                response_format=case.response_format,
            )
            result = check_full(case, response)
            if case.judge and judge_model is not None:
                result.judge = await evaluate_response(case, response, judge_model)
            return result
    except Exception as e:
        return EvalResult(
            name=case.name,
            passed=False,
            error=f"{type(e).__name__}: {e}",
        )


async def run_suite(
    suite: str,
    model: str,
    agent_factory: AgentFactory,
    judge_model: str | None = None,
) -> list[EvalResult]:
    cases = load_cases(suite)
    results = []
    for case in cases:
        result = await run_case(case, agent_factory, model, judge_model=judge_model)
        results.append(result)
    return results


def get_all_suites() -> list[str]:
    return [p.stem for p in CASES_DIR.glob("*.yaml")]


def make_output_path(suite: str, model: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    safe_model = model.replace("/", "_")
    RESULTS_DIR.mkdir(exist_ok=True)
    return str(RESULTS_DIR / f"{ts}_{suite}_{safe_model}.json")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Archie agent eval runner")
    parser.add_argument("--suite", help="Suite name (default: all suites)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Command + output model")
    parser.add_argument("--compare", help="Second model for A/B comparison")
    parser.add_argument("--diff", help="Path to previous JSON report for comparison")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable LLM-as-judge for cases with judge: true in config",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_MODEL,
        help="Model to use for LLM judge (default: same as --model)",
    )
    args = parser.parse_args()

    judge_model = args.judge_model if args.judge else None
    agent_factory = AgentFactory(demo_mode=True)

    suites = [args.suite] if args.suite else get_all_suites()
    if not suites:
        print("No suite files found in", CASES_DIR)
        sys.exit(1)

    all_results_a: list[EvalResult] = []
    all_results_b: list[EvalResult] = []

    for suite in suites:
        results_a = await run_suite(
            suite, args.model, agent_factory, judge_model=judge_model
        )
        all_results_a.extend(results_a)
        summary_a = build_run_summary(results_a, args.model, suite)
        print_summary(summary_a)
        out_path = make_output_path(suite, args.model)
        export_json(summary_a, out_path)
        print(f"  Saved: {out_path}\n")

        if args.compare:
            results_b = await run_suite(
                suite, args.compare, agent_factory, judge_model=judge_model
            )
            all_results_b.extend(results_b)
            summary_b = build_run_summary(results_b, args.compare, suite)
            print_summary(summary_b)
            print_diff(summary_b, summary_a)

    if args.diff:
        previous = load_json_summary(args.diff)
        combined_suite = args.suite or "all"
        current_summary = build_run_summary(all_results_a, args.model, combined_suite)
        print_diff(current_summary, previous)


if __name__ == "__main__":
    asyncio.run(main())
