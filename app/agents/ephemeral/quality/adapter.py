"""Replay diagnostics with fixed evidence and grade goals and prompt rules."""

import json
import logging
import time
from datetime import UTC, datetime
from archie_shared.evals.models import (
    EvalAssertionFailure,
    EvalCaseInfo,
    EvalCaseResult,
)
from archie_shared.monitoring.models import MonitorRuleScore
from pydantic import BaseModel, Field, JsonValue
from app.agents.ephemeral.models import EphemeralResult, SpawnRequest
from app.agents.ephemeral.quality import configurator as configurator_eval
from app.agents.ephemeral.runner import EphemeralRunner
from app.platform.llm.decisions import decide
from app.platform.quality.adapter import StepCallback
from app.platform.quality.judge import build_questions, parse_answers
from app.platform.quality.rules import (
    RulePolicy,
    SpecRule,
    select_spec_rules,
    spec_fingerprints,
)
from app.platform.runtime.spawn import SpawnContext
from app.platform.tools.models import ToolCall, ToolContext, ToolResult


logger = logging.getLogger(__name__)
_SPECS = ["ephemeral_base", "ephemeral_ha_diagnosis"]


class EphemeralScenario(BaseModel):
    goal: str = Field(description="Diagnosis requested by the parent")
    context: str = Field("", description="Fixed background facts")
    tool_results: dict[str, dict[str, JsonValue]] = Field(
        default_factory=dict,
        description="Fixed tool evidence; no real tools execute during eval",
    )


class EphemeralEvaluation(BaseModel):
    goal_achieved: bool = Field(
        description="The result addresses the achievable goal and states evidence gaps"
    )
    rules: list[MonitorRuleScore] = Field(description="Prompt rule scores")
    model: str = Field(description="Jev model that graded the record")
    cost: float | None = Field(
        None, description="Jev cost when returned by the provider"
    )


class EphemeralQualityAdapter:
    kind = "ephemeral"

    def rules_for(self, target: str = "ha_diagnosis") -> list[SpecRule]:
        return [
            rule
            for spec in self._specs(target)
            for rule in select_spec_rules(spec, {}, RulePolicy())
        ]

    def spec_versions(self, target: str = "ha_diagnosis") -> dict[str, str]:
        return spec_fingerprints(self._specs(target))

    @staticmethod
    def _specs(target: str) -> list[str]:
        """ha_diagnosis: the swarm preset; configurator: ARCHIE-323 cases."""
        if target == "ha_diagnosis":
            return _SPECS
        if target == configurator_eval.TARGET:
            return [configurator_eval.TARGET]
        raise ValueError(f"Unknown ephemeral target: {target}")

    async def judge(
        self,
        goal: str,
        result: EphemeralResult,
        model: str,
        rule_keys: list[str] | None = None,
    ) -> EphemeralEvaluation:
        rules = [
            rule
            for rule in self.rules_for()
            if rule_keys is None or rule.key in rule_keys
        ]
        questions = build_questions(rules)
        questions["goal"] = {
            "type": "choice",
            "instructions": "Did the result address the goal using available evidence and clearly state any missing evidence?",
            "criteria": {
                "achieved": "The achievable diagnosis is complete and evidence gaps are explicit",
                "not_achieved": "The result misses the goal or claims unsupported conclusions",
            },
        }
        data = await decide(
            model, {"goal": goal, "record": result.model_dump(mode="json")}, questions
        )
        answer = data["answers"]["goal"]
        probabilities = answer["probabilities"]
        return EphemeralEvaluation(
            goal_achieved=max(probabilities, key=probabilities.get) == "achieved",
            rules=parse_answers(rules, data["answers"]),
            model=data.get("model", model),
            cost=data.get("usage", {}).get("cost"),
        )

    async def run_case(
        self,
        run_id: str,
        case: EvalCaseInfo,
        model: str,
        attempt: int,
        judge_model: str | None,
        on_step: StepCallback,
    ) -> EvalCaseResult:
        if case.response_format == configurator_eval.TARGET:
            return await self._run_configurator_case(
                run_id, case, model, attempt, judge_model, on_step
            )
        started = time.monotonic()
        failures = []
        error = None
        result: EphemeralResult | None = None
        try:
            scenario = EphemeralScenario.model_validate_json(case.input)

            async def run_tools(
                calls: list[ToolCall], _context: ToolContext
            ) -> list[ToolResult]:
                return [
                    ToolResult(
                        tool_name=call.tool_name,
                        output=scenario.tool_results.get(
                            call.tool_name,
                            {"error": "No tool result supplied by the case"},
                        ),
                        success=call.tool_name in scenario.tool_results,
                    )
                    for call in calls
                ]

            await on_step("ephemeral_run")
            request = SpawnRequest(
                goal=scenario.goal,
                context=scenario.context,
                parent_run_id=run_id,
                publish_to="none",
            )
            result = await EphemeralRunner(run_tools=run_tools, model=model).run(
                request,
                SpawnContext(source="eval", parent_run_id=run_id, demo_mode=True),
            )
            if result.status != "completed":
                failures.append(
                    EvalAssertionFailure(
                        key="status",
                        expected='"completed"',
                        actual=json.dumps(result.status),
                    )
                )
            if case.expect.content_not_empty and not result.summary.strip():
                failures.append(
                    EvalAssertionFailure(
                        key="content_not_empty", expected="true", actual="false"
                    )
                )
            text = result.summary + "\n" + "\n".join(result.findings)
            contains = case.expect.text_contains or case.expect.reply_contains
            if contains and contains.casefold() not in text.casefold():
                failures.append(
                    EvalAssertionFailure(
                        key="text_contains",
                        expected=json.dumps(contains),
                        actual=json.dumps(text),
                    )
                )
            if case.expect.rules_followed:
                await on_step("rules")
                evaluation = await self.judge(
                    scenario.goal,
                    result,
                    judge_model or "~typesafe/jev-latest",
                    case.rules or None,
                )
                for score in evaluation.rules:
                    if score.grade in ("red", None):
                        failures.append(
                            EvalAssertionFailure(
                                key=f"{score.spec}:{score.rule_id}",
                                expected='"not red and applicable"',
                                actual=json.dumps(score.grade),
                            )
                        )
                if not evaluation.goal_achieved:
                    failures.append(
                        EvalAssertionFailure(
                            key="goal_achieved", expected="true", actual="false"
                        )
                    )
        except Exception as exc:
            logger.exception(f"ephemeral_eval_error_001: {case.id} {exc}")
            error = f"{type(exc).__name__}: {exc}"
        return EvalCaseResult(
            run_id=run_id,
            case_id=case.id,
            model=model,
            attempt=attempt,
            status="error" if error else "failed" if failures else "passed",
            error=error,
            failures=failures,
            total_ms=int((time.monotonic() - started) * 1000),
            total_tokens=(
                result.usage.input_tokens + result.usage.output_tokens
                if result
                else None
            ),
            output_json=result.model_dump_json() if result else None,
            finished_at=datetime.now(UTC).isoformat(),
        )

    async def _run_configurator_case(
        self,
        run_id: str,
        case: EvalCaseInfo,
        model: str,
        attempt: int,
        judge_model: str | None,
        on_step: StepCallback,
    ) -> EvalCaseResult:
        started = time.monotonic()
        failures: list[EvalAssertionFailure] = []
        error = None
        result = None
        try:
            await on_step("configurator")
            result, failures = await configurator_eval.run_scenario(
                case.input, run_id, model, bool(case.expect.rules_followed), judge_model
            )
        except Exception as exc:
            logger.exception(f"ephemeral_eval_error_002: {case.id} {exc}")
            error = f"{type(exc).__name__}: {exc}"
        return EvalCaseResult(
            run_id=run_id,
            case_id=case.id,
            model=model,
            attempt=attempt,
            status="error" if error else "failed" if failures else "passed",
            error=error,
            failures=failures,
            total_ms=int((time.monotonic() - started) * 1000),
            total_tokens=(
                result.input_tokens + result.output_tokens if result else None
            ),
            output_json=result.model_dump_json() if result else None,
            finished_at=datetime.now(UTC).isoformat(),
        )
