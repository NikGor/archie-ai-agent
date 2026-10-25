"""Runs one attempt of one eval case through the quality adapter of its agent
kind (ARCHIE-296). Cases without a kind belong to the chat."""

from archie_shared.evals.models import EvalCaseInfo, EvalCaseResult
from app.agents.registry import load_agent_kinds
from app.platform.quality.adapter import StepCallback, quality_adapter


DEFAULT_AGENT_KIND = "chat"


class EvalAttemptRunner:
    def __init__(self) -> None:
        load_agent_kinds()

    async def run(
        self,
        run_id: str,
        case: EvalCaseInfo,
        model: str,
        attempt: int,
        judge_model: str | None,
        on_step: StepCallback,
    ) -> EvalCaseResult:
        kind = getattr(case, "agent_kind", None) or DEFAULT_AGENT_KIND
        return await quality_adapter(kind).run_case(
            run_id, case, model, attempt, judge_model, on_step
        )
