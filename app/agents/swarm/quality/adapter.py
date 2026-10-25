"""The swarm's QualityAdapter (ARCHIE-297)."""

from archie_shared.evals.models import EvalCaseInfo, EvalCaseResult
from app.platform.quality.adapter import StepCallback
from app.platform.quality.rules import SpecRule, spec_fingerprints
from .replay import SwarmCaseReplay
from .rules import case_rules, case_specs


class SwarmQualityAdapter:
    kind = "swarm"

    def __init__(self) -> None:
        self.replay = SwarmCaseReplay()

    def rules_for(self, target: str) -> list[SpecRule]:
        """`target`: an agent id; rules of its turn specs."""
        return list(case_rules(target, "swarm_turn", None))

    def spec_versions(self, target: str) -> dict[str, str]:
        return spec_fingerprints(case_specs(target, "swarm_turn"))

    async def run_case(
        self,
        run_id: str,
        case: EvalCaseInfo,
        model: str,
        attempt: int,
        judge_model: str | None,
        on_step: StepCallback,
    ) -> EvalCaseResult:
        return await self.replay.run(run_id, case, model, attempt, judge_model, on_step)
