"""The chat's QualityAdapter (ARCHIE-296): its spec rules, spec versions and
eval replay, as the platform quality engine sees them."""

from archie_shared.evals.models import EvalCaseInfo, EvalCaseResult
from app.platform.quality.adapter import StepCallback
from app.platform.quality.rules import SpecRule
from .replay import ChatCaseReplay
from .rules import rules_for_turn, spec_versions


class ChatQualityAdapter:
    kind = "chat"

    def __init__(self) -> None:
        self._replay: ChatCaseReplay | None = None

    @property
    def replay(self) -> ChatCaseReplay:
        """Built on first use: the replay owns a chat pipeline."""
        if self._replay is None:
            self._replay = ChatCaseReplay()
        return self._replay

    def rules_for(self, response_format: str) -> list[SpecRule]:
        return rules_for_turn(response_format, trace_found=True)

    def spec_versions(self, response_format: str) -> dict[str, str]:
        return spec_versions(response_format, trace_found=True)

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
