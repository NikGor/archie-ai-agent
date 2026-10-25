"""QualityAdapter: what an agent kind implements to get monitoring, eval and
the prompt autofix cycle (ARCHIE-296). The platform keeps one adapter per kind."""

import logging
from collections.abc import Awaitable, Callable
from typing import Protocol
from archie_shared.evals.models import EvalCaseInfo, EvalCaseResult
from .rules import SpecRule


logger = logging.getLogger(__name__)

StepCallback = Callable[[str], Awaitable[None]]


class QualityAdapter(Protocol):
    kind: str

    def rules_for(self, target: str) -> list[SpecRule]:
        """Every judged rule of the kind's specs for a target: a response
        format for the chat, an agent id for the swarm."""
        ...

    def spec_versions(self, target: str) -> dict[str, str]:
        """Spec -> fingerprint of the specs that produce a record of `target`."""
        ...

    async def run_case(
        self,
        run_id: str,
        case: EvalCaseInfo,
        model: str,
        attempt: int,
        judge_model: str | None,
        on_step: StepCallback,
    ) -> EvalCaseResult:
        """Replay one eval case on the kind's agent and check it."""
        ...


class UnknownAgentKindError(KeyError):
    """No quality adapter is registered for this agent kind."""


_ADAPTERS: dict[str, QualityAdapter] = {}


def register_quality_adapter(adapter: QualityAdapter) -> None:
    _ADAPTERS[adapter.kind] = adapter
    logger.info(f"quality_adapter_001: Registered \033[36m{adapter.kind}\033[0m")


def quality_adapter(kind: str) -> QualityAdapter:
    try:
        return _ADAPTERS[kind]
    except KeyError:
        raise UnknownAgentKindError(kind) from None
