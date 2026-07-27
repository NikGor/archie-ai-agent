"""Per-stage timing accumulation for one arun() call."""

from dataclasses import dataclass, field
from archie_shared.chat.models import LllmTrace, PipelineTrace
from ..utils.trace_utils import build_pipeline_trace


@dataclass
class PipelineMetrics:
    """Replaces the manually threaded stage1/stage2 accumulator variables
    that used to live as local variables inside AgentFactory.arun()."""

    stage1_duration_ms: int = 0
    stage1_llm_traces: list[LllmTrace] = field(default_factory=list)
    stage2_duration_ms: int = 0

    def add_stage1(self, duration_ms: int, llm_trace: LllmTrace | None) -> None:
        self.stage1_duration_ms += duration_ms
        if llm_trace:
            self.stage1_llm_traces.append(llm_trace)

    def add_stage2(self, duration_ms: int) -> None:
        self.stage2_duration_ms += duration_ms

    def build_trace(
        self,
        total_ms: int,
        stage3_duration_ms: int,
        stage3_llm_trace: LllmTrace | None,
        stage3_ttft_ms: int | None,
    ) -> PipelineTrace:
        return build_pipeline_trace(
            total_ms=total_ms,
            stage3_duration_ms=stage3_duration_ms,
            stage3_llm_trace=stage3_llm_trace,
            stage3_ttft_ms=stage3_ttft_ms,
            stage1_duration_ms=self.stage1_duration_ms,
            stage1_llm_traces=self.stage1_llm_traces,
            stage2_duration_ms=self.stage2_duration_ms,
        )
