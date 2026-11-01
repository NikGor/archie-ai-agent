"""AgentSystemComposer (ARCHIE-340): task -> proposal of an agent system ->
checked AgentSystemSpec. The LLM proposes the topology and every agent;
`check_system` decides what may run. One repeat gets the errors back.
"""

import json
import logging
from typing import Literal
from pydantic import BaseModel, Field
from app.agents.ephemeral.configurator import AgentConfigurator, SpecDraft, to_spec
from app.config import settings
from app.platform.llm.profiles import select_model
from app.platform.prompts.compiler import PromptCompiler
from app.platform.runtime.budget import Budget, BudgetExceededError, BudgetMeter
from app.platform.runtime.completion import (
    BudgetEstimateError,
    CompletionCall,
    InvalidOutputError,
    complete,
    output_cap,
)
from app.platform.runtime.spec import CallerPolicy
from app.platform.runtime.system import (
    AgentSystemSpec,
    NodeRole,
    Protocol,
    SystemBudget,
    SystemCheck,
    Topology,
    check_system,
)


logger = logging.getLogger(__name__)

PROMPT = "composer"
COMPOSER_BUDGET = Budget(max_tokens=80000, max_cost_usd=0.15, max_seconds=120)
# Up to six agent drafts in one reply
COMPOSER_OUTPUT_TOKENS = 12000


class NodeDraft(BaseModel):
    id: str = Field(description="Node slug: lowercase letters, digits, _")
    role: NodeRole = Field(description="Role that fits the topology")
    step: str = Field(description="What this node does with the task")
    depends_on: list[str] = Field(description="Workflow only: ids of input nodes")
    spec: SpecDraft


class SystemDraft(BaseModel):
    name: str = Field(description="System slug: lowercase letters, digits, _")
    topology: Topology = Field(description="Control topology")
    reason: str = Field(description="Why this topology and these agents")
    protocol: Protocol = Field(description="Decentralized only; else independent")
    debate_rounds: int = Field(description="Debate only: 1 or 2; else 0")
    nodes: list[NodeDraft] = Field(description="1 to 6 nodes")


class CompositionResult(BaseModel):
    reason: str = Field("", description="Why this system, or the failure")
    spec: AgentSystemSpec | None = Field(None, description="Last proposal")
    check: SystemCheck | None = Field(None, description="Validator verdict")
    errors: list[str] = Field(default_factory=list, description="Call errors")
    attempts: int = 0
    model: str = ""
    cost_usd: float = 0.0
    prompt_version: str = ""


def to_system(draft: SystemDraft, task: str, policy: CallerPolicy) -> dict:
    proposes = any(n.spec.permissions_mode == "propose_actions" for n in draft.nodes)
    # Every node gets at most its share of the default system budget
    share = SystemBudget().max_cost_usd / max(1, len(draft.nodes))
    return {
        "name": draft.name,
        "description": draft.reason[:500],
        "topology": draft.topology,
        "autonomy": {"external_actions": "propose" if proposes else "none"},
        "coordination": {
            "protocol": draft.protocol,
            "debate_rounds": max(0, min(2, draft.debate_rounds)),
        },
        "task": task,
        "nodes": [
            {
                "id": n.id,
                "role": n.role,
                "step": n.step,
                "depends_on": n.depends_on,
                "spec": to_spec(
                    n.spec.model_copy(
                        update={"max_cost_usd": min(n.spec.max_cost_usd, share)}
                    ),
                    n.step or task,
                    policy.limits.timeout_seconds,
                ),
            }
            for n in draft.nodes
        ],
    }


class AgentSystemComposer:
    def __init__(
        self,
        configurator: AgentConfigurator,
        call: CompletionCall | None = None,
        compiler: PromptCompiler | None = None,
    ) -> None:
        self.configurator = configurator
        self.call = call or complete
        self.compiler = compiler or PromptCompiler()

    def messages(
        self, task: str, policy: CallerPolicy, topology: str | None
    ) -> list[dict[str, str]]:
        system = (
            self.compiler.compile(PROMPT)
            + "\nCapabilities:\n"
            + json.dumps(self.configurator.capabilities(policy), ensure_ascii=False)
        )
        if topology:
            system += f"\nThe user chose topology {topology}: use it."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"task": task}, ensure_ascii=False)},
        ]

    async def compose(
        self,
        task: str,
        policy: CallerPolicy,
        skills: list[str],
        topology: Literal["centralized", "workflow", "hierarchical", "decentralized"]
        | None = None,
    ) -> CompositionResult:
        selection = select_model(settings.agent_configurator_profile, "low")
        meter = BudgetMeter(COMPOSER_BUDGET)
        messages = self.messages(task, policy, topology)
        result = CompositionResult(
            model=selection.model, prompt_version=self.compiler.fingerprint(PROMPT)
        )
        for attempt in range(2):
            result.attempts = attempt + 1
            try:
                cap = output_cap(
                    meter,
                    messages,
                    SystemDraft,
                    selection.model,
                    COMPOSER_OUTPUT_TOKENS,
                )
                completion = await self.call(
                    messages,
                    SystemDraft,
                    selection.model,
                    cap,
                    **({"effort": selection.effort} if selection.effort else {}),
                )
            except (BudgetExceededError, BudgetEstimateError) as exc:
                result.errors = [str(exc)]
                break
            except InvalidOutputError as exc:
                self._count(result, meter, exc.trace)
                result.errors = [str(exc)]
                messages.append(
                    {"role": "user", "content": f"Invalid JSON ({exc}). Reply again."}
                )
                continue
            self._count(result, meter, completion.trace)
            draft = completion.content
            assert isinstance(draft, SystemDraft)
            result.reason = draft.reason
            check = check_system(
                to_system(draft, task, policy),
                policy,
                self.configurator.registry,
                skills,
            )
            result.check = check
            result.spec = check.spec
            result.errors = []
            if check.valid:
                logger.info(
                    f"agent_composer_001: {draft.topology} with {len(draft.nodes)} nodes"
                )
                return result
            logger.warning(
                f"agent_composer_warning_001: attempt {attempt + 1}: {check.errors}"
            )
            messages.append({"role": "assistant", "content": draft.model_dump_json()})
            messages.append(
                {
                    "role": "user",
                    "content": f"Validation failed: {'; '.join(check.errors)}. Fix the system.",
                }
            )
        result.reason = result.reason or "no valid system"
        return result

    @staticmethod
    def _count(result: CompositionResult, meter: BudgetMeter, trace) -> None:
        if trace is None:
            return
        result.cost_usd += trace.total_cost or 0.0
        meter.add(trace)
