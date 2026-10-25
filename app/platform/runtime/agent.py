"""The entry point implemented by independently registered agent kinds."""

from typing import Protocol
from pydantic import BaseModel, Field
from app.platform.runtime.budget import Trigger, TriggerKind


class RunContext(BaseModel):
    run_id: str = Field(description="ID assigned by the runtime")
    parent_run_id: str | None = Field(None, description="Run that caused this run")
    depth: int = Field(0, ge=0, description="Trusted spawn depth")
    user_timezone: str = Field("UTC", description="Owner timezone")
    demo_mode: bool = Field(False, description="Use simulated reads for evals")
    payload: str = Field("", description="Kind-specific validated JSON request")


class AgentKind(Protocol):
    name: str
    triggers: tuple[TriggerKind, ...]

    async def run(self, trigger: Trigger, context: RunContext) -> BaseModel: ...
