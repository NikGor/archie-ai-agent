"""Registry of tools (ARCHIE-292).

A tool is an async function `name(...) -> dict` in a module whose last path
segment is the same name. Tools are registered in groups; a package that owns
tools only it may use (e.g. the swarm) registers its group as exclusive, so
other agent kinds do not see it. Which tools an agent sees is decided by the
agent kind (chat: by response format, swarm: by agent profile), not here.
"""

import importlib
import inspect
import logging
from collections.abc import Callable, Iterable
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator
from ..llm.catalog import get_provider_for_model
from ..llm.registry import get_provider
from .models import ToolContext


logger = logging.getLogger(__name__)


class ToolCapability(BaseModel):
    """What a tool does to the world (ARCHIE-320). Policies read this, not lists."""

    effect: Literal["read", "write"] = Field(
        description="read: only reads data; write: has side effects"
    )
    read_actions: tuple[str, ...] = Field(
        default=(),
        description="Values of the `action` argument that only read, for a write tool",
    )
    approval: bool = Field(
        default=False,
        description="A side effect needs the user's approval before it runs",
    )


READ = ToolCapability(effect="read")
WRITE = ToolCapability(effect="write", approval=True)


class ToolGroup(BaseModel):
    """Tools registered together: tool name -> module path."""

    name: str = Field(description="Group name, e.g. smarthome")
    tools: dict[str, str] = Field(description="Tool name -> module path")
    capabilities: dict[str, ToolCapability] = Field(
        description="Tool name -> capability; every tool of the group must have one"
    )
    exclusive: bool = Field(
        default=False,
        description="Only the agent kind that registered it may see these tools",
    )

    @model_validator(mode="after")
    def every_tool_has_capability(self) -> "ToolGroup":
        if set(self.capabilities) != set(self.tools):
            raise ValueError(
                f"Group {self.name}: capabilities must match tools exactly"
            )
        return self


class UnknownToolError(KeyError):
    """No registered tool has this name."""


class ToolRegistry:
    def __init__(self, groups: Iterable[ToolGroup] = ()) -> None:
        self.groups: dict[str, ToolGroup] = {}
        self._functions: dict[str, Callable] = {}
        for group in groups:
            self.add_group(group)

    def add_group(self, group: ToolGroup) -> None:
        """Register a group; registering the same group again replaces it."""
        self.groups[group.name] = group
        logger.info(
            f"tool_registry_001: Group \033[36m{group.name}\033[0m: \033[33m{len(group.tools)}\033[0m tools"
            + (" (exclusive)" if group.exclusive else "")
        )

    def names(
        self, groups: Iterable[str] | None = None, include_exclusive: bool = True
    ) -> list[str]:
        """Tool names in registration order, optionally of some groups only."""
        wanted = set(groups) if groups is not None else None
        return [
            name
            for group in self.groups.values()
            if (wanted is None or group.name in wanted)
            and (include_exclusive or not group.exclusive)
            for name in group.tools
        ]

    def capability(self, name: str) -> ToolCapability:
        for group in self.groups.values():
            if name in group.capabilities:
                return group.capabilities[name]
        raise UnknownToolError(name)

    def is_read_only(self, name: str, arguments: dict[str, str]) -> bool:
        """True when this call only reads data; unknown tools are not read-only."""
        try:
            capability = self.capability(name)
        except UnknownToolError:
            return False
        return (
            capability.effect == "read"
            or arguments.get("action") in capability.read_actions
        )

    def module_path(self, name: str) -> str:
        for group in self.groups.values():
            if name in group.tools:
                return group.tools[name]
        raise UnknownToolError(name)

    def function(self, name: str) -> Callable:
        """The tool function, imported on first use."""
        if name not in self._functions:
            module_path = self.module_path(name)
            module = importlib.import_module(module_path)
            self._functions[name] = getattr(module, module_path.rsplit(".", 1)[-1])
            logger.info(
                f"tool_registry_002: Loaded \033[36m{name}\033[0m from \033[36m{module_path}\033[0m"
            )
        return self._functions[name]

    def schemas(self, model: str, names: Iterable[str]) -> list[dict[str, Any]]:
        """Tool schemas in the format of the provider that serves `model`.

        Order follows the registry, not `names`, so the prompt prefix stays
        stable for prompt caching.
        """
        wanted = set(names)
        provider = get_provider(get_provider_for_model(model))
        schemas: list[dict[str, Any]] = []
        for name in self.names():
            if name not in wanted:
                continue
            try:
                schemas.append(provider.tool_schema(self.function(name)))
            except Exception as e:
                logger.error(
                    f"tool_registry_error_001: No schema for \033[31m{name}\033[0m: {e}"
                )
        logger.info(
            f"tool_registry_003: \033[33m{len(schemas)}\033[0m {provider.name} tool schemas for \033[36m{model}\033[0m"
        )
        return schemas

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Run a tool. Fields of `context` (timezone, demo_mode) go to the tool
        only if its signature declares them. Errors come back as {"error": ...}."""
        logger.info(
            f"tool_registry_004: Executing \033[36m{name}\033[0m with args: {arguments}"
        )
        try:
            func = self.function(name)
        except (UnknownToolError, ImportError, AttributeError) as e:
            logger.error(
                f"tool_registry_error_002: Unknown tool: \033[31m{name}\033[0m ({e})"
            )
            return {"error": f"Unknown tool: {name}"}
        parameters = inspect.signature(func).parameters
        facts = context.model_dump() if context else {"demo_mode": False}
        injected = {key: value for key, value in facts.items() if key in parameters}
        if injected:
            logger.info(
                f"tool_registry_005: Injected context: \033[35m{injected}\033[0m"
            )
        try:
            return await func(**{**arguments, **injected})
        except Exception as e:
            logger.error(
                f"tool_registry_error_003: Tool execution failed: \033[31m{e}\033[0m"
            )
            return {"error": f"Tool execution failed: {e!s}"}
