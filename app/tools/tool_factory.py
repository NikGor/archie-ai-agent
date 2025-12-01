"""Factory for managing and executing tools."""

import inspect
import logging
import importlib
from typing import Any, Callable
from ..config import MODEL_PROVIDERS, TOOLS_CONFIG
from ..utils.tools_utils import openai_parse, gemini_parse, oss_parse


logger = logging.getLogger(__name__)


class ToolFactory:
    """Factory for registering and executing agent tools."""

    def __init__(self, demo_mode: bool = False):
        self.tools: dict[str, Callable] = {}
        self.tools_config = TOOLS_CONFIG
        self.model_providers = MODEL_PROVIDERS
        self.demo_mode = demo_mode
        logger.info(
            f"tool_factory_001: Initialized with \033[33m{len(self.tools_config)}\033[0m tool groups, "
            f"demo_mode: \033[35m{demo_mode}\033[0m"
        )

    def _load_tool_function(self, module_path: str) -> Callable | None:
        """Dynamically import and return tool function from module path."""
        try:
            module = importlib.import_module(module_path)
            module_name = module_path.split(".")[-1]
            if hasattr(module, module_name):
                func = getattr(module, module_name)
                if callable(func):
                    logger.info(
                        f"tool_factory_002: Loaded function \033[36m{module_name}\033[0m from \033[36m{module_path}\033[0m"
                    )
                    return func
            logger.warning(
                f"tool_factory_warning_001: Function \033[33m{module_name}\033[0m not found in module"
            )
            return None
        except Exception as e:
            logger.error(
                f"tool_factory_error_002: Failed to import \033[31m{module_path}\033[0m: {e}"
            )
            return None

    def get_tools_for_response_format(self, response_format: str) -> list[str]:
        """Get list of tool groups based on response format."""
        if response_format in ["dashboard", "widget"]:
            logger.info(
                f"tool_factory_003: Response format \033[36m{response_format}\033[0m -> using 'smarthome' tools"
            )
            return ["smarthome"]

        # Default: all groups including smarthome
        all_groups = list(self.tools_config.keys())
        logger.info(
            f"tool_factory_004: Response format \033[36m{response_format}\033[0m -> using groups: \033[33m{all_groups}\033[0m"
        )
        return all_groups

    def _get_provider_for_model(self, model: str) -> str:
        """Get provider name for a given model."""
        for provider, models in self.model_providers.items():
            if model in models:
                return provider
        return "openai"  # default fallback

    def get_tool_schemas(
        self, model: str, response_format: str
    ) -> list[dict[str, Any]]:
        """Get tool schemas formatted for specific model."""
        logger.info(
            f"tool_factory_005: Building schemas for model \033[36m{model}\033[0m, format \033[36m{response_format}\033[0m"
        )

        # Determine which tool groups to include
        enabled_groups = self.get_tools_for_response_format(response_format)

        # Collect tool functions from enabled groups
        tool_functions: list[Callable] = []
        for group_name in enabled_groups:
            if group_name not in self.tools_config:
                logger.warning(
                    f"tool_factory_warning_002: Group \033[33m{group_name}\033[0m not found in config"
                )
                continue

            group_tools = self.tools_config[group_name]
            for tool_name, module_path in group_tools.items():
                func = self._load_tool_function(module_path)
                if func:
                    tool_functions.append(func)
                    # Register for execution
                    self.tools[tool_name] = func
                    self.tools[func.__name__] = func

        # Determine provider based on model
        provider = self._get_provider_for_model(model)

        # Choose parser based on provider
        if provider == "openai":
            parser = openai_parse
            parser_name = "OpenAI"
        elif provider == "gemini":
            parser = gemini_parse
            parser_name = "Gemini"
        else:
            parser = oss_parse
            parser_name = "OSS"

        logger.info(
            f"tool_factory_006: Using \033[36m{parser_name}\033[0m parser for \033[33m{len(tool_functions)}\033[0m tools"
        )

        # Parse all functions into schemas
        schemas = []
        for func in tool_functions:
            try:
                schema = parser(func)
                schemas.append(schema)
                logger.info(
                    f"tool_factory_007: Parsed schema for \033[36m{schema['name']}\033[0m"
                )
            except Exception as e:
                logger.error(
                    f"tool_factory_error_003: Failed to parse \033[31m{func.__name__}\033[0m: {e}"
                )

        return schemas

    async def execute_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> Any:
        """Execute a tool by name with given arguments."""
        logger.info(
            f"tool_factory_008: Executing \033[36m{tool_name}\033[0m with args: {tool_arguments}"
        )
        if tool_name not in self.tools:
            logger.error(
                f"tool_factory_error_004: Unknown tool: \033[31m{tool_name}\033[0m"
            )
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            tool_func = self.tools[tool_name]
            sig = inspect.signature(tool_func)
            if "demo_mode" in sig.parameters:
                tool_arguments["demo_mode"] = self.demo_mode
                logger.info(
                    f"tool_factory_008a: Injected demo_mode=\033[35m{self.demo_mode}\033[0m"
                )
            result = await tool_func(**tool_arguments)
            logger.info(
                f"tool_factory_009: Tool \033[36m{tool_name}\033[0m executed successfully"
            )
            return result
        except Exception as e:
            logger.error(
                f"tool_factory_error_005: Tool execution failed: \033[31m{e}\033[0m"
            )
            return {"error": f"Tool execution failed: {e!s}"}
