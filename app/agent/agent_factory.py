"""Agent factory for orchestrating AI agent components."""

import json
import logging
import os
from typing import Any
from dotenv import load_dotenv
from ..models.response_models import AgentResponse
from ..tools.tool_factory import ToolFactory
from ..utils.openai_utils import create_llm_trace_from_openai_response
from .openai_client import OpenAIClient
from .gemini_client import GeminiClient
from .prompt_builder import PromptBuilder
from .state_service import StateService


load_dotenv()
logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating and orchestrating AI agent responses."""

    MODEL_PROVIDERS = {
        "openai": [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-5",
            "gpt-5.1",
            "gpt-5-mini",
            "gpt-5-nano",
        ],
        "gemini": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
    }

    def __init__(
        self,
        prompt_builder: PromptBuilder | None = None,
        tool_factory: ToolFactory | None = None,
        state_service: StateService | None = None,
    ):
        self.openai_client = OpenAIClient()
        self.gemini_client = GeminiClient()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.tool_factory = tool_factory or ToolFactory()
        self.state_service = state_service or StateService()

        # Create clients dictionary for easy provider switching
        self.clients = {
            "openai": self.openai_client,
            "gemini": self.gemini_client,
        }

        logger.info("agent_factory_001: Initialized AgentFactory")

    def _get_provider_for_model(self, model: str) -> str:
        """Returns 'openai' or 'gemini' based on model name."""
        for provider, models in self.MODEL_PROVIDERS.items():
            if model in models:
                return provider
        return "openai"  # default fallback

    async def create_agent_response(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4.1",
        response_format: str = "plain",
        previous_response_id: str | None = None,
        user_name: str | None = None,
    ) -> AgentResponse:
        """Create an agent response by orchestrating all components."""
        logger.info("=== AgentFactory: Creating Agent Response ===")

        # Determine provider and select client
        provider = self._get_provider_for_model(model)
        client = self.clients[provider]
        logger.info(
            f"agent_factory_001b: Using provider: \033[34m{provider}\033[0m for model: \033[36m{model}\033[0m"
        )

        if user_name:
            self.state_service.user_name = user_name
            logger.info(
                f"agent_factory_001c: Set user_name: \033[35m{user_name}\033[0m"
            )

        user_state = self.state_service.get_user_state()

        # Persona comes from Redis or defaults to business
        persona_key = user_state.get("persona", "business")
        logger.info(f"agent_factory_002: Persona: \033[35m{persona_key}\033[0m")
        logger.info(f"agent_factory_003: Format: \033[36m{response_format}\033[0m")
        full_system_prompt = self.prompt_builder.build_full_prompt(
            persona=persona_key,
            response_format=response_format,
            state=user_state,
        )
        formatted_messages = []
        formatted_messages.append({"role": "system", "content": full_system_prompt})
        formatted_messages.extend(messages)
        logger.info(
            f"agent_factory_004: Prepared messages - System: 1, User/Assistant: \033[33m{len(messages)}\033[0m, Total: \033[33m{len(formatted_messages)}\033[0m"
        )
        is_dashboard = response_format == "dashboard"
        tools = self.tool_factory.get_tool_definitions(for_dashboard=is_dashboard)
        tool_choice = "required" if is_dashboard else "auto"
        if is_dashboard:
            logger.info("agent_factory_004b: Dashboard format - tool call required")
        response = await client.create_completion(
            messages=formatted_messages,
            model=model,
            response_format=AgentResponse,
            tools=tools,
            tool_choice=tool_choice,
            previous_response_id=previous_response_id,
        )

        # Handle response (unified for OpenAI and Gemini)
        if response.output[0].type == "function_call":
            response = await self._handle_function_call(
                response, formatted_messages, model
            )
        elif response.output[0].type == "web_search_call":
            response = await self._handle_web_search_call(
                response, formatted_messages, model
            )
        parsed_result = response.output[0].content[0].parsed
        llm_trace = create_llm_trace_from_openai_response(response)
        response_id = response.id if hasattr(response, "id") else None

        result = AgentResponse(
            content=parsed_result.content,
            sgr=parsed_result.sgr,
            llm_trace=llm_trace,
        )
        if response_id:
            result.response_id = response_id
        self._log_response(result)
        logger.info("=== AgentFactory: Response Created ===")
        return result

    async def _handle_function_call(
        self,
        response: Any,
        messages: list[dict[str, Any]],
        model: str,
    ) -> Any:
        """Handle function call response from OpenAI."""
        logger.info("agent_factory_005: Handling function call")
        tool_arguments = json.loads(response.output[0].arguments)
        tool_name = response.output[0].name
        tool_result = await self.tool_factory.execute_tool(tool_name, tool_arguments)
        logger.info(f"agent_factory_006: Tool result: {tool_result}")
        messages.append(
            {
                "role": "assistant",
                "content": f"Function Result: {tool_name}: {tool_result}",
            }
        )
        logger.info(
            f"agent_factory_006b: Added tool result, new total: \033[33m{len(messages)}\033[0m"
        )
        return await self.openai_client.create_completion(
            messages=messages,
            model=model,
            response_format=AgentResponse,
        )

    async def _handle_web_search_call(
        self,
        response: Any,
        messages: list[dict[str, Any]],
        model: str,
    ) -> Any:
        """Handle web search call response from OpenAI."""
        logger.info("agent_factory_007: Handling web search call")
        for output_item in response.output:
            if output_item.type == "message":
                messages.append(
                    {
                        "role": "assistant",
                        "content": output_item.content[0].text,
                    }
                )
                logger.info(
                    f"agent_factory_007b: Added web search result, new total: \033[33m{len(messages)}\033[0m"
                )
                break
        return await self.openai_client.create_completion(
            messages=messages,
            model=model,
            response_format=AgentResponse,
        )

    def _log_response(self, result: AgentResponse) -> None:
        """Log the agent response."""
        content_text = str(result.content) if result.content else ""
        logger.info(
            f"agent_factory_008: Response length: \033[33m{len(content_text)}\033[0m"
        )

        if result.sgr and result.sgr.reasoning:
            logger.info(
                f"agent_factory_010: Reasoning:\n\033[35m{result.sgr.reasoning}\033[0m"
            )
        if result.sgr and result.sgr.ui_reasoning:
            logger.info(
                f"agent_factory_011: UI Reasoning:\n\033[35m{result.sgr.ui_reasoning}\033[0m"
            )

        try:
            response_dict = {
                "content": (
                    result.content.model_dump(mode="json") if result.content else None
                ),
            }
            logger.info(
                f"agent_factory_009: Full response:\n\033[32m{json.dumps(response_dict, indent=2, ensure_ascii=False)}\033[0m"
            )
        except Exception as e:
            logger.warning(
                f"agent_factory_warning_001: Could not serialize response: {e}"
            )
