"""Agent factory for orchestrating AI agent components."""

import logging
from typing import Any
from dotenv import load_dotenv
from ..config import MODEL_PROVIDERS
from ..models.orchestration_sgr import DecisionResponse
from ..models.output_models import AgentResponse
from ..tools.tool_factory import ToolFactory
from ..tools.create_output_tool import create_output
from ..utils.llm_parser import parse_llm_response
from ..utils.tool_executor import execute_tool_calls_parallel
from .openai_client import OpenAIClient
from .gemini_client import GeminiClient
from .prompt_builder import PromptBuilder
from .state_service import StateService


load_dotenv()
logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating and orchestrating AI agent responses."""

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
        self.model_providers = MODEL_PROVIDERS

        # Create clients dictionary for easy provider switching
        self.clients = {
            "openai": self.openai_client,
            "gemini": self.gemini_client,
        }

        logger.info("agent_factory_001: Initialized AgentFactory")

    def _get_provider_for_model(self, model: str) -> str:
        """Returns 'openai' or 'gemini' based on model name."""
        for provider, models in self.model_providers.items():
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
        """
        Create an agent response through 3-stage orchestration.

        Stage 1: Decision-making (orchestration)
        Stage 2: Tool execution (if needed)
        Stage 3: Final response generation
        """
        logger.info("=== AgentFactory: Creating Agent Response ===")

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
        persona_key = user_state.get("persona", "business")
        logger.info(f"agent_factory_002: Persona: \033[35m{persona_key}\033[0m")
        logger.info(f"agent_factory_003: Format: \033[36m{response_format}\033[0m")

        # Extract user input from messages
        user_input = messages[-1]["content"] if messages else ""

        # STAGE 1: Orchestration - Make decision about what to do
        decision = await self._make_orchestration_decision(
            user_input=user_input,
            model=model,
            provider=provider,
            user_state=user_state,
            response_format=response_format,
        )

        logger.info(
            f"agent_factory_004: Action type: \033[36m{decision.sgr.action.type}\033[0m"
        )
        logger.info(
            f"agent_factory_005: Intent: \033[36m{decision.sgr.routing.intent}\033[0m"
        )

        tool_results = []
        orchestration_summary = decision.sgr.reasoning

        # STAGE 2: Execute tools if needed
        if decision.sgr.action.type == "function_call":
            if decision.sgr.tool_calls:
                logger.info(
                    f"agent_factory_006: Executing \033[33m{len(decision.sgr.tool_calls)}\033[0m tools"
                )
                tool_results = await execute_tool_calls_parallel(
                    tool_calls=decision.sgr.tool_calls,
                    tool_factory=self.tool_factory,
                )

                # Check if need to make another orchestration decision
                # (for now, go directly to final response)
                orchestration_summary += f"\n\nTools executed: {len(tool_results)}"
            else:
                logger.warning(
                    "agent_factory_warning_001: function_call type but no tool_calls"
                )

        # STAGE 3: Generate final response
        logger.info("agent_factory_007: Creating final output")
        final_response = await create_output(
            user_input=user_input,
            orchestration_summary=orchestration_summary,
            tool_results=tool_results if tool_results else None,
            response_format=response_format,
            model=model,
            state=user_state,
        )

        logger.info("=== AgentFactory: Response Created ===")
        return final_response

    async def _make_orchestration_decision(
        self,
        user_input: str,
        model: str,
        provider: str,
        user_state: dict[str, Any],
        response_format: str,
    ) -> DecisionResponse:
        """
        Stage 1: Make orchestration decision using cmd_prompt.

        Returns DecisionResponse with routing, action type, and tool calls.
        """
        logger.info("=== Stage 1: Orchestration Decision ===")

        client = self.clients[provider]

        # Build command prompt for orchestration
        cmd_prompt_template = self.prompt_builder.env.get_template("cmd_prompt.jinja2")
        cmd_prompt = cmd_prompt_template.render(state=user_state)

        # Get tool schemas for this response format
        tools = self.tool_factory.get_tool_schemas(model, response_format)
        tools_list = "\n".join([f"- {t['name']}: {t['description']}" for t in tools])

        system_message = f"{cmd_prompt}\n\nAvailable Tools:\n{tools_list}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
        ]

        logger.info(f"agent_factory_008: Making orchestration decision with {provider}")

        raw_response = await client.create_completion(
            messages=messages,
            model=model,
            response_format=DecisionResponse,
            tools=None,  # No tools for orchestration decision
        )

        # Parse response
        parsed = parse_llm_response(
            raw_response=raw_response,
            provider=provider,
            expected_type=DecisionResponse,
        )

        logger.info(
            f"agent_factory_009: Decision made - Action: \033[36m{parsed.parsed_content.sgr.action.type}\033[0m"
        )

        return parsed.parsed_content
