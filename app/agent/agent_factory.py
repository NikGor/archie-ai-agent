import logging
from typing import Any
from dotenv import load_dotenv
from ..config import MODEL_PROVIDERS, MAX_COMMAND_ITERATIONS
from ..models.orchestration_sgr import DecisionResponse
from ..models.output_models import AgentResponse
from ..tools.tool_factory import ToolFactory
from ..tools.create_output_tool import create_output
from ..utils.llm_parser import parse_llm_response
from ..utils.tool_executor import execute_tool_calls
from ..backend.openai_client import OpenAIClient
from ..backend.gemini_client import GeminiClient
from .prompt_builder import PromptBuilder
from ..backend.state_service import StateService


load_dotenv()
logger = logging.getLogger(__name__)


class AgentFactory:
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
        return "openai"

    async def _make_command_call(
        self,
        user_input: str,
        model: str,
        provider: str,
        user_state: dict[str, Any],
        response_format: str,
        previous_results: list[dict[str, Any]] | None = None,
    ) -> DecisionResponse:
        """
        Stage 1: Analyze request and decide action using cmd_prompt.

        Returns DecisionResponse with routing, action type, and tool calls.
        Can be called multiple times in a loop with previous_results from prior iterations.
        """
        logger.info("=== Stage 1: Command Decision ===")
        client = self.clients[provider]
        cmd_prompt_template = self.prompt_builder.env.get_template("cmd_prompt.jinja2")
        cmd_prompt = cmd_prompt_template.render(state=user_state)
        tools = self.tool_factory.get_tool_schemas(model, response_format)
        tools_list = "\n".join([f"- {t['name']}: {t['description']}" for t in tools])
        system_message = f"{cmd_prompt}\n\nAvailable Tools:\n{tools_list}"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
        ]
        if previous_results:
            results_context = "\n\nPrevious Tool Results:\n"
            for result in previous_results:
                tool_name = result.get("tool_name", "unknown")
                tool_output = result.get("output", {})
                results_context += f"- {tool_name}: {tool_output}\n"
            messages.append({"role": "assistant", "content": results_context})
            logger.info(
                f"agent_factory_008a: Added \033[33m{len(previous_results)}\033[0m previous results to context"
            )
        logger.info(f"agent_factory_008: Making command call with {provider}")
        raw_response = await client.create_completion(
            messages=messages,
            model=model,
            response_format=DecisionResponse,
        )
        parsed = parse_llm_response(
            raw_response=raw_response,
            provider=provider,
            expected_type=DecisionResponse,
        )
        logger.info(
            f"agent_factory_009: Decision made - Action: \033[36m{parsed.parsed_content.sgr.action.type}\033[0m"
        )

        return parsed.parsed_content

    async def arun(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4.1",
        response_format: str = "plain",
        previous_response_id: str | None = None,
        user_name: str | None = None,
    ) -> AgentResponse:
        """
        Main entry point: Create an agent response through 3-stage flow.

        Stage 1: Command - Analyze request and decide action
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
        user_input = messages[-1]["content"] if messages else ""
        tool_results = []
        command_history = []
        iteration = 0
        while iteration < MAX_COMMAND_ITERATIONS:
            iteration += 1
            logger.info(
                f"agent_factory_003a: Command iteration \033[33m{iteration}\033[0m"
            )

            # STAGE 1: Command - Analyze request and decide action
            decision = await self._make_command_call(
                user_input=user_input,
                model=model,
                provider=provider,
                user_state=user_state,
                response_format=response_format,
                previous_results=tool_results if tool_results else None,
            )
            logger.info(
                f"agent_factory_004: Action type: \033[36m{decision.sgr.action.type}\033[0m"
            )
            logger.info(
                f"agent_factory_005: Intent: \033[36m{decision.sgr.routing.intent}\033[0m"
            )
            command_history.append(
                {
                    "iteration": iteration,
                    "action_type": decision.sgr.action.type,
                    "intent": decision.sgr.routing.intent,
                    "reasoning": decision.sgr.reasoning,
                }
            )
            if decision.sgr.action.type != "function_call":
                logger.info(
                    f"agent_factory_005a: Exiting command loop - action type: \033[36m{decision.sgr.action.type}\033[0m"
                )
                break
            if decision.sgr.tool_calls:
                logger.info(
                    f"agent_factory_006: Executing \033[33m{len(decision.sgr.tool_calls)}\033[0m tools"
                )
                new_results = await execute_tool_calls(
                    tool_calls=decision.sgr.tool_calls,
                    tool_factory=self.tool_factory,
                )
                tool_results.extend(new_results)
                logger.info(
                    f"agent_factory_006a: Total tool results: \033[33m{len(tool_results)}\033[0m"
                )
            else:
                logger.warning(
                    "agent_factory_warning_001: function_call type but no tool_calls"
                )
                break
        if iteration >= MAX_COMMAND_ITERATIONS:
            logger.warning(
                f"agent_factory_warning_002: Reached max iterations ({MAX_COMMAND_ITERATIONS})"
            )
        command_summary = "\n\n".join(
            [
                f"Iteration {h['iteration']}: {h['action_type']} - {h['reasoning']}"
                for h in command_history
            ]
        )
        command_summary += f"\n\nTotal tools executed: {len(tool_results)}"
        logger.info("agent_factory_007: Creating final output")
        final_response = await create_output(
            user_input=user_input,
            command_summary=command_summary,
            tool_results=tool_results if tool_results else None,
            response_format=response_format,
            model=model,
            state=user_state,
        )
        logger.info("=== AgentFactory: Response Created ===")
        return final_response
