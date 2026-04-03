"""Prompt builder for constructing system and assistant prompts."""

import json
import logging
import os
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..models.tool_models import ToolResult


logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builder for constructing prompts from Jinja2 templates."""

    def __init__(self, templates_dir: str | None = None):
        if templates_dir is None:
            templates_dir = os.path.join(os.path.dirname(__file__), "prompts")
        self.templates_dir = templates_dir
        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=select_autoescape(enabled_extensions=("jinja2",)),
        )
        logger.info(
            f"prompt_builder_001: Initialized with templates: \033[36m{templates_dir}\033[0m"
        )

    def build_command_messages(
        self,
        user_input: str,
        state: dict,
        tools: list[dict[str, Any]],
        provider: str,
        previous_results: list[ToolResult] | None = None,
        chat_history: str | None = None,
    ) -> list[dict[str, str]]:
        """Build the full message list for Stage 1 command call."""
        cmd_prompt_template = self.env.get_template("cmd_prompt.jinja2")
        cmd_prompt = cmd_prompt_template.render(state=state)
        tools_list = "\n".join(
            [
                f"- {t['name']}: {t.get('description', '')}\n  Parameters: {json.dumps(t.get('parameters', {}), ensure_ascii=False)}"
                for t in tools
            ]
        )
        system_message = f"{cmd_prompt}\n\nAvailable Tools:\n{tools_list}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
        ]
        if chat_history and provider != "openai":
            messages.insert(
                1,
                {"role": "system", "content": f"Chat History:\n{chat_history}"},
            )
            logger.info(
                f"prompt_builder_010: Added chat_history to context (len: \033[33m{len(chat_history)}\033[0m)"
            )
        if previous_results:
            results_context = "\n\nPrevious Tool Results:\n"
            for result in previous_results:
                results_context += f"- {result.tool_name}: {result.output}\n"
            messages.append({"role": "assistant", "content": results_context})
            logger.info(
                f"prompt_builder_011: Added \033[33m{len(previous_results)}\033[0m previous results to context"
            )
        return messages

    def build_assistant_prompt(
        self,
        state: dict,
        response_format: str,
    ) -> str:
        """Build assistant context prompt from template."""
        logger.info("prompt_builder_003: Building assistant prompt")
        template = self.env.get_template("assistant_prompt.jinja2")
        return template.render(
            state=state,
            response_format=response_format,
        )

    def build_format_instructions(
        self, response_format: str, intents: list[str] | None = None
    ) -> str:
        """Build format-specific instructions."""
        format_aliases = {"voice": "plain"}
        actual_format = format_aliases.get(response_format, response_format)
        if actual_format in [
            "plain",
            "ui_answer",
            "level2_answer",
            "level3_answer",
            "dashboard",
            "widget",
        ]:
            format_template_name = f"format_prompts/format_{actual_format}.jinja2"
        else:
            format_template_name = "format_prompts/format_formatted_text.jinja2"
        format_template_path = os.path.join(self.templates_dir, format_template_name)
        if not os.path.exists(format_template_path):
            logger.warning(
                f"prompt_builder_004: Format template missing: \033[31m{format_template_name}\033[0m"
            )
            return ""
        logger.info(
            f"prompt_builder_005: Loaded format template: \033[36m{format_template_name}\033[0m"
        )
        template = self.env.get_template(format_template_name)
        return template.render(response_format=response_format, intents=intents or [])
