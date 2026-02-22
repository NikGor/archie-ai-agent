"""Prompt builder for constructing system and assistant prompts."""

import logging
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape


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

    def build_format_instructions(self, response_format: str) -> str:
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
        return template.render(response_format=response_format)
