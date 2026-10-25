"""Prompt builder for constructing system and assistant prompts."""

import json
import logging
from typing import Any
from archie_shared.chat.models import ImageAttachment
from app.platform.llm.registry import get_provider
from app.platform.prompts.compiler import PromptCompiler
from app.platform.prompts.context import datetime_note
from app.platform.tools.models import ToolResult
from app.utils.skill_utils import list_skills


# Header of the skill list in the Stage 1 prompt; chat monitoring finds it in traces
SKILLS_HEADER = (
    "Available Skills (call skill_loader_tool with skill_name to load "
    "full instructions):"
)


logger = logging.getLogger(__name__)


class ChatPromptBuilder:
    """Builds the message lists of each stage from compiled prompt specs (ARCHIE-212)."""

    def __init__(self, compiler: PromptCompiler | None = None):
        self.compiler = compiler or PromptCompiler()

    def build_command_messages(
        self,
        user_input: str,
        state: dict,
        tools: list[dict[str, Any]],
        provider: str,
        response_format: str,
        previous_results: list[ToolResult] | None = None,
        chat_history: str | None = None,
        images: list[ImageAttachment] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the full message list for Stage 1 command call.

        Order keeps a stable, cacheable prefix (static instructions → chat
        history) ahead of the volatile tail (datetime, tool results, query) so
        prompt caching stays valid across turns (ARCHIE-180).
        """
        cmd_prompt = self.compiler.compile(
            "cmd_prompt", state=state, response_format=response_format
        )
        tools_list = "\n".join(
            [
                f"- {t['name']}: {t.get('description', '')}\n  Parameters: {json.dumps(t.get('parameters', {}), ensure_ascii=False)}"
                for t in tools
            ]
        )
        system_message = f"{cmd_prompt}\n\nAvailable Tools:\n{tools_list}"
        skills = list_skills()
        if skills:
            skills_list = "\n".join(
                f"- {s['name']}: {s['description']}" for s in skills
            )
            system_message += f"\n\n{SKILLS_HEADER}\n{skills_list}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_message, "cacheable": True},
        ]
        if chat_history:
            messages.append(
                {
                    "role": "system",
                    "content": f"Chat History:\n{chat_history}",
                    "cacheable": True,
                }
            )
            logger.info(
                f"prompt_builder_010: Added chat_history to context (len: \033[33m{len(chat_history)}\033[0m)"
            )
        volatile = datetime_note(state)
        if previous_results:
            volatile += "\n\nPrevious Tool Results:\n"
            for result in previous_results:
                volatile += f"- {result.tool_name}: {result.output}\n"
            logger.info(
                f"prompt_builder_011: Added \033[33m{len(previous_results)}\033[0m previous results to context"
            )
        messages.append({"role": "system", "content": volatile})
        messages.append(
            {
                "role": "user",
                "content": get_provider(provider).user_content(user_input, images),
            }
        )
        return messages

    def build_assistant_prompt(
        self,
        state: dict,
        response_format: str,
    ) -> str:
        """Build the assistant context prompt (user state and persona)."""
        logger.info("prompt_builder_003: Building assistant prompt")
        return self.compiler.compile(
            "assistant_prompt", state=state, response_format=response_format
        )

    @staticmethod
    def format_spec_name(response_format: str) -> str:
        """Return the format spec name for a response format."""
        format_aliases = {"voice": "plain"}
        actual_format = format_aliases.get(response_format, response_format)
        if actual_format in [
            "plain",
            "gemini_tts",
            "ui_answer",
            "level2_answer",
            "level3_answer",
            "dashboard",
            "widget",
        ]:
            return f"format_{actual_format}"
        return "format_formatted_text"

    def build_format_instructions(
        self, response_format: str, intents: list[str] | None = None
    ) -> str:
        """Build format-specific instructions."""
        spec_name = self.format_spec_name(response_format)
        if not self.compiler.has_spec(spec_name):
            logger.warning(
                f"prompt_builder_004: Format spec missing: \033[31m{spec_name}\033[0m"
            )
            return ""
        logger.info(
            f"prompt_builder_005: Compiling format spec: \033[36m{spec_name}\033[0m"
        )
        return self.compiler.compile(
            spec_name, response_format=response_format, intents=intents or []
        )
