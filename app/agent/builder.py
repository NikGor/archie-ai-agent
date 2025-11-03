import logging
import os

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..models.response_models import AgentResponse
from .openai_client import create_agent_response
from .state import get_state

logger = logging.getLogger(__name__)
load_dotenv()
DEFAULT_USER_NAME = os.getenv("DEFAULT_USER_NAME")
DEFAULT_PERSONA = os.getenv("DEFAULT_PERSONA")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    autoescape=select_autoescape(enabled_extensions=("jinja2",)),
)


async def create_main_agent_response(
    messages: list[dict[str, str]], 
    previous_response_id: str | None = None,
    model: str = "gpt-4.1",
    response_format: str = "plain",
) -> AgentResponse:
    """Create a response using the main agent configuration."""
    state = get_state(
        user_name=DEFAULT_USER_NAME or "User",
        persona=DEFAULT_PERSONA or "business",
    )
    persona_key = (
        state.get("persona", "business").lower().strip() if state else "business"
    )
    logger.info(f"agent_001: Loaded persona: \033[35m{persona_key}\033[0m")
    logger.info(f"agent_002: Response format: \033[36m{response_format}\033[0m")
    
    # Check persona template exists
    persona_template_path = os.path.join(PROMPTS_DIR, f"persona_{persona_key}.jinja2")
    
    # Load format-specific prompt
    if response_format in ["plain", "ui_answer"]:
        format_template_name = f"format_{response_format}.jinja2"
    else:
        format_template_name = "format_formatted_text.jinja2"
    
    format_template_path = os.path.join(PROMPTS_DIR, format_template_name)
    if not os.path.exists(format_template_path):
        logger.warning(f"agent_004: Format template missing: \033[31m{format_template_name}\033[0m")
        format_prompt = ""
    else:
        format_prompt = _env.get_template(format_template_name).render(
            response_format=response_format
        )
        logger.info(f"agent_005: Loaded format template: \033[36m{format_template_name}\033[0m")
    
    system_prompt = _env.get_template("main_agent_prompt.jinja2").render(
        persona=persona_key,
        response_format=response_format,
        format_instructions=format_prompt,
    )
    assistant_prompt = _env.get_template("assistant_prompt.jinja2").render(
        state=state or {},
        response_format=response_format,
    )
    full_system_prompt = f"{system_prompt}\n\n# Assistant Context\n{assistant_prompt}"
    formatted_messages = []
    formatted_messages.append({"role": "system", "content": full_system_prompt})
    formatted_messages.extend(messages)
    logger.info(
        f"agent_006: loaded msgs: \033[33m{len(formatted_messages)}\033[0m, "
        f"Prompt: \033[33m{len(full_system_prompt)}\033[0m chars"
    )
    return await create_agent_response(
        messages=formatted_messages,
        model=model,
        previous_response_id=previous_response_id,
    )
