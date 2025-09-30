import logging
import os

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .openai_client import AgentResponse, create_agent_response
from .state import get_state

logger = logging.getLogger(__name__)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_USER_NAME = os.getenv("DEFAULT_USER_NAME")
DEFAULT_PERSONA = os.getenv("DEFAULT_PERSONA")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    autoescape=select_autoescape(enabled_extensions=("jinja2",)),
)

# Simplified prompt prefix instead of using agents.extensions
RECOMMENDED_PROMPT_PREFIX = """You are a helpful, harmless, and honest AI assistant."""


async def create_main_agent_response(messages: list[dict[str, str]]) -> AgentResponse:
    """
    Create a response using the main agent configuration.
    
    Args:
        messages: List of conversation messages in OpenAI format
        
    Returns:
        AgentResponse with response text and metadata
    """
    state = get_state(
        user_name=DEFAULT_USER_NAME or "User",
        persona=DEFAULT_PERSONA or "business"
    )
    persona_key = state.get("persona", "business").lower().strip() if state else "business"

    logger.info(f"agent_001: Persona: \033[35m{persona_key}\033[0m")

    # Check if persona template exists
    persona_template_path = os.path.join(PROMPTS_DIR, f"persona_{persona_key}.jinja2")
    if not os.path.exists(persona_template_path):
        logger.warning(f"agent_002: Template missing: \033[31m{persona_key}\033[0m")

    # Render system prompt
    system_prompt = _env.get_template("main_agent_prompt.jinja2").render(
        persona=persona_key,
    )

    # Render assistant context
    assistant_prompt = _env.get_template("assistant_prompt.jinja2").render(
        state=state or {},
    )

    # Combine prompts
    full_system_prompt = f"{system_prompt}\n\n# Assistant Context\n{assistant_prompt}"

    # Combine system prompt with messages
    formatted_messages = []
    formatted_messages.append({"role": "system", "content": full_system_prompt})
    formatted_messages.extend(messages)

    logger.info(f"agent_003: Msgs: \033[33m{len(formatted_messages)}\033[0m, Prompt: \033[33m{len(full_system_prompt)}\033[0m")

    # Create response using OpenAI client
    return await create_agent_response(
        messages=formatted_messages,
        model="gpt-4.1",  # Updated to gpt-4.1 model
    )
