import logging
import os

from agents import (
    Agent,
    WebSearchTool,
    set_default_openai_key,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

from .models import (
    Metadata,
)
from .state import get_state

logger = logging.getLogger(__name__)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
set_default_openai_key(OPENAI_API_KEY)

DEFAULT_USER_NAME = os.getenv("DEFAULT_USER_NAME", "Николай")
DEFAULT_PERSONA = os.getenv("DEFAULT_PERSONA", "business")

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    autoescape=select_autoescape(enabled_extensions=("jinja2",)),
)

# ==== Models ====


class AgentResponse(BaseModel):
    """Response model for AI agent output"""

    response: str = Field(
        description="""
        Main text response from the AI agent in the specified response format
        Don't duplicate metadata information in the main response text.
        """
    )
    metadata: Metadata = Field(
        description="Additional metadata for enriching the response"
    )


# ==== Agent Builder ====


def build_main_agent() -> Agent:
    state = get_state(user_name=DEFAULT_USER_NAME, persona=DEFAULT_PERSONA)
    persona_key = state["persona"].lower().strip()
    persona_template_path = os.path.join(PROMPTS_DIR, f"persona_{persona_key}.jinja2")
    if not os.path.exists(persona_template_path):
        logger.warning(
            "Persona template not found: %s (path=%s). Proceeding without injected persona block.",
            persona_key,
            persona_template_path,
        )

    system_prompt = _env.get_template("main_agent_prompt.jinja2").render(
        recommended_prompt_prefix=RECOMMENDED_PROMPT_PREFIX,
        persona=persona_key,
    )
    assistant_prompt = _env.get_template("assistant_prompt.jinja2").render(
        state=state,
    )
    instructions = f"{system_prompt}\n\n# Assistant Context\n{assistant_prompt}"

    agent = Agent(
        name=f"MainAgent[{persona_key}]",
        instructions=instructions,
        output_type=AgentResponse,
        tools=[WebSearchTool()],
        model="gpt-4.1",
    )
    logger.info("Initialized MainAgent with persona '%s'", persona_key)
    return agent
