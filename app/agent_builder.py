import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape
from .state import get_state
from agents import (
    Agent,
    WebSearchTool,
    set_default_openai_key,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from .tools import greet_user

logger = logging.getLogger(__name__)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
set_default_openai_key(OPENAI_API_KEY)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    autoescape=select_autoescape(enabled_extensions=("j2",)),
)

def build_main_agent(
    state: Dict[str, Any],
) -> Agent:
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
        tools=[WebSearchTool()],
        model='gpt-4.1-mini',
    )
    logger.info("Initialized MainAgent with persona '%s'", persona_key)
    return agent
