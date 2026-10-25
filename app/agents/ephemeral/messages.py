"""Small goal-bound messages compiled from the ephemeral prompt specs."""

import json
from app.agents.ephemeral.models import SpawnRequest
from app.platform.prompts.compiler import PromptCompiler
from app.platform.tools.registry import ToolRegistry


def build_messages(
    request: SpawnRequest, model: str, registry: ToolRegistry, compiler: PromptCompiler
) -> list[dict[str, str]]:
    instructions = (
        compiler.compile("ephemeral_base")
        + "\n"
        + compiler.compile("ephemeral_ha_diagnosis")
    )
    schemas = registry.schemas(model, request.tools)
    return [
        {
            "role": "system",
            "content": instructions
            + "\nAllowed read tools:\n"
            + json.dumps(schemas, ensure_ascii=False),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "goal": request.goal,
                    "context": request.context,
                    "limitation": "HA logs are unavailable; report this gap.",
                },
                ensure_ascii=False,
            ),
        },
    ]
