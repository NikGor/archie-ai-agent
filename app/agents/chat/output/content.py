"""Map a parsed chat response to its public Content payload."""

from archie_shared.chat.models import Content
from pydantic import BaseModel
from app.agents.chat.formats import resolve_format


def build_content_from_parsed(
    parsed_content: BaseModel, response_format: str
) -> Content:
    spec = resolve_format(response_format)
    value = (
        getattr(parsed_content, "text", str(parsed_content))
        if spec.content_field == "text"
        else getattr(parsed_content, spec.content_field)
    )
    return Content(content_format=spec.content_format, **{spec.content_field: value})
