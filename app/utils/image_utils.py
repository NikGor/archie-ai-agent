"""Build provider-specific user message content with attached images."""

from typing import Any
from archie_shared.chat.models import ImageAttachment


def build_user_content(
    text: str, images: list[ImageAttachment] | None, provider: str
) -> str | list[dict[str, Any]]:
    """Return plain text, or a multimodal content-part list when images are attached.

    OpenAI Responses API uses `input_text`/`input_image`; chat-completions
    providers (OpenRouter) use `text`/`image_url`.
    """
    if not images:
        return text
    if provider == "openai":
        parts: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        parts += [
            {
                "type": "input_image",
                "image_url": f"data:{img.media_type};base64,{img.data}",
            }
            for img in images
        ]
        return parts
    parts = [{"type": "text", "text": text}]
    parts += [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{img.media_type};base64,{img.data}"},
        }
        for img in images
    ]
    return parts
