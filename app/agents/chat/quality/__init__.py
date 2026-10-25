"""Quality of the chat agent: rules it is judged by, the monitoring judge of
stored turns, Langfuse traces and eval replay (ARCHIE-213, ARCHIE-296)."""

from app.platform.quality.adapter import register_quality_adapter
from .adapter import ChatQualityAdapter


register_quality_adapter(ChatQualityAdapter())
