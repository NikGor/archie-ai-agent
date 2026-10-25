"""Quality adapter registration of the ephemeral agent kind."""

from app.agents.ephemeral.quality.adapter import EphemeralQualityAdapter
from app.platform.quality.adapter import register_quality_adapter


register_quality_adapter(EphemeralQualityAdapter())
