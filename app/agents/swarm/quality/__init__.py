"""Quality of the swarm: swarm spec rules, the judge of swarm agent messages
and swarm eval replay (ARCHIE-297)."""

from app.platform.quality.adapter import register_quality_adapter
from .adapter import SwarmQualityAdapter


register_quality_adapter(SwarmQualityAdapter())
