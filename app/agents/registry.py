"""Agent kinds of this process (ARCHIE-296). Importing a kind's quality package
registers its QualityAdapter; upper layers (API, evals, monitoring) call this."""

import functools
import importlib


AGENT_KIND_PACKAGES = ("app.agents.chat.quality", "app.agents.swarm.quality")


@functools.cache
def load_agent_kinds() -> None:
    for package in AGENT_KIND_PACKAGES:
        importlib.import_module(package)
