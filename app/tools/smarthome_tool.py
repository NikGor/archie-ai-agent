"""Smart home tools for controlling lights and climate."""

import logging


logger = logging.getLogger(__name__)


async def control_lights(action: str, target: str = "all") -> str:
    """Control smart home lighting devices."""
    logger.info(
        f"smarthome_light_001: Light control - action: \033[36m{action}\033[0m, target: \033[36m{target}\033[0m"
    )
    return f"Light control executed: {action} for {target}"


async def control_climate(action: str, temperature: float | None = None) -> str:
    """Control smart home climate devices."""
    logger.info(
        f"smarthome_climate_001: Climate control - action: \033[36m{action}\033[0m, temp: \033[33m{temperature}\033[0m"
    )
    return f"Climate control executed: {action}" + (f" at {temperature}°C" if temperature else "")
