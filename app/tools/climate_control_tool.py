"""Climate control tool for smart home integration."""

import logging


logger = logging.getLogger(__name__)


async def climate_control_tool(
    device_id: str,
    action: str,
    temperature: float | None = None,
    mode: str | None = None,
) -> dict[str, str]:
    """
    Control climate devices in the home.

    Args:
        device_id (str): Unique identifier of the climate device (e.g., 'thermostat_living_room', 'ac_bedroom')
        action (str): Action to perform. Allowed values: set_temperature, increase, decrease, turn_on, turn_off
        temperature (float | None): Target temperature in Celsius (only for 'set_temperature' action)
        mode (str | None): Climate mode. Allowed values: heat, cool, auto, fan_only

    Returns:
        dict[str, str]: Dictionary with status and message
    """
    logger.info(
        f"climate_control_tool_001: Controlling device \033[36m{device_id}\033[0m"
    )
    logger.info(
        f"climate_control_tool_002: Action: \033[33m{action}\033[0m, temp: \033[33m{temperature}\033[0m, mode: \033[33m{mode}\033[0m"
    )

    return {
        "status": "success",
        "message": f"Climate device {device_id} controlled successfully",
        "device_id": device_id,
        "applied_settings": {
            "action": action,
            "temperature": temperature,
            "mode": mode,
        },
    }
