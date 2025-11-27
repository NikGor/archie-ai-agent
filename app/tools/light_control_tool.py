import json
import logging
import os
import redis


logger = logging.getLogger(__name__)

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_db = int(os.getenv("REDIS_DB", "0"))
redis_client = redis.Redis(
    host=redis_host,
    port=redis_port,
    db=redis_db,
    decode_responses=True,
)


async def light_control_tool(
    device_id: str,
    is_on: bool | None = None,
    brightness: int | None = None,
    color_temp: int | None = None,
    rgb_color: str | None = None,
    user_name: str = "Niko",
) -> dict[str, str]:
    """
    Execute light control command and update Redis state.

    Args:
        device_id: Unique identifier of the lamp (e.g., 'light_001')
        is_on: Turn light on (True) or off (False)
        brightness: Brightness level (0-100)
        color_temp: Color temperature in Kelvin (2700-6500)
        rgb_color: RGB color in hex format (e.g., '#FF5733')
        user_name: User name for Redis key lookup

    Returns:
        dict[str, str]: Dictionary with status and message
    """
    logger.info(f"light_control_tool_001: Controlling lamp \033[36m{device_id}\033[0m")
    logger.info(
        f"light_control_tool_002: is_on: \033[33m{is_on}\033[0m, brightness: \033[33m{brightness}\033[0m, "
        f"color_temp: \033[33m{color_temp}\033[0m, rgb: \033[33m{rgb_color}\033[0m"
    )
    redis_key = f"user_state:name:{user_name}"
    try:
        user_data_json = redis_client.get(redis_key)
        if not user_data_json:
            return {
                "status": "error",
                "message": f"User state not found for {user_name}",
            }
        user_data = json.loads(user_data_json)
        devices = user_data.get("smarthome_light", {}).get("devices", [])
        device_found = False
        for device in devices:
            if device.get("device_id") == device_id:
                device_found = True
                if is_on is not None:
                    device["is_on"] = is_on
                    device["color"] = "yellow" if is_on else "gray"
                if brightness is not None:
                    device["brightness"] = brightness
                if color_temp is not None:
                    device["color_temp"] = color_temp
                    device["color_mode"] = "temperature"
                if rgb_color is not None:
                    device["rgb_color"] = rgb_color
                    device["color_mode"] = "rgb"
                break
        if not device_found:
            return {
                "status": "error",
                "message": f"Device {device_id} not found",
            }
        on_count = sum(1 for d in devices if d.get("is_on", False))
        total_count = len(devices)
        user_data["smarthome_light"]["on_count"] = on_count
        user_data["smarthome_light"]["subtitle"] = f"{on_count} из {total_count} включены"
        redis_client.set(redis_key, json.dumps(user_data, ensure_ascii=False))
        logger.info(
            f"light_control_tool_003: Updated Redis state for \033[36m{device_id}\033[0m"
        )
        return {
            "status": "success",
            "message": f"Light {device_id} controlled successfully",
            "device_id": device_id,
            "applied_settings": {
                "is_on": is_on,
                "brightness": brightness,
                "color_temp": color_temp,
                "rgb_color": rgb_color,
            },
        }
    except redis.RedisError as e:
        logger.error(f"light_control_tool_error_001: Redis error: \033[31m{e}\033[0m")
        return {
            "status": "error",
            "message": f"Redis error: {e}",
        }
    except json.JSONDecodeError as e:
        logger.error(f"light_control_tool_error_002: JSON error: \033[31m{e}\033[0m")
        return {
            "status": "error",
            "message": f"JSON decode error: {e}",
        }
