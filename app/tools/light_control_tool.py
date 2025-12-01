import json
import logging
import os
import redis
from openai import OpenAI
from pydantic import BaseModel, Field


from typing import Literal


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
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LightDeviceId = Literal["light_001", "light_002", "light_003"]


class AssistantButton(BaseModel):
    """Button that triggers assistant request."""

    text: str = Field(description="Button label text")
    style: str = Field(description="Button style: 'primary' or 'secondary'")
    icon: str = Field(description="Lucide icon name (e.g., 'power', 'sun', 'moon')")
    type: str = Field(default="assistant_button")
    assistant_request: str = Field(
        description="Request to send to assistant when clicked"
    )


class QuickActionsResponse(BaseModel):
    """Response with two quick action buttons."""

    button_1: AssistantButton
    button_2: AssistantButton


def generate_quick_actions(
    user_input: str,
    device_name: str,
    current_state: dict,
) -> list[dict]:
    """Generate contextual quick action buttons using LLM."""
    logger.info(
        f"light_control_tool_004: Generating quick actions for \033[36m{device_name}\033[0m"
    )
    system_prompt = """
You are a smart home assistant. Generate 2 contextual follow-up actions based on the user's light control request.
Each button should be a natural next step the user might want to take.

Rules:
- button_1: primary style, most likely next action
- button_2: secondary style, alternative action
- Use simple Russian text for button labels (2-3 words max)
- assistant_request should be a natural command in Russian
- Icons: power, power-off, sun, moon, lightbulb, lamp, palette, thermometer

Examples of good buttons:
- After turning on: "Ярче" / "Теплее свет"
- After turning off: "Включить обратно" / "Включить все"
- After brightness change: "Ещё ярче" / "Приглушить"
"""
    user_prompt = f"""
User request: {user_input}
Device: {device_name}
Current state after action: {json.dumps(current_state, ensure_ascii=False)}

Generate 2 follow-up action buttons.
"""
    try:
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=QuickActionsResponse,
        )
        result = response.choices[0].message.parsed
        logger.info(
            f"light_control_tool_005: Generated buttons: \033[35m{result.button_1.text}\033[0m, \033[35m{result.button_2.text}\033[0m"
        )
        return [result.button_1.model_dump(), result.button_2.model_dump()]
    except Exception as e:
        logger.error(f"light_control_tool_error_003: LLM error: \033[31m{e}\033[0m")
        return [
            {
                "text": "Выключить",
                "style": "primary",
                "icon": "power-off",
                "type": "assistant_button",
                "assistant_request": "Выключи свет",
            },
            {
                "text": "Все лампы",
                "style": "secondary",
                "icon": "lightbulb",
                "type": "assistant_button",
                "assistant_request": "Покажи все лампы",
            },
        ]


async def light_control_tool(
    user_input: str,
    device_id: LightDeviceId,
    is_on: bool | None = None,
    brightness: int | None = None,
    color_temp: int | None = None,
    rgb_color: str | None = None,
    user_name: str = "Niko",
    demo_mode: bool = False,
) -> dict[str, str]:
    """
    Control smart home lights. Use this tool to turn lights on/off, adjust brightness, change color temperature or RGB color.

    Args:
        user_input: Original user request in natural language
        device_id: Smart light device - 'light_001' (Торшер в гостиной), 'light_002' (Потолочный свет на кухне), 'light_003' (Свет в спальне)
        is_on: True to turn on, False to turn off. Set based on user intent ('включи'/'выключи')
        brightness: 0-100 percent. Interpret naturally: 'ярче'=+20, 'приглушить'=-20, 'максимум'=100, 'минимум'=10
        color_temp: Color warmth in Kelvin. 2700=warm/тёплый, 4000=neutral, 6500=cold/холодный. 'теплее'=-500K, 'холоднее'=+500K
        rgb_color: Hex color for RGB mode, e.g. '#FF0000' for red, '#00FF00' for green
        user_name: User name for state lookup

    Returns:
        dict[str, str]: Status and updated device state
    """
    logger.info(
        f"light_control_tool_001: Controlling lamp \033[36m{device_id}\033[0m, "
        f"demo_mode: \033[35m{demo_mode}\033[0m"
    )
    logger.info(f"light_control_tool_001a: User input: \033[35m{user_input}\033[0m")
    logger.info(
        f"light_control_tool_002: is_on: \033[33m{is_on}\033[0m, brightness: \033[33m{brightness}\033[0m, "
        f"color_temp: \033[33m{color_temp}\033[0m, rgb: \033[33m{rgb_color}\033[0m"
    )
    if demo_mode:
        return {
            "status": "demo",
            "message": f"[DEMO] Light {device_id} would be controlled",
            "device_id": device_id,
            "applied_settings": {
                "is_on": is_on,
                "brightness": brightness,
                "color_temp": color_temp,
                "rgb_color": rgb_color,
            },
        }
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
        user_data["smarthome_light"][
            "subtitle"
        ] = f"{on_count} из {total_count} включены"
        updated_device = next(
            (d for d in devices if d.get("device_id") == device_id), {}
        )
        quick_actions = generate_quick_actions(
            user_input=user_input,
            device_name=updated_device.get("name", device_id),
            current_state=updated_device,
        )
        user_data["smarthome_light"]["quick_actions"] = quick_actions
        redis_client.set(redis_key, json.dumps(user_data, ensure_ascii=False))
        logger.info(
            f"light_control_tool_003: Updated Redis state for \033[36m{device_id}\033[0m"
        )
        return {
            "status": "success",
            "message": f"Light {device_id} controlled successfully",
            "device_id": device_id,
            "device_name": updated_device.get("name", device_id),
            "applied_settings": {
                "is_on": is_on,
                "brightness": brightness,
                "color_temp": color_temp,
                "rgb_color": rgb_color,
            },
            "quick_actions": quick_actions,
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
