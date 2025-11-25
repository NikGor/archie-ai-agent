import base64
import logging
import os
import mimetypes
from typing import Any, Literal
from io import BytesIO
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

logger = logging.getLogger(__name__)
load_dotenv()


async def profi_image_generation_tool(
    prompt: str,
    aspect_ratio: Literal["1:1", "3:4", "4:3", "9:16", "16:9"] = "1:1",
    image_size: Literal["1K", "2K"] = "2K",
) -> dict[str, Any]:
    """
    Professional high-quality image generation using Gemini 3 Pro Image (Nano Banana Pro).
    Produces highest quality, detailed images with advanced rendering and text support.
    Best for complex scenes, professional artwork, detailed illustrations, text in images.

    Args:
        prompt: Detailed text description of the image to generate
        aspect_ratio: Image aspect ratio (1:1, 3:4, 4:3, 9:16, 16:9)
        image_size: Image resolution - 1K or 2K

    Returns:
        Dict with generated images as base64 strings and metadata
    """
    logger.info(
        f"profi_image_gen_001: Professional generation for: \033[36m{prompt[:50]}...\033[0m"
    )
    logger.info(
        f"profi_image_gen_002: Parameters - ratio: \033[33m{aspect_ratio}\033[0m, "
        f"size: \033[33m{image_size}\033[0m"
    )

    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error(
                "profi_image_gen_error_001: \033[31mGEMINI_API_KEY not found\033[0m"
            )
            return {
                "success": False,
                "message": "GEMINI_API_KEY not configured",
                "prompt": prompt,
            }

        client = genai.Client(api_key=api_key)

        model = "gemini-3-pro-image-preview"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            ),
            system_instruction=[
                types.Part.from_text(
                    text="Ensure the image has perfect composition and correct geometry. High fidelity, sharp details, no artifacts. If text is present, it must be legible, correctly spelled, and naturally integrated into the surface. High aesthetic quality."
                ),
            ],
        )

        logger.info("profi_image_gen_003: Calling \033[36mNano Banana Pro\033[0m")

        images_data = []
        image_index = 0

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue

            if (
                chunk.candidates[0].content.parts[0].inline_data
                and chunk.candidates[0].content.parts[0].inline_data.data
            ):
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                image_bytes = inline_data.data

                image_obj = Image.open(BytesIO(image_bytes))

                img_byte_arr = BytesIO()
                image_obj.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)
                image_bytes_final = img_byte_arr.read()

                image_base64 = base64.b64encode(image_bytes_final).decode("utf-8")

                images_data.append(
                    {
                        "index": image_index + 1,
                        "base64": image_base64,
                        "format": "PNG",
                        "size": {
                            "width": image_obj.width,
                            "height": image_obj.height,
                        },
                    }
                )

                logger.info(
                    f"profi_image_gen_004: Image {image_index + 1} - \033[33m{image_obj.width}x{image_obj.height}\033[0m"
                )
                image_index += 1

        if not images_data:
            logger.warning("profi_image_gen_warning_002: No images generated")
            return {
                "success": False,
                "message": "No images generated",
                "prompt": prompt,
            }

        logger.info(
            f"profi_image_gen_005: Successfully generated \033[33m{len(images_data)}\033[0m professional images"
        )

        return {
            "success": True,
            "prompt": prompt,
            "images": images_data,
            "count": len(images_data),
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "model": "gemini-3-pro-image-preview",
            "message": f"Successfully generated {len(images_data)} high-quality image(s) using Nano Banana Pro",
        }

    except Exception as e:
        logger.error(f"profi_image_gen_error_002: \033[31m{e!s}\033[0m")
        return {
            "success": False,
            "message": f"Professional image generation failed: {e!s}",
            "prompt": prompt,
        }
