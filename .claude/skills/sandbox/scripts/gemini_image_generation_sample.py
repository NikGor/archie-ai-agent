"""
Sandbox: Gemini Image Generation Benchmark
Compares Nano Banana 2 (gemini-3.1-flash-image-preview) vs Nano Banana Pro (gemini-3-pro-image-preview)

Metrics: generation time, image resolution, file size
Images saved to /tmp/img_bench/ for visual comparison

Run:
    PYTHONPATH=. poetry run python .claude/skills/sandbox/scripts/gemini_image_generation_sample.py
"""

import os
import time
import mimetypes
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass

from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path("/tmp/img_bench")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_PROMPTS = [
    ("icon", "A minimalist flat icon of a coffee cup, white background"),
    ("scene", "A busy Tokyo street at night with neon signs, rain reflections, crowds"),
    (
        "portrait",
        "A professional headshot of a young woman, studio lighting, neutral background",
    ),
    ("text", "A vintage poster with bold text 'SUMMER SALE 50% OFF', retro style"),
    (
        "abstract",
        "An abstract watercolor painting of ocean waves, blue and teal palette",
    ),
]

MODELS = [
    {
        "name": "Nano Banana Pro",
        "id": "gemini-3-pro-image-preview",
        "config": lambda: types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(
                aspect_ratio="1:1",
                image_size="1K",
            ),
            system_instruction=[
                types.Part.from_text(
                    text="Ensure the image has perfect composition and correct geometry. High fidelity, sharp details, no artifacts."
                ),
            ],
        ),
    },
    {
        "name": "Nano Banana 2",
        "id": "gemini-3.1-flash-image-preview",
        "config": lambda: types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level="MINIMAL",
            ),
            image_config=types.ImageConfig(
                aspect_ratio="1:1",
                image_size="512",
            ),
            response_modalities=["IMAGE"],
            media_resolution="MEDIA_RESOLUTION_LOW",
        ),
    },
]


@dataclass
class BenchResult:
    prompt_type: str
    elapsed: float
    width: int
    height: int
    file_size_kb: float
    success: bool
    error: str = ""


def run_generation(
    client: genai.Client, model_id: str, config, prompt: str, save_path: Path
) -> BenchResult:
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    start = time.time()
    try:
        image_data = None
        for chunk in client.models.generate_content_stream(
            model=model_id,
            contents=contents,
            config=config,
        ):
            if chunk.candidates is None or not chunk.candidates:
                continue
            candidate = chunk.candidates[0]
            if candidate.content is None or candidate.content.parts is None:
                continue
            part = candidate.content.parts[0]
            if part.inline_data and part.inline_data.data:
                image_data = part.inline_data.data
                break

        elapsed = time.time() - start

        if not image_data:
            return BenchResult(
                prompt_type="",
                elapsed=elapsed,
                width=0,
                height=0,
                file_size_kb=0,
                success=False,
                error="No image data received",
            )

        img = Image.open(BytesIO(image_data))
        img.save(str(save_path), format="PNG")
        file_size_kb = save_path.stat().st_size / 1024

        return BenchResult(
            prompt_type="",
            elapsed=elapsed,
            width=img.width,
            height=img.height,
            file_size_kb=file_size_kb,
            success=True,
        )

    except Exception as e:
        elapsed = time.time() - start
        return BenchResult(
            prompt_type="",
            elapsed=elapsed,
            width=0,
            height=0,
            file_size_kb=0,
            success=False,
            error=str(e),
        )


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        return

    client = genai.Client(api_key=api_key)

    all_results: dict[str, list[BenchResult]] = {}

    for model_spec in MODELS:
        model_name = model_spec["name"]
        model_id = model_spec["id"]
        config_fn = model_spec["config"]

        print(f"\n{'='*60}")
        print(f"Testing: {model_name} ({model_id})")
        print(f"{'='*60}")

        results = []
        for prompt_type, prompt in TEST_PROMPTS:
            model_slug = model_id.replace(".", "_").replace("-", "_")
            save_path = OUTPUT_DIR / f"{model_slug}_{prompt_type}.png"

            print(f"  [{prompt_type:10s}] Generating...", end=" ", flush=True)
            result = run_generation(client, model_id, config_fn(), prompt, save_path)
            result.prompt_type = prompt_type

            if result.success:
                print(
                    f"✅ {result.elapsed:.1f}s | {result.width}x{result.height} | {result.file_size_kb:.0f} KB"
                )
            else:
                print(f"❌ {result.elapsed:.1f}s | {result.error[:60]}")

            results.append(result)

        all_results[model_name] = results

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Avg time':>10} {'Success':>10}")
    print(f"{'-'*25} {'-'*10} {'-'*10}")

    avg_times = {}
    for model_name, results in all_results.items():
        successful = [r for r in results if r.success]
        avg_time = (
            sum(r.elapsed for r in successful) / len(successful) if successful else 0
        )
        avg_times[model_name] = avg_time
        print(f"{model_name:<25} {avg_time:>9.1f}s {len(successful):>4}/{len(results)}")

    times = list(avg_times.values())
    names = list(avg_times.keys())
    if len(times) == 2 and times[0] > 0 and times[1] > 0:
        ratio = times[0] / times[1]
        faster = names[1] if ratio > 1 else names[0]
        factor = max(ratio, 1 / ratio)
        print(f"\n⚡ {faster} is {factor:.1f}x faster")

    print(f"\n📁 Images saved to: {OUTPUT_DIR}")
    print(f"   Files: {', '.join(p.name for p in sorted(OUTPUT_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
