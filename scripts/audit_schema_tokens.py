"""ARCHIE-69: Audit JSON Schema token cost for UI response models.

Usage:
    poetry run python scripts/audit_schema_tokens.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tiktoken

from app.models.output_models import (
    UIResponse,
    DashboardResponse,
    WidgetResponse,
    Level2Response,
    Level3Response,
)

MODELS = {
    "UIResponse": UIResponse,
    "DashboardResponse": DashboardResponse,
    "WidgetResponse": WidgetResponse,
    "Level2Response": Level2Response,
    "Level3Response": Level3Response,
}

ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str | None) -> int:
    if not isinstance(text, str):
        return 0
    return len(ENCODING.encode(text))


def extract_descriptions(schema: dict, path: str = "") -> list[tuple[str, str]]:
    """Recursively extract all Field descriptions with their paths."""
    results = []
    if isinstance(schema, dict):
        if "description" in schema:
            results.append((path, schema["description"]))
        for key, value in schema.items():
            if key == "description":
                continue
            results.extend(
                extract_descriptions(value, f"{path}.{key}" if path else key)
            )
    elif isinstance(schema, list):
        for i, item in enumerate(schema):
            results.extend(extract_descriptions(item, f"{path}[{i}]"))
    return results


def audit_model(name: str, model_class: type) -> dict:
    schema = model_class.model_json_schema()
    schema_json = json.dumps(schema, ensure_ascii=False)
    token_count = count_tokens(schema_json)
    byte_size = len(schema_json.encode("utf-8"))
    descriptions = extract_descriptions(schema)
    desc_tokens = sum(count_tokens(d) for _, d in descriptions)
    return {
        "name": name,
        "bytes": byte_size,
        "tokens": token_count,
        "description_tokens": desc_tokens,
        "description_count": len(descriptions),
        "descriptions": sorted(
            descriptions, key=lambda x: count_tokens(x[1]), reverse=True
        ),
    }


def main():
    print("=" * 70)
    print("ARCHIE-69: JSON Schema Token Audit")
    print("=" * 70)

    results = []
    for name, model_class in MODELS.items():
        result = audit_model(name, model_class)
        results.append(result)

    for r in results:
        print(f"\n{'─' * 70}")
        print(f"Model: {r['name']}")
        print(f"  Schema size:        {r['bytes']:>7,} bytes")
        print(f"  Total tokens:       {r['tokens']:>7,}")
        print(
            f"  Description tokens: {r['description_tokens']:>7,}  ({r['description_tokens'] / r['tokens'] * 100:.1f}% of total)"
        )
        print(f"  Field count:        {r['description_count']:>7}")
        print(f"\n  Top 10 longest Field descriptions:")
        for path, desc in r["descriptions"][:10]:
            t = count_tokens(desc)
            preview = desc[:80].replace("\n", " ") + ("..." if len(desc) > 80 else "")
            print(f"    [{t:>4} tok] {path}")
            print(f"             {preview}")

    print(f"\n{'=' * 70}")
    print("TOTALS:")
    total_tokens = sum(r["tokens"] for r in results)
    total_desc_tokens = sum(r["description_tokens"] for r in results)
    print(f"  All models combined: {total_tokens:,} tokens")
    print(
        f"  Description tokens:  {total_desc_tokens:,} ({total_desc_tokens / total_tokens * 100:.1f}%)"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
