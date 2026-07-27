"""Band-aid repairs for malformed ui_answer content emitted by the LLM."""

import json
import logging
from json_repair import repair_json
from ...models.output_models import UIResponse


logger = logging.getLogger(__name__)


def clear_card_image_prompts(ui_response: UIResponse) -> None:
    """Clear image_prompt on all Card objects inside CardGrid items."""
    for item in ui_response.ui_answer.items:
        if item.type == "card_grid":
            for card in item.content.cards:  # type: ignore[union-attr]
                if hasattr(card, "image_prompt"):
                    card.image_prompt = None
    logger.info("create_output_008: no_image=True — cleared image_prompt on all cards")


def repair_json_string(raw: str) -> str | None:
    """Repair for LLM-emitted JSON: close unterminated strings/brackets, strip
    trailing commas, fix other common malformations. Returns valid JSON string
    or None if nothing usable could be salvaged. Delegates to the `json-repair`
    library (ARCHIE-154) instead of a hand-rolled brace/bracket tracker.

    Note: `json-repair` is deliberately best-effort — unlike the old bracket
    tracker, it does not bail out on mismatched delimiters (e.g. `{"a": ]}`);
    it treats the offending token as garbage and repairs around it. `None` is
    now returned only for empty input or the rare case where the library's
    output still fails to parse.
    """
    candidate = raw.strip()
    if not candidate:
        return None
    repaired = repair_json(candidate)
    if not repaired:
        return None
    try:
        json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return repaired


def sanitize_chart_items(ui_response: UIResponse) -> None:
    """Validate chart_config JSON on chart items; repair when possible, drop otherwise."""
    items = ui_response.ui_answer.items
    for item in list(items):
        if item.type != "chart":
            continue
        chart_config = item.content.chart_config  # type: ignore[union-attr]
        try:
            json.loads(chart_config)
            continue
        except (json.JSONDecodeError, TypeError):
            pass
        repaired = repair_json_string(chart_config)
        if repaired is not None:
            item.content.chart_config = repaired  # type: ignore[union-attr]
            logger.warning(
                "create_output_warning_001: repaired invalid chart_config JSON"
            )
        else:
            items.remove(item)
            logger.warning(
                "create_output_warning_002: dropped chart item with unrepairable chart_config"
            )
