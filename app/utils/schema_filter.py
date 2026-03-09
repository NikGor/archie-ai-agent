"""Dynamic filtered Pydantic model builder for ui_answer Stage 3 schema optimization."""

import logging
from functools import lru_cache
from typing import Literal, Union
from archie_shared.ui.models import Image, QuickActionButtons, Table, TextAnswer
from pydantic import BaseModel, Field, create_model
from ..models.output_models import SGROutput
from .intent_config import ITEM_TYPE_TO_CONTENT_CLASS, resolve_ui_types


logger = logging.getLogger(__name__)


@lru_cache(maxsize=64)
def build_filtered_ui_response(intents: tuple[str, ...], no_image: bool = False) -> type[BaseModel]:
    """
    Build a filtered UIResponse Pydantic model for the given set of intents.

    Only includes card types and item types relevant to the active intents.
    Results are cached — same intents tuple always returns the same class.

    Args:
        intents: Sorted tuple of active intent strings (use tuple(sorted(intents_list)))

    Returns:
        A dynamically created Pydantic model class equivalent to UIResponse
        but with a restricted schema.
    """
    card_types, item_types = resolve_ui_types(intents)
    if no_image:
        item_types = [it for it in item_types if it != "image"]
        logger.debug("schema_filter_001b: no_image=True — excluded 'image' from item_types")

    logger.debug(
        f"schema_filter_001: Building filtered model for intents={intents}, "
        f"card_types={[c.__name__ for c in card_types]}, item_types={item_types}"
    )

    # --- FilteredCardGrid ---
    CardUnion = card_types[0] if len(card_types) == 1 else Union[tuple(card_types)]  # noqa: UP007

    FilteredCardGrid = create_model(
        "FilteredCardGrid",
        grid_dimensions=(
            Literal["1_column", "2_columns"],
            Field(description="Grid layout choice"),
        ),
        cards=(
            list[CardUnion],  # type: ignore[valid-type]
            Field(description="List of cards to display in the grid."),
        ),
    )

    # --- Content union for FilteredAdvancedAnswerItem ---
    # Base content classes + card_grid (uses FilteredCardGrid) + intent-specific
    content_classes: list[type] = [TextAnswer, FilteredCardGrid, Table]
    if not no_image:
        content_classes.append(Image)
    for it in item_types:
        if it in ITEM_TYPE_TO_CONTENT_CLASS and ITEM_TYPE_TO_CONTENT_CLASS[it] not in content_classes:
            content_classes.append(ITEM_TYPE_TO_CONTENT_CLASS[it])

    if len(content_classes) == 1:
        ContentUnion = content_classes[0]
    else:
        ContentUnion = Union[tuple(content_classes)]  # noqa: UP007

    # --- Item type Literal ---
    ItemTypeLiteral = Literal[tuple(item_types)]  # type: ignore[valid-type]

    # --- FilteredAdvancedAnswerItem ---
    FilteredAdvancedAnswerItem = create_model(
        "FilteredAdvancedAnswerItem",
        order=(int, Field(description="Display order (1-based). Use increments of 10.")),
        type=(ItemTypeLiteral, Field(description="UI component type.")),
        content=(ContentUnion, Field(description="Component content payload.")),
        layout_hint=(
            Literal["full_width", "half_width", "inline", "emphasis"] | None,
            Field(default="full_width", description="Visual layout hint."),
        ),
        spacing=(
            Literal["tight", "normal", "loose"] | None,
            Field(default="normal", description="Vertical spacing."),
        ),
    )

    # --- FilteredUIAnswer ---
    FilteredUIAnswer = create_model(
        "FilteredUIAnswer",
        intro_text=(TextAnswer | None, Field(default=None, description="Introductory paragraph")),
        items=(list[FilteredAdvancedAnswerItem], Field(description="List of items in the UI answer")),  # type: ignore[valid-type]
        quick_action_buttons=(
            QuickActionButtons | None,
            Field(default=None, description="Quick action buttons for the UI"),
        ),
    )

    # --- FilteredUIResponse ---
    FilteredUIResponse = create_model(
        "FilteredUIResponse",
        ui_answer=(FilteredUIAnswer, Field(description="Full UI elements content")),
        sgr=(SGROutput, Field(description="Output reasoning trace")),
    )

    logger.debug(f"schema_filter_002: Built FilteredUIResponse for intents={intents}")
    return FilteredUIResponse
