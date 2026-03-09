"""Intent registry for dynamic UI schema filtering in Stage 3 (ui_answer)."""

from typing import Literal
from archie_shared.ui.models import (
    ArticleCard,
    Card,
    Chart,
    ContactCard,
    DocumentCard,
    EmailForm,
    EventForm,
    InternalNoteForm,
    LocationCard,
    MovieCard,
    MusicCard,
    ProductCard,
    SeriesCard,
    ShoppingListCard,
    WeatherCard,
)


IntentType = Literal[
    "get_weather",
    "search_place",
    "search_movie",
    "search_series",
    "search_music",
    "search_product",
    "create_shopping_list",
    "search_article",
    "search_document",
    "search_contact",
    "create_email",
    "create_event",
    "create_note",
    "analyze_data",
    "get_football",
    "control_light",
    "control_climate",
]

# Base item types always present in ui_answer (type field of AdvancedAnswerItem)
BASE_ITEM_TYPES: list[str] = ["text_answer", "card_grid", "table", "image"]

# Base card types always available inside CardGrid
BASE_CARD_TYPES: list[type] = [Card]

# Intent → additional card types and item types unlocked
INTENT_EXTENSIONS: dict[str, dict[str, list]] = {
    "get_weather": {"cards": [WeatherCard], "item_types": []},
    "search_place": {"cards": [LocationCard], "item_types": []},
    "search_movie": {"cards": [MovieCard], "item_types": []},
    "search_series": {"cards": [SeriesCard], "item_types": []},
    "search_music": {"cards": [MusicCard], "item_types": []},
    "search_product": {"cards": [ProductCard], "item_types": ["chart"]},
    "create_shopping_list": {"cards": [ShoppingListCard], "item_types": []},
    "search_article": {"cards": [ArticleCard], "item_types": []},
    "search_document": {"cards": [DocumentCard], "item_types": []},
    "search_contact": {"cards": [ContactCard], "item_types": []},
    "create_email": {"cards": [], "item_types": ["email_form"]},
    "create_event": {"cards": [], "item_types": ["event_form"]},
    "create_note": {"cards": [], "item_types": ["note_form"]},
    "analyze_data": {"cards": [], "item_types": ["chart"]},
    "get_football": {"cards": [], "item_types": []},
    "control_light": {"cards": [], "item_types": []},
    "control_climate": {"cards": [], "item_types": []},
}

# Content type classes for each item type (for FilteredAdvancedAnswerItem.content union)
ITEM_TYPE_TO_CONTENT_CLASS: dict[str, type] = {
    "chart": Chart,
    "email_form": EmailForm,
    "event_form": EventForm,
    "note_form": InternalNoteForm,
}


def resolve_ui_types(intents: tuple[str, ...]) -> tuple[list[type], list[str]]:
    """
    Resolve active card types and item types from a set of intents.

    Returns:
        (card_types, item_types) — union of base + all intent extensions
    """
    card_types: list[type] = list(BASE_CARD_TYPES)
    item_types: list[str] = list(BASE_ITEM_TYPES)

    for intent in intents:
        ext = INTENT_EXTENSIONS.get(intent, {})
        for card in ext.get("cards", []):
            if card not in card_types:
                card_types.append(card)
        for it in ext.get("item_types", []):
            if it not in item_types:
                item_types.append(it)

    return card_types, item_types
