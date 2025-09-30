from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class ButtonOption(BaseModel):
    text: str
    command: str
    # url: Optional[str] = Field(description="URL to navigate to when clicked")


class DropdownOption(BaseModel):
    """Option for dropdown menu
    Prefer buttons for 2–3 options; use dropdowns only for 4 or more choices.
    """
    label: str = Field(description="Display text for the option")
    command: str = Field(description="Command to execute when selected")
    # url: Optional[str] = Field(description="URL to navigate to when selected")


class ChecklistOption(BaseModel):
    """
    Checkbox option in a task list
    """
    label: str = Field(description="Task text")
    checked: bool = Field(False, description="Checkbox state (checked/unchecked)")
    command: str = Field(description="Command to execute when state changes")
    # url: Optional[str] = Field(description="URL to navigate to when clicked")


class UIElements(BaseModel):
    buttons: list[ButtonOption] | None = None
    # dropdown: Optional[list[DropdownOption]] = Field(
    #     default=None, description="List of dropdown menu options"
    # )
    # checklist: Optional[list[ChecklistOption]] = Field(
    #     default=None, description="List of checkbox tasks"
    # )


class Card(BaseModel):
    title: str | None = None
    text: str
    options: UIElements | None = None

class NavigationCard(BaseModel):
    title: str
    description: str
    url: str
    buttons: list[ButtonOption] = Field(
        default=[
            ButtonOption(text="🗺️ Show on map", command="show_on_map"),
            ButtonOption(text="🚗 Route", command="route")
        ]
    )

class ContactCard(BaseModel):
    name: str
    email: str | None = None
    phone: str | None = None
    buttons: list[ButtonOption] = Field(
        default=[
            ButtonOption(text="📞 Call", command="call"),
            ButtonOption(text="✉️ Email", command="email"),
            ButtonOption(text="💬 Message", command="message")
        ]
    )

class ToolCard(BaseModel):
    name: str
    description: str | None = None

class TableCell(BaseModel):
    content: str

class Table(BaseModel):
    headers: list[str]
    rows: list[list[TableCell]]

class ElementsItem(BaseModel):
    title: str
    value: str

class Elements(BaseModel):
    items: list[ElementsItem]

class Metadata(BaseModel):
    cards: list[Card] | None = None
    options: UIElements | None = None
    tool_cards: list[ToolCard] | None = None
    navigation_card: NavigationCard | None = None
    contact_card: ContactCard | None = None
    table: Table | None = None
    elements: Elements | None = None


class LllmTrace(BaseModel):
    """LLM usage tracking information"""
    model: str = Field(description="Name of the LLM model used")
    input_tokens: int = Field(description="Number of input tokens consumed")
    output_tokens: int = Field(description="Number of output tokens generated")
    total_tokens: int = Field(description="Total number of tokens used")
    total_cost: float = Field(description="Total cost of the request")


class ChatMessage(BaseModel):
    """Chat message model for all communications"""
    message_id: str | None = Field(default=None, description="Unique identifier for the message")
    role: Literal["user", "assistant", "system"] = Field(description="Role of the message sender")
    text_format: Literal["plain", "markdown", "html", "voice"] = Field("plain", description="Format of the message text")
    text: str = Field(description="Content of the message")
    metadata: Metadata | None = Field(default=None, description="Additional metadata for the message")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the message was created",
    )
    conversation_id: str | None = Field(default=None, description="ID of the conversation this message belongs to")
    llm_trace: LllmTrace | None = Field(default=None, description="LLM usage tracking information")


class ChatRequest(BaseModel):
    """Chat request model for incoming messages"""
    role: Literal["user", "assistant", "system"] = Field(description="Role of the message sender")
    text: str = Field(description="Content of the message")
    text_format: Literal["plain", "markdown", "html", "voice"] = Field("plain", description="Format of the message text")
    conversation_id: str | None = Field(default=None, description="ID of the conversation this message belongs to")
