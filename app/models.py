from datetime import datetime, timezone
from typing import Literal, Optional, List
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
    buttons: Optional[List[ButtonOption]] = None
    # dropdown: Optional[list[DropdownOption]] = Field(
    #     default=None, description="List of dropdown menu options"
    # )
    # checklist: Optional[list[ChecklistOption]] = Field(
    #     default=None, description="List of checkbox tasks"
    # )


class Card(BaseModel):
    title: Optional[str] = None
    text: str
    options: Optional[UIElements] = None

class NavigationCard(BaseModel):
    title: str
    description: str
    url: str
    buttons: List[ButtonOption] = Field(
        default=[
            ButtonOption(text="🗺️ Show on map", command="show_on_map"),
            ButtonOption(text="🚗 Route", command="route")
        ]
    )

class ContactCard(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    buttons: List[ButtonOption] = Field(
        default=[
            ButtonOption(text="📞 Call", command="call"),
            ButtonOption(text="✉️ Email", command="email"),
            ButtonOption(text="💬 Message", command="message")
        ]
    )

class ToolCard(BaseModel):
    name: str
    description: Optional[str] = None

class TableCell(BaseModel):
    content: str

class Table(BaseModel):
    headers: List[str]
    rows: List[List[TableCell]]

class ElementsItem(BaseModel):
    title: str
    value: str

class Elements(BaseModel):
    items: List[ElementsItem]

class Metadata(BaseModel):
    cards: Optional[List[Card]] = None
    options: Optional[UIElements] = None
    tool_cards: Optional[List[ToolCard]] = None
    navigation_card: Optional[NavigationCard] = None
    contact_card: Optional[ContactCard] = None
    table: Optional[Table] = None
    elements: Optional[Elements] = None


class LllmTrace(BaseModel):
    """LLM usage tracking information"""
    model: str = Field(description="Name of the LLM model used")
    input_tokens: int = Field(description="Number of input tokens consumed")
    output_tokens: int = Field(description="Number of output tokens generated")
    total_tokens: int = Field(description="Total number of tokens used")
    total_cost: float = Field(description="Total cost of the request")


class ChatMessage(BaseModel):
    """Chat message model for all communications"""
    message_id: Optional[str] = Field(default=None, description="Unique identifier for the message")
    role: Literal["user", "assistant", "system"] = Field(description="Role of the message sender")
    text_format: Literal["plain", "markdown", "html", "voice"] = Field("plain", description="Format of the message text")
    text: str = Field(description="Content of the message")
    metadata: Optional[Metadata] = Field(default=None, description="Additional metadata for the message")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the message was created",
    )
    conversation_id: Optional[str] = Field(default=None, description="ID of the conversation this message belongs to")
    llm_trace: Optional[LllmTrace] = Field(default=None, description="LLM usage tracking information")


class ChatRequest(BaseModel):
    """Chat request model for incoming messages"""
    role: Literal["user", "assistant", "system"] = Field(description="Role of the message sender")
    text: str = Field(description="Content of the message")
    text_format: Literal["plain", "markdown", "html", "voice"] = Field("plain", description="Format of the message text")
    conversation_id: Optional[str] = Field(default=None, description="ID of the conversation this message belongs to")
