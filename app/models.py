from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class ButtonOption(BaseModel):
    """Button option for user interface"""

    text: str = Field(description="Text displayed on the button")
    command: str = Field(None, description="Command to execute when clicked")
    # url: Optional[str] = Field(None, description="URL to navigate to when clicked")


class DropdownOption(BaseModel):
    """Option for dropdown menu"""

    label: str = Field(description="Display text for the option")
    command: str = Field(None, description="Command to execute when selected")
    # url: Optional[str] = Field(None, description="URL to navigate to when selected")


class ChecklistOption(BaseModel):
    """Checkbox option in a task list"""

    label: str = Field(description="Task text")
    checked: bool = Field(False, description="Checkbox state (checked/unchecked)")
    command: str = Field(None, description="Command to execute when state changes")
    # url: Optional[str] = Field(None, description="URL to navigate to when clicked")


class UIElements(BaseModel):
    """Collection of user interface elements"""

    dropdown: list[DropdownOption] | None = Field(
        None, description="List of dropdown menu options"
    )
    buttons: list[ButtonOption] | None = Field(None, description="List of buttons")
    checklist: list[ChecklistOption] | None = Field(
        None, description="List of checkbox tasks"
    )


class Card(BaseModel):
    """Card with information and interactive elements"""

    title: str | None = Field(None, description="Card title")
    text: str = Field(None, description="Card main text content")
    # image_url: Optional[str] = Field(None, description="URL of the card image")
    options: UIElements | None = Field(
        None, description="Interactive elements of the card"
    )


class NavigationCard(BaseModel):
    """Navigation card for section transitions"""

    title: str = Field(description="Section name")
    description: str = Field(None, description="Section description")
    url: str = Field(None, description="google maps URL to start navigation")
    buttons: list[ButtonOption] | None = Field(
        None, description="Navigation action buttons"
    )


class ContactCard(BaseModel):
    """Contact information card"""

    name: str = Field(description="Contact name")
    email: str | None = Field(None, description="Email address")
    phone: str | None = Field(None, description="Phone number")
    buttons: list[ButtonOption] | None = Field(
        None, description="Contact action buttons"
    )


class ToolCard(BaseModel):
    """Tool or function card"""

    name: str = Field(description="Tool name")
    description: str | None = Field(None, description="Tool functionality description")
    # icon_url: Optional[str] = Field(None, description="URL of the tool icon")


class Metadata(BaseModel):
    """Metadata for enriching response with additional information"""

    # url: Optional[str] = Field(None, description="Main topic link")
    # image_url: Optional[str] = Field(None, description="URL of the main image")
    cards: list[Card] | None = Field(None, description="List of information cards")
    options: UIElements | None = Field(
        None, description="Interactive interface elements"
    )
    tool_cards: list[ToolCard] | None = Field(
        None, description="List of available tools"
    )
    navigation_card: NavigationCard | None = Field(None, description="Navigation card")


class LllmTrace(BaseModel):
    """LLM usage tracking information"""

    model: str = Field(description="Name of the LLM model used")
    input_tokens: int = Field(description="Number of input tokens consumed")
    output_tokens: int = Field(description="Number of output tokens generated")
    total_tokens: int = Field(description="Total number of tokens used")
    total_cost: float = Field(description="Total cost of the request")


class ChatMessage(BaseModel):
    """Chat message model for all communications"""

    message_id: str | None = Field(
        None, description="Unique identifier for the message"
    )
    role: Literal["user", "assistant", "system"] = Field(
        description="Role of the message sender"
    )
    text_format: Literal["plain", "markdown", "html", "voice"] = Field(
        "plain", description="Format of the message text"
    )
    text: str = Field(description="Content of the message")
    metadata: Metadata | None = Field(
        None, description="Additional metadata for the message"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the message was created",
    )
    conversation_id: str | None = Field(
        None, description="ID of the conversation this message belongs to"
    )
    llm_trace: LllmTrace | None = Field(
        None, description="LLM usage tracking information"
    )


class ChatRequest(BaseModel):
    """Chat request model for incoming messages"""

    role: Literal["user", "assistant", "system"] = Field(
        description="Role of the message sender"
    )
    text: str = Field(description="Content of the message")
    text_format: Literal["plain", "markdown", "html", "voice"] = Field(
        "plain", description="Format of the message text"
    )
    conversation_id: str | None = Field(
        None, description="ID of the conversation this message belongs to"
    )
