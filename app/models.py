from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ButtonOption(BaseModel):
    """Button option for user interface"""
    text: str = Field(description="Text displayed on the button")
    command: str = Field(description="Command to execute when clicked")
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
    """Collection of user interface elements"""
    buttons: Optional[list[ButtonOption]] = Field(description="List of buttons")
    # dropdown: Optional[list[DropdownOption]] = Field(
    #     description="List of dropdown menu options"
    # )
    # checklist: Optional[list[ChecklistOption]] = Field(
    #     description="List of checkbox tasks"
    # )


class Card(BaseModel):
    """Card with information and interactive elements
    Every card must have at least one button or action and an icon in the content.
    """
    title: Optional[str] = Field(description="Card title")
    text: str = Field(description="Card main text content")
    # image_url: Optional[str] = Field(description="URL of the card image")
    options: Optional[UIElements] = Field(description="Interactive elements of the card")


class NavigationCard(BaseModel):
    """Navigation card for section transitions"""
    title: str = Field(description="Section name")
    description: str = Field(description="Section description")
    url: str = Field(description="google maps URL to start navigation")
    buttons: Optional[list[ButtonOption]] = Field(description="Navigation action buttons")


class ContactCard(BaseModel):
    """Contact information card"""
    name: str = Field(description="Contact name")
    email: Optional[str] = Field(description="Email address")
    phone: Optional[str] = Field(description="Phone number")
    buttons: Optional[list[ButtonOption]] = Field(description="Contact action buttons")

class ToolCard(BaseModel):
    """Tool or function card"""
    name: str = Field(description="Tool name")
    description: Optional[str] = Field(description="Tool functionality description")
    # icon_url: Optional[str] = Field(description="URL of the tool icon")


class Metadata(BaseModel):
    """Metadata for enriching response with additional information"""
    # url: Optional[str] = Field(description="Main topic link")
    # image_url: Optional[str] = Field(description="URL of the main image")
    cards: Optional[list[Card]] = Field(description="List of information cards")
    options: Optional[UIElements] = Field(description="Interactive interface elements")
    tool_cards: Optional[list[ToolCard]] = Field(description="List of available tools")
    navigation_card: Optional[NavigationCard] = Field(description="Navigation card")


class LllmTrace(BaseModel):
    """LLM usage tracking information"""
    model: str = Field(description="Name of the LLM model used")
    input_tokens: int = Field(description="Number of input tokens consumed")
    output_tokens: int = Field(description="Number of output tokens generated")
    total_tokens: int = Field(description="Total number of tokens used")
    total_cost: float = Field(description="Total cost of the request")


class ChatMessage(BaseModel):
    """Chat message model for all communications"""
    message_id: Optional[str] = Field(description="Unique identifier for the message")
    role: Literal["user", "assistant", "system"] = Field(description="Role of the message sender")
    text_format: Literal["plain", "markdown", "html", "voice"] = Field("plain", description="Format of the message text")
    text: str = Field(description="Content of the message")
    metadata: Optional[Metadata] = Field(description="Additional metadata for the message")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the message was created",
    )
    conversation_id: Optional[str] = Field(description="ID of the conversation this message belongs to")
    llm_trace: Optional[LllmTrace] = Field(description="LLM usage tracking information")


class ChatRequest(BaseModel):
    """Chat request model for incoming messages"""
    role: Literal["user", "assistant", "system"] = Field(description="Role of the message sender")
    text: str = Field(description="Content of the message")
    text_format: Literal["plain", "markdown", "html", "voice"] = Field("plain", description="Format of the message text")
    conversation_id: Optional[str] = Field(description="ID of the conversation this message belongs to")
