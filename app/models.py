from datetime import datetime, timezone
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class ButtonOption(BaseModel):
    text: str
    command: Optional[str] = None
    url: Optional[str] = None

class DropdownOption(BaseModel):
    label: str
    value: str
    command: Optional[str] = None
    url: Optional[str] = None
    
class ChecklistOption(BaseModel):
    label: str
    value: str
    checked: bool = False
    command: Optional[str] = None
    url: Optional[str] = None

class UIElements(BaseModel):
    dropdown: Optional[List[DropdownOption]] = None
    buttons: Optional[List[ButtonOption]] = None
    checklist: Optional[List[ChecklistOption]] = None

class Card(BaseModel):
    title: Optional[str] = None
    subtitle: Optional[str] = None
    image_url: Optional[str] = None
    options: Optional[UIElements] = None

class NavigationCard(BaseModel):
    title: str
    description: Optional[str] = None
    url: Optional[str] = None
    image_url: Optional[str] = None

class ContactCard(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    image_url: Optional[str] = None
    
class ToolCard(BaseModel):
    name: str
    description: Optional[str] = None
    icon_url: Optional[str] = None

class Metadata(BaseModel):
    url: Optional[str] = None
    image_url: Optional[str] = None
    cards: Optional[List[Card]] = None
    options: Optional[UIElements] = None
    tool_cards: Optional[List[ToolCard]] = None
    navigation_card: Optional[NavigationCard] = None

class LllmTrace(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float
    
class ChatMessage(BaseModel):
    message_id: str
    role: Literal["user", "assistant", "system"]
    text_format: Literal["plain", "markdown", "html", "voice"] = "plain"
    text: str
    metadata: Optional[Metadata] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_id: Optional[str] = None
    llm_trace: Optional[LllmTrace] = None

class Conversation(BaseModel):
    conversation_id: str
    messages: List[ChatMessage]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    llm_trace: Optional[LllmTrace] = None
