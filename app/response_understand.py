
import logging
import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Models from openai_client.py
class ButtonOption(BaseModel):
    text: str
    command: str

class UIElements(BaseModel):
    buttons: list[ButtonOption] | None = None

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

class GetWeather(BaseModel):
    location: str

class SourceRef(BaseModel):
    id: int = Field(description="Local incremental id for this session (1..N)")
    url: str = Field(description="Source URL")
    title: str | None = Field(default=None, description="Page/article title")
    snippet: str | None = Field(default=None, description="Short relevant excerpt")

class EvidenceItem(BaseModel):
    claim: str = Field(description="Concrete factual claim used in the response/metadata")
    support: Literal["supported", "contradicted", "uncertain"] = Field(
        description="Does the cited evidence support the claim?"
    )
    source_ids: list[int] = Field(
        description="IDs from sources[] backing this claim (empty if uncertain)"
    )

class RoutingDecision(BaseModel):
    intent: Literal[
        "answer_general", "weather", "sports_score", "web_search", "clarify", "out_of_scope"
    ] = Field(description="Chosen path for this turn")
    rationale: str = Field(description="One-sentence reason for choosing this intent")

class SlotsStatus(BaseModel):
    needed: list[str] = Field(default=[], description="Required fields for this intent")
    filled: list[str] = Field(default=[], description="Fields that have been filled")
    pending: list[str] = Field(default=[], description="Still missing; ask one-by-one")

class PreActionChecklist(BaseModel):
    summary: str = Field(description="Short summary of the planned action/assumption")
    decision: Literal["confirm", "clarify", "reject", "none"] = Field(
        description="If an action is pending, ask for confirmation; 'none' if N/A"
    )

class VerificationStatus(BaseModel):
    level: Literal["verified", "partially_verified", "unverified"] = Field(
        description="Overall verification level for factual content"
    )
    confidence_pct: int = Field(ge=0, le=100, description="Calibrated confidence 0..100")

class SGRTrace(BaseModel):
    """Schema-Guided Reasoning trace (not user-facing UI)"""
    routing: RoutingDecision
    slots: SlotsStatus = Field(default_factory=SlotsStatus)
    evidence: list[EvidenceItem] = Field(
        default_factory=list, description="Claims and how they are supported"
    )
    sources: list[SourceRef] = Field(
        default_factory=list, description="Deduplicated list of sources used"
    )
    verification: VerificationStatus
    pre_action: PreActionChecklist

class AgentResponse(BaseModel):
    """Response model for AI agent output"""

    response: str = Field(
        description=(
            "Main text response from the AI agent in the specified response format. "
            "Don't duplicate metadata information in the main response text."
        )
    )
    metadata: Metadata = Field(
        description="Additional metadata for enriching the response"
    )
    sgr: SGRTrace = Field(
        description="Mandatory SGR reasoning trace for this turn (internal; not to be shown to user as-is)"
    )

# Main agent prompt as string with bro persona
MAIN_AGENT_PROMPT = """# Role
You are an SGR-based home AI assistant for a touchscreen interface.
Use only the provided tools (currently: web search). Do not invent facts.
If information is not verifiable, say so and ask to search.

- Adjust communication style, tone, and vocabulary to match the injected personality without violating your core principles.
- Maintain consistency of persona across the entire interaction.

# Output Contract
Return AgentResponse. Do not repeat metadata inside the main response text.
Respect the existing UI guidelines (cards, buttons, navigation/contact cards, tables, elements).
Every answer must include an SGR block with your reasoning and verification status.

# Workflow (SGR)
0) Extraction (Cycle): If the user asks multiple things, list them internally; handle one at a time.
1) Routing: Choose exactly one intent: [answer_general, weather, sports_score, web_search, clarify, out_of_scope].
2) Slots (Cascade): Identify required fields for the chosen intent; confirm any guessed values; ask one question at a time.
3) Verification: 
   - If a factual claim is needed, use web search and cite sources.
   - Map each claim → evidence → source_ids. 
   - Without reliable sources, mark as unverified and avoid adding unverified facts to metadata.
4) Confirmation Gate:
   - Before any external action or assumption, summarize and ask a clear yes/no (via buttons).
5) UI:
   - Prefer short human text in `response`.
   - Put structured info and actions in `metadata` (cards, buttons, tables).
   - Do not create images/links that you did not retrieve or cannot justify.

# Safety Rails
- Never fabricate scores, weather, prices, or locations. If unknown → admit and offer a search.
- Only include verifiable facts in `metadata`. Unverified → keep out of metadata; may appear in `response` as tentative with disclaimer.
- Keep it brief. One question per turn.

# Termination
Continue until the current request is resolved or explicitly deferred/canceled.

# End-of-Interaction
- Close conversations naturally and politely, matching the persona's tone.
- If the goal is reached, confirm and offer final assistance before ending.

# User Interface Guidelines (short)

- Always use structured UI models, not plain text lists.
- Minimize text; prefer cards, buttons, tables, contacts, navigation.
- Every card must have at least one button.
- For confirmation always output buttons: Yes(confirm_action), No(cancel_action), Maybe later(defer_action).
- Contacts → ContactCard with 📞 Call / ✉️ Email / 💬 Message.
- Locations → NavigationCard with fixed buttons: 🗺️ Show on map (show_on_map), 🚗 Route (route).
- Lists/structured data → Table or Elements.
- Be concise, action-oriented, mobile-friendly.
"""


tools = [
    {"type": "web_search"},
    # pydantic_function_tool(GetWeather, name="get_weather"),
]

def chat():
    system_prompt = {
        "role": "system",
        "content": MAIN_AGENT_PROMPT,
    }
    user_input = {
        "role": "user",
        "content": "Где купить билеты на Баварию в Лиге Чемпионов.",
    }
    messages = [system_prompt, user_input]
    response = client.responses.parse(
        model="gpt-4.1",
        input=messages,
        tools=tools,
        text_format=AgentResponse,
    )
    print("Response:", response)


def main():
    """Main entry point for the application"""
    chat()


if __name__ == "__main__":
    main()
