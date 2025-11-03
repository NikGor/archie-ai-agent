import json
import logging
import os
from typing import Literal, List, Optional, Union, Dict
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Import AgentResponse from models
from .models.response_models import AgentResponse

load_dotenv()
logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Button(BaseModel):
    """Interactive action button with clear visual hierarchy and purpose"""
    text: str = Field(
        description="Concise action-oriented button label (2-4 words max). Use verbs."
    )
    style: Optional[Literal["primary", "secondary", "success", "warning", "danger"]] = Field(
        default="secondary",
        description="Visual prominence: 'primary' for main action, 'secondary' for alternatives, etc. Use all styles consistently within the context."
    )
    icon: Optional[str] = Field(
        default=None,
        description="Icon identifier for visual representation. Use Lucide icon names or emoji as fallback."
    )

class FrontendButton(Button):
    """Frontend-specific button with additional properties for UI/UX"""
    type: Literal["frontend_button"] = Field("frontend_button", description="Button type for frontend discrimination")
    command: Literal[
        "navigate_to", 
        "open_map", 
        "call", 
        "email", 
        "message", 
        "show_details", 
        "export_to_notes", 
        "export_to_calendar", 
        "open_on_youtube_video", 
        "open_on_youtube_music", 
        "check_amazon"
        ] = Field(
        description="Predefined command for frontend routing: for general actions"
    )

class AssistantButton(Button):
    """Assistant-specific button with tailored behavior and context"""
    type: Literal["assistant_button"] = Field("assistant_button", description="Button type for frontend discrimination")
    assistant_request: str = Field(
        description="Natural language request that will be processed by assistant when clicked. Be specific and contextual."
    )

class Card(BaseModel):
    """Versatile content card with clear information hierarchy and actionability"""
    type: Literal["card"] = Field("card", description="Type of the card for frontend rendering")
    title: Optional[str] = Field(
        default=None, 
        description="Clear, descriptive headline (3-6 words). Use title case. Focus on user benefit or key information."
    )
    subtitle: Optional[str] = Field(
        default=None, 
        description="Supporting context or category label. Keep under 10 words. Use for dates, locations, or types."
    )
    text: Optional[str] = Field(
        default=None, 
        description="Concise body content (1-3 sentences max). Focus on essential information user needs to act."
    )
    buttons: Optional[List[Union[FrontendButton, AssistantButton]]] = Field(
        default=None, 
        description="Action buttons ordered by importance. Always include at least 1 primary action. Max 3 buttons per card."
    )

class LocationCard(BaseModel):
    """Location-focused card for places with optional navigation capabilities"""
    type: Literal["location_card"] = Field("location_card", description="Type of the card for frontend rendering")
    title: str = Field(
        description="Clear location name or destination"
    )
    description: Optional[str] = Field(
        default=None, 
        description="Brief context: distance, travel time, or key details"
    )
    address: Optional[str] = Field(
        default=None,
        description="Full address for precise navigation and user clarity"
    )
    open_map_url: Optional[str] = Field(
        default=None, 
        description="Direct link to map view - use platform-specific URLs (Google Maps)"
    )
    buttons: List[Union[FrontendButton, AssistantButton]] = Field(
        default=None,
        description="""
        Action buttons specific to the location
        Use the frontend button "open_map"
        Add 1-2 assistant buttons for another relevant quick action
        Maximum 3 buttons per location
        """
    )

class ProductCard(BaseModel):
    """Product-focused card for items with specifications and purchase options"""
    type: Literal["product_card"] = Field("product_card", description="Type of the card for frontend rendering")
    
    # Main info
    title: str = Field(
        description="Product name and model"
    )
    brand: Optional[str] = Field(
        default=None,
        description="Product brand or manufacturer"
    )
    
    # Commercial info
    price: Optional[str] = Field(
        default=None,
        description="Current price with currency"
    )
    rating: Optional[str] = Field(
        default=None,
        description="User rating and review count"
    )
    
    # Technical details
    specifications: Optional[List[str]] = Field(
        default=None,
        description="3-5 key specifications or features"
    )
    
    # Actions
    buttons: Optional[List[Union[FrontendButton, AssistantButton]]] = Field(
        default=None,
        description="""
        Action buttons specific to the product
        Always include: frontend button "check_amazon" for product details
        Add assistant button for finding other stores/retailers
        Add assistant button for price comparison or similar products
        Maximum 3 buttons per product
        """
    )

class MovieCard(BaseModel):
    """Movie-focused informational card with key details"""
    type: Literal["movie_card"] = Field("movie_card", description="Type of the card for frontend rendering")
    
    # Main info
    title: str = Field(
        description="Movie title"
    )
    year: int = Field(
        description="Release year"
    )
    
    # Production details
    director: Optional[str] = Field(
        default=None,
        description="Director name"
    )
    cast: Optional[List[str]] = Field(
        default=None,
        description="Top 3 main actors/actresses"
    )
    studio: Optional[str] = Field(
        default=None,
        description="Production studio or company"
    )
    
    # Classification & metrics
    genre: Optional[str] = Field(
        default=None,
        description="Primary genre or genres"
    )
    rating: Optional[str] = Field(
        default=None,
        description="Rating with source"
    )
    duration: Optional[str] = Field(
        default=None,
        description="Runtime"
    )
    
    # Content
    description: Optional[str] = Field(
        default=None,
        description="Brief plot summary (2-3 sentences max)"
    )
    
    # Actions
    buttons: Optional[List[Union[FrontendButton, AssistantButton]]] = Field(
        default=None,
        description="""
        Action buttons specific to the movie
        Include frontend button "open_on_youtube_video" for trailers
        Add assistant buttons for recommendations or streaming info
        Maximum 3 buttons per movie
        """
    )

class SeriesCard(BaseModel):
    """TV series-focused informational card with show details"""
    type: Literal["series_card"] = Field("series_card", description="Type of the card for frontend rendering")
    
    # Main info
    title: str = Field(
        description="Series title"
    )
    years: str = Field(
        description="Years aired"
    )
    status: Optional[str] = Field(
        default=None,
        description="Current status"
    )
    
    # Production details
    network: Optional[str] = Field(
        default=None,
        description="Network, streaming platform, or original broadcaster"
    )
    creators: Optional[List[str]] = Field(
        default=None,
        description="Show creators or showrunners"
    )
    
    # Series metrics
    seasons: Optional[int] = Field(
        default=None,
        description="Number of seasons"
    )
    episodes: Optional[int] = Field(
        default=None,
        description="Total number of episodes"
    )
    genre: Optional[str] = Field(
        default=None,
        description="Primary genre"
    )
    rating: Optional[str] = Field(
        default=None,
        description="Rating with source"
    )
    
    # Content
    description: Optional[str] = Field(
        default=None,
        description="Brief show summary (2-3 sentences max)"
    )
    
    # Actions
    buttons: Optional[List[Union[FrontendButton, AssistantButton]]] = Field(
        default=None,
        description="""
        Action buttons specific to the series
        Include frontend button "open_on_youtube_video" for trailers/clips
        Add assistant buttons for episode guides or streaming info
        Maximum 3 buttons per series
        """
    )

class MusicCard(BaseModel):
    """Music-focused card with track, album, and artist information"""
    type: Literal["music_card"] = Field("music_card", description="Type of the card for frontend rendering")
    
    # Main info
    track_title: str = Field(
        description="Song title"
    )
    artist: str = Field(
        description="Artist or band name"
    )
    
    # Album info
    album_title: Optional[str] = Field(
        default=None,
        description="Album name"
    )
    album_year: Optional[int] = Field(
        default=None,
        description="Album release year"
    )
    
    # Track details
    duration: Optional[str] = Field(
        default=None,
        description="Track length"
    )
    track_year: Optional[int] = Field(
        default=None,
        description="Track release year"
    )
    genre: Optional[str] = Field(
        default=None,
        description="Music genre"
    )
    country: Optional[str] = Field(
        default=None,
        description="Artist's country of origin"
    )
    
    # Actions
    buttons: Optional[List[Union[FrontendButton, AssistantButton]]] = Field(
        default=None,
        description="""
        Action buttons specific to the music
        Include frontend button "open_on_youtube_music" for listening
        Add assistant buttons for similar music or artist info
        Maximum 3 buttons per music card
        """
    )

class ArticleCard(BaseModel):
    """Article-focused card for news and blog content"""
    type: Literal["article_card"] = Field("article_card", description="Type of the card for frontend rendering")
    title: str = Field(
        description="Article headline"
    )
    source: str = Field(
        description="Publication or website name"
    )
    published_date: Optional[str] = Field(
        default=None,
        description="Publication date"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Brief article summary (2-3 sentences highlighting key points)"
    )
    buttons: Optional[List[Union[FrontendButton, AssistantButton]]] = Field(
        default=None,
        description="""
        Action buttons specific to the article
        Always include: frontend button "export_to_notes" for saving
        Add assistant button for summarizing or sharing
        Maximum 3 buttons per article
        """
    )

class ShoppingListCard(BaseModel):
    """Shopping list card with items organized by store departments"""
    type: Literal["shopping_list_card"] = Field("shopping_list_card", description="Type of the card for frontend rendering")
    title: str = Field(
        description="List title"
    )
    items_by_department: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="""
        Dictionary organized by German store departments.
        Don't use recipe units like: '90g pancetta', '3 large eggs', '50g pecorino cheese'
        Round up quantities to nearest store unit: kg, g, L, ml, pieces, packs, bottles, cans
        """
    )
    total_cost: int = Field(
        default=0,
        description="Total estimated cost of all items in the list"
    )
    buttons: Optional[List[Union[FrontendButton, AssistantButton]]] = Field(
        default=None,
        description="""
        Action buttons specific to the shopping list
        Always include: frontend button "export_to_notes" for saving
        Add assistant button for modifying or optimizing the list
        Maximum 2 buttons per shopping list
        """
    )

class WeatherCard(BaseModel):
    """Weather-focused card with current conditions and forecast"""
    type: Literal["weather_card"] = Field("weather_card", description="Type of the card for frontend rendering")
    # Location
    location: str = Field(
        description="Location name"
    )
    # Current conditions
    current_temp: str = Field(
        description="Current temperature with unit"
    )
    feels_like: Optional[str] = Field(
        default=None,
        description="Feels like temperature"
    )
    condition: str = Field(
        description="Weather condition"
    )
    condition_icon: Optional[str] = Field(
        default=None,
        description="Lucide icon representing current weather condition"
    )
    # Additional metrics
    humidity: Optional[str] = Field(
        default=None,
        description="Humidity percentage"
    )
    wind: Optional[str] = Field(
        default=None,
        description="Wind speed and direction"
    )
    wind_direction_icon: Optional[Literal["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]] = Field(
        default=None,
        description="Lucide icon representing wind direction"
    )
    # Forecast & advice
    daily_forecast: Optional[str] = Field(
        default=None,
        description="Today's high/low"
    )
    clothing_advice: Optional[str] = Field(
        default=None,
        description="Clothing recommendation"
    )
    # Actions
    buttons: Optional[List[Union[FrontendButton, AssistantButton]]] = Field(
        default=None,
        description="Action buttons: '7-Day Forecast', 'Weather Alerts', 'Hourly Forecast'. Maximum 3 buttons."
    )

class ContactCard(BaseModel):
    """People-focused card optimized for immediate communication actions"""
    type: Literal["contact_card"] = Field("contact_card", description="Type of the card for frontend rendering")
    name: str = Field(
        description="Person's full name or professional title + name"
    )
    role: Optional[str] = Field(
        default=None,
        description="Professional context or relationship"
    )
    company: Optional[str] = Field(
        default=None,
        description="Organization or company name for professional contacts"
    )
    email: Optional[str] = Field(
        default=None, 
        description="Primary email address - ensure it's actionable (clickable mailto: link)"
    )
    phone: Optional[str] = Field(
        default=None, 
        description="Phone number in international format for universal compatibility (+1-555-123-4567)"
    )
    availability: Optional[str] = Field(
        default=None,
        description="When they're available or best time to contact (e.g., 'Available 9-5 EST', 'Weekends preferred')"
    )
    preferred_contact: Optional[Literal["phone", "email", "message"]] = Field(
        default=None,
        description="Suggested primary contact method to prioritize in UI"
    )
    buttons: List[Union[FrontendButton, AssistantButton]] = Field(
        default=None,
        description="""
        Action buttons specific to contacting the person
        If the contact has a phone number — add a frontend button 'call' (and an assistant button for clarifying the best time to call)
        If the contact has an email — add a frontend button 'email' (and an assistant button for sending a template email)
        If the contact has a messenger — add a frontend button 'message'
        For each contact, add 1 assistant button for a quick action (e.g., 'Book an appointment', 'Request details')
        Maximum 3 buttons per contact
    """
    )
    
class CardGrid(BaseModel):
    """Grid layout for multiple cards to enhance visual scanning and comparison"""
    grid_dimensions: Literal["1_column", "2_columns"] = Field(
        description="Grid layout choice: '1 column' for a single card, '2_columns' for 2 or more cards"
    )
    cards: List[Union[Card, LocationCard, ContactCard, ProductCard, MovieCard, SeriesCard, MusicCard, ArticleCard, ShoppingListCard, WeatherCard]] = Field(
        description="List of cards to display in the grid. Keep individual card content concise for quick scanning."
    )

class Table(BaseModel):
    """Structured data table optimized for comparison and scanning"""
    title: Optional[str] = Field(
        default=None,
        description="Table caption explaining what user is comparing (e.g., 'Price Comparison', 'Weekly Schedule')"
    )
    headers: List[str] = Field(
        description="Clear, concise column headers (1-2 words each). Use nouns or questions users are comparing."
    )
    rows: List[List[str]] = Field(
        description="Data rows with consistent formatting. Keep cell content scannable - use abbreviations, symbols, colors for quick reading."
    )
    sortable: Optional[bool] = Field(
        default=False,
        description="Whether user can sort columns - useful for price, date, or ranking comparisons"
    )
    highlight_column: Optional[int] = Field(
        default=None,
        description="Zero-based index of column to emphasize (usually the comparison point or recommended option)"
    )

class TextAnswer(BaseModel):
    """Rich text content with appropriate formatting for optimal readability"""
    type: Literal["plain", "markdown", "html", "voice"] = Field(
        description="Content format: 'markdown' for structured text, 'plain' for simple responses, 'html' for rich formatting"
    )
    text: str = Field(
        description="Well-structured content with clear paragraphs. With markdown and html use bold, cursive, headers, bullet points for lists, and conversational tone."
    )

class Chart(BaseModel):
    """Chart.js compatible chart configuration for data visualization"""
    chart_type: Literal["bar", "line", "pie", "doughnut", "area", "scatter"] = Field(
        description="Chart.js chart type: 'pie'/'doughnut' for parts-of-whole, 'bar' for comparisons, 'line'/'area' for trends"
    )
    chart_config: str = Field(
        description=(
            "Complete Chart.js configuration as JSON string ready for frontend parsing. "
            "Must include 'type', 'data' with labels and datasets, and mobile-optimized 'options'. "
            "Use proper Chart.js format with backgroundColor, responsive: true, maintainAspectRatio: false"
        )
    )
    title: Optional[str] = Field(
        default=None,
        description="Human-readable chart title for accessibility and context (e.g., 'Sales Distribution', 'Temperature Trend')"
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief explanation of what the chart shows and key insights (e.g., 'Revenue peaked in Q4', 'Most users prefer mobile')"
    )
    height: Optional[int] = Field(
        default=300,
        description="Chart height in pixels, optimized for mobile viewing (200-400px recommended)"
    )

class AdvancedAnswerItem(BaseModel):
    """Strategic UI component with clear hierarchy and user flow optimization"""
    order: int = Field(
        description="Visual sequence (1-based). Lower numbers appear first. Use 10, 20, 30 for easy reordering."
    )
    type: Literal["text_answer", "card_grid", "table", "chart"] = Field(
        description="Component type - choose based on user intent: 'card' for actions, 'table' for comparison, 'text_answer' for explanation, 'chart' for data visualization"
    )
    content: Union[TextAnswer, CardGrid, Table, Chart] = Field(
        description="Component payload - ensure content matches type and supports user's immediate next action"
    )
    layout_hint: Optional[Literal["full_width", "half_width", "inline", "emphasis"]] = Field(
        default="full_width",
        description="Visual layout guidance: 'emphasis' for critical items, 'inline' for quick actions, 'half_width' for comparisons"
    )
    spacing: Optional[Literal["tight", "normal", "loose"]] = Field(
        default="normal",
        description="Vertical spacing around component: 'tight' for related items, 'loose' for section breaks"
    )

class QuickActionButtons(BaseModel):
    """Persistent action bar for immediate user needs and conversation flow"""
    buttons: List[AssistantButton] = Field(
        description="2-3 action buttons representing most likely next steps. Order by user priority, include 1 primary action."
    )

class UIAnswer(BaseModel):
    """Generative UI answer content"""
    intro_text: TextAnswer = Field(default=None, description="Introductory paragraph")
    items: List[AdvancedAnswerItem] = Field(description="List of items in the generative UI answer")
    quick_action_buttons: Optional[QuickActionButtons] = Field(default=None, description="Quick action buttons for the UI")

class Content(BaseModel):
    """Content of a chat message, can be text or structured data"""
    type: Literal["plain_text_answer", "formatted_text_answer", "ui_answer"] = Field(
        description="Type of content"
    )
    formatted_text_type: Optional[Literal["plain", "markdown", "html", "python", "ssml"]] = Field(
        description="Type of formatted text content. Used only if type is 'formatted_text_answer'"
    )
    plain_text_answer: Optional[str] = Field(default=None, description="Plain text answer content. If type is 'plain_text_answer', use this field to provide the content.")
    formatted_text_answer: Optional[str] = Field(default=None, description="Formatted text answer content. If type is 'formatted_text_answer', use this field to provide the content exactly as specified in 'formatted_text_type'.")
    ui_answer: Optional[UIAnswer] = Field(default=None, description="Generative UI answer content. If type is 'ui_answer', use this field to provide the content.")
    
MAIN_AGENT_PROMPT = """# User Context & Settings
- **User**: {user_name}
- **Location**: {default_city}, {default_country}
- **Current Date**: {current_date} ({current_weekday})
- **Current Time**: {current_time}
- **Timezone**: {user_timezone}
- **Language**: {language}
- **Currency**: {currency}
- **Date Format**: {date_format}
- **Time Format**: {time_format}
- **Measurement Units**: {measurement_units}
- **Regional Holidays**: {holidays}
- **Check Business Hours**: {check_open_now}
- **Preferred Transport**: {transport_modes}
- **Preferred Cuisine**: {cuisine_types}
- **Persona Style**: {persona}

# Current Environment
- **Weather**: {weather_status}
- **Sunrise**: {sunrise_time}
- **Sunset**: {sunset_time}

# Role
You are an expert UI/UX-focused AI assistant specializing in touchscreen interfaces.
Design responses that follow mobile-first principles with clear visual hierarchy and intuitive user flows.

# UI Design Philosophy
- **Progressive Disclosure**: Start with essential information, provide paths to details
- **Touch-Friendly**: Design for finger navigation with clear tap targets and logical grouping
- **Contextual Actions**: Every piece of information should suggest relevant next steps
- **Scannable Content**: Use visual hierarchy to support quick decision-making

# Response Architecture Strategy
1. **Open with Impact**: Begin with the most important answer or action the user needs
2. **Structure Information Flow**: 
   - Critical actions → Core information → Supporting details → Alternative options
   - Use cards for actionable content, text for explanations, tables for comparisons
3. **Anticipate Next Steps**: Always provide 2-3 logical next actions via buttons
4. **Visual Grouping**: Related content should be visually connected, separate concepts should have clear breaks

# UI Component Usage Rules
- **Cards**: Use for any actionable information or when user might want to interact. Don't use for specialized purposes below.
- **Location Cards**: For locations, addresses, places - never for abstract concepts  
- **Contact Cards**: Only for people, organizations with communication methods
- **Product Cards**: For physical/digital products with specifications and purchase research - use "show_details" frontend button
- **Movie/Series Cards**: For entertainment media - use "open_on_youtube_video" frontend button for trailers
- **Music Cards**: For music content - use "open_on_youtube_music" frontend button for listening
- **Article Cards**: For news, blog posts, written content - use "export_to_notes" frontend button for saving
- **Shopping List Cards**: For organized grocery/shopping lists - use "export_to_notes" frontend button for saving
- **Weather Cards**: For weather conditions, forecasts, and clothing recommendations
- **Tables**: For comparing items, structured data, or step-by-step processes
- **Charts**: For numerical data, trends, statistics, comparisons - provide complete Chart.js config as JSON string
- **Text Answers**: For explanations, context, or responses that don't need immediate action
- **Quick Action Buttons**: 2-3 buttons representing most likely user needs after seeing your response

# Chart Component Guidelines
When using charts, provide complete Chart.js configuration including:
- Proper chart type (bar, line, pie, doughnut, area, scatter)
- Data with labels and datasets with backgroundColor colors
- Mobile-optimized options: responsive: true, maintainAspectRatio: false
- Clear title and description explaining insights

# Mobile UX Best Practices
- Use active voice and action-oriented language
- Design for one-handed use with important actions easily reachable
- Consider user context: time pressure, mobility, distraction levels
"""


tools = [
    {"type": "web_search"},
    # pydantic_function_tool(GetWeather, name="get_weather"),
]


def chat():
    from app.agent.state import get_state
    
    # Get current application state
    state = get_state()
    
    # Format the prompt with all state variables
    formatted_prompt = MAIN_AGENT_PROMPT.format(
        user_name=state["user_name"],
        default_city=state["default_city"],
        default_country=state["default_country"],
        current_date=state["current_date"],
        current_weekday=state["current_weekday"],
        current_time=state["current_time"],
        user_timezone=state["user_timezone"],
        language=state["language"],
        currency=state["currency"],
        date_format=state["date_format"],
        time_format=state["time_format"],
        measurement_units=state["measurement_units"],
        holidays=state["commercial"]["holidays"],
        check_open_now=state["commercial"]["check_open_now"],
        transport_modes=", ".join(state["user_preferences"]["transport"]),
        cuisine_types=", ".join(state["user_preferences"]["cuisine"]),
        persona=state["persona"],
        weather_status=state["current_environment"]["weather"] or "Unknown",
        sunrise_time=state["current_environment"]["sunrise"] or "Unknown",
        sunset_time=state["current_environment"]["sunset"] or "Unknown"
    )
    
    system_prompt = {
        "role": "system",
        "content": formatted_prompt,
    }
    user_input = {
        "role": "user",
        "content": "Расскажи мне о ближайших итальянских ресторанах в ssml формате",
    }
    messages = [system_prompt, user_input]
    response = client.responses.parse(
        model="gpt-4.1",
        input=messages,
        # tools=tools,
        text_format=AgentResponse,
    )
    # print("Response:", response)
    result = response.output[0].content[0].parsed
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

def main():
    """Main entry point for the application"""
    chat()


if __name__ == "__main__":
    main()

# чтобы запустить:
# PYTHONPATH=. poetry run python app/response_understand.py
