"""Pydantic models for user and application state."""

from pydantic import BaseModel, Field


class UserState(BaseModel):
    """User state for prompt context."""

    user_name: str = Field(description="User's name")
    persona: str = Field(description="Agent persona")
    default_city: str = Field(description="User's default city")
    default_country: str = Field(description="User's default country")
    current_date: str = Field(description="Current date")
    current_time: str = Field(description="Current time")
    current_weekday: str = Field(description="Current weekday")
    user_timezone: str = Field(description="User's timezone")
    measurement_units: str = Field(description="Measurement units (metric/imperial)")
    language: str = Field(description="User's language")
    currency: str = Field(description="User's currency")
    date_format: str = Field(description="Date format preference")
    time_format: str = Field(description="Time format preference")
    commercial_holidays: str = Field(description="Commercial holidays region")
    commercial_check_open_now: bool = Field(description="Check if places are open now")
    transport_preferences: list[str] = Field(description="Preferred transport modes")
    cuisine_preferences: list[str] = Field(description="Preferred cuisines")
