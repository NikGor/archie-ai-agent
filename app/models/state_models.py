"""Pydantic models for user and application state."""

from pydantic import BaseModel, Field


class SpotifyPlaybackState(BaseModel):
    """Ambient Spotify playback context, injected into every LLM call."""

    is_playing: bool = Field(default=False, description="Is music currently playing")
    track_title: str | None = Field(default=None, description="Currently playing track title")
    track_artist: str | None = Field(default=None, description="Currently playing track artist")
    progress_seconds: int = Field(default=0, description="Current playback position in seconds")
    volume: int = Field(default=50, description="Volume level 0-100%")
    shuffle: bool = Field(default=False, description="Shuffle mode enabled")
    repeat: str = Field(default="off", description="Repeat mode: 'off', 'track', or 'context'")


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
    spotify: SpotifyPlaybackState | None = Field(
        default=None, description="Current Spotify playback state, if available"
    )
